"""Sequential YAML sweep runner with identity-checked reuse and seed aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from supermimic.config_loader import config_dict, load_config
from supermimic.experiment import git_info

EVALUATION_PROTOCOL_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _dataset_identity(value: str | Path | None) -> dict:
    if value is None:
        return {"path": None, "file_count": 0, "manifest_sha256": _hash([])}
    path = Path(value).expanduser().resolve()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.npz"))
        if not files:
            files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        return {"path": str(path), "missing": True, "file_count": 0, "manifest_sha256": _hash([])}
    entries = []
    for file in files:
        stat = file.stat()
        entries.append({
            "path": str(file.relative_to(path)) if path.is_dir() else file.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {"path": str(path), "file_count": len(entries), "manifest_sha256": _hash(entries)}


def _clean_git_identity() -> dict:
    info = git_info()
    return {"commit": info.get("commit"), "dirty": bool(info.get("dirty")), "status": info.get("status", "unknown")}


def _train_identity(row: dict, output_root: Path) -> tuple[dict, str]:
    overrides = [f"run_name={row['name']}", f"log_root={output_root}", *row["set"]]
    cfg = load_config(row["cfg"], overrides)
    resolved = config_dict(cfg)
    # Run placement and display name do not change the computation being trained.
    resolved.pop("run_name", None)
    resolved.pop("log_root", None)
    resolved.pop("resume", None)
    resolved.pop("init_from", None)
    data = resolved.get("data", {})
    dataset = {
        "npz_dir": _dataset_identity(data.get("npz_dir")),
        "library": _dataset_identity(data.get("library")) if data.get("library") else None,
        "manifest": _dataset_identity(data.get("manifest")) if data.get("manifest") else None,
    }
    identity = {
        "experiment_id": row["id"],
        "variant_id": row["variant_id"],
        "params": row["params"],
        "seed": row["seed"],
        "config": resolved,
        "dataset": dataset,
        "code": _clean_git_identity(),
    }
    return identity, _hash(identity)


def _checkpoint_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "sha256": digest.hexdigest()}


def _evaluation_identity(checkpoint: Path, eval_cfg: dict) -> tuple[dict, str]:
    dataset = _dataset_identity(eval_cfg.get("dataset"))
    params = {key: value for key, value in eval_cfg.items() if key not in {"enabled", "dataset", "checkpoint"}}
    auxiliary_inputs = {
        key: _dataset_identity(eval_cfg[key])
        for key in ("manifest", "clip_list")
        if eval_cfg.get(key)
    }
    identity = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "dataset": dataset,
        "auxiliary_inputs": auxiliary_inputs,
        "parameters": params,
        "checkpoint_selector": eval_cfg.get("checkpoint", "latest"),
        "checkpoint": _checkpoint_identity(checkpoint),
        "code": _clean_git_identity(),
    }
    return identity, _hash(identity)


def expand(spec: dict, cfg_path: Path) -> list[dict]:
    exps = spec.get("experiments", [])
    grid = spec.get("grid", {})
    grid_keys = sorted(grid)
    combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*(grid[k] for k in grid_keys))] if grid_keys else [{}]
    rows = []
    for exp in exps:
        path = (cfg_path.parent / exp.get("cfg", "../experiments/baseline.yaml")).resolve()
        for seed in exp.get("seeds", spec.get("seeds", [0])):
            for combo in combos:
                variant_id = _hash(combo)[:12]
                overrides = list(exp.get("set", []))
                overrides.extend(f"{k}={v}" for k, v in combo.items())
                overrides.append(f"seed={seed}")
                name = f"{exp['id']}_v{variant_id}_seed{seed}"
                rows.append({
                    "id": exp["id"], "seed": seed, "variant_id": variant_id,
                    "params": combo, "name": name, "cfg": str(path), "set": overrides,
                })
    return rows


def _eval_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    clips = int(summary.get("clips", 0))
    failed = int(summary.get("failed", 0))
    if clips < 1 or failed < 0 or failed > clips:
        raise ValueError(f"invalid benchmark clip counts: clips={clips}, failed={failed}")
    mean_return = summary.get("mean_norm_return")
    weighted_return = summary.get("step_weighted_norm_return")
    if mean_return is None or weighted_return is None:
        raise ValueError("benchmark result is missing one or more aggregate returns")
    for name, value in (("mean_norm_return", mean_return), ("step_weighted_norm_return", weighted_return)):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"benchmark metric {name} is not finite: {value!r}")
    return {
        "mean_norm_return": float(mean_return),
        "step_weighted_return": float(weighted_return),
        "failed": failed,
        "clips": clips,
        "failure_rate": failed / clips if clips else None,
    }


def _select_checkpoint(run_dir: Path, metadata: dict, eval_cfg: dict) -> Path:
    selected = eval_cfg.get("checkpoint", "latest")
    if selected == "latest":
        checkpoint = Path(metadata.get("checkpoint") or run_dir / "checkpoints" / "ckpt_latest.pt")
    else:
        candidate = Path(selected)
        checkpoint = candidate if candidate.is_absolute() else run_dir / "checkpoints" / candidate
    if not checkpoint.is_file():
        raise FileNotFoundError(f"evaluation checkpoint does not exist: {checkpoint}")
    return checkpoint


def _aggregate(results: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        if result.get("id") is None or result.get("variant_id") is None:
            continue
        key = f"{result['id']}::{result['variant_id']}"
        grouped.setdefault(key, []).append(result)
    summary = {}
    for key, rows in grouped.items():
        expected = sorted({row["seed"] for row in rows})
        evaluated = {}
        for row in rows:
            if row.get("mean_norm_return") is not None:
                evaluated[row["seed"]] = row
        valid_rows = [evaluated[seed] for seed in sorted(evaluated)]
        values = {}
        for metric in ("mean_norm_return", "step_weighted_return", "failure_rate", "failed"):
            samples = [float(row[metric]) for row in valid_rows if row.get(metric) is not None]
            values[f"{metric}_mean"] = statistics.mean(samples) if samples else None
            values[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else None
        first = rows[0]
        values.update({
            "id": first["id"], "variant_id": first["variant_id"], "params": first.get("params", {}),
            "seeds_expected": len(expected), "seeds_evaluated": len(valid_rows),
            "missing_seeds": [seed for seed in expected if seed not in evaluated],
        })
        summary[key] = values
    return summary


def _archive_evaluation(run_dir: Path, old_hash: str | None) -> None:
    files = [run_dir / "benchmark.json", run_dir / "eval_metrics.json"]
    existing = [path for path in files if path.is_file()]
    if not existing:
        return
    label = old_hash[:12] if old_hash else "unverified"
    archive = run_dir / "evaluation_history" / label
    suffix = 1
    while archive.exists():
        suffix += 1
        archive = run_dir / "evaluation_history" / f"{label}_{suffix:02d}"
    archive.mkdir(parents=True)
    for path in existing:
        shutil.copy2(path, archive / path.name)


def _benchmark(run_dir: Path, checkpoint: Path, eval_cfg: dict, gpu: str | None) -> tuple[dict | None, str | None]:
    dataset = Path(eval_cfg["dataset"]).resolve()
    num_envs = int(eval_cfg.get("num_envs", 256))
    benchmark_path = run_dir / "benchmark.json"
    metrics_path = run_dir / "eval_metrics.json"
    resolved_path = run_dir / "resolved.yaml"
    train_cfg = yaml.safe_load(resolved_path.read_text()) if resolved_path.exists() else {}
    device = eval_cfg.get("device", train_cfg.get("device", "cuda:0"))
    repo_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable, str(repo_root / "scripts" / "eval_benchmark.py"),
        "--ckpt", str(checkpoint.resolve()), "--npz-dir", str(dataset),
        "--num-envs", str(num_envs), "--device", str(device), "--out", str(benchmark_path),
    ]
    if eval_cfg.get("clips"):
        command.extend(["--clips", *map(str, eval_cfg["clips"])])
    if eval_cfg.get("clip_list"):
        command.extend(["--clip-list", str(Path(eval_cfg["clip_list"]).resolve())])
    if eval_cfg.get("manifest"):
        command.extend(["--manifest", str(Path(eval_cfg["manifest"]).resolve())])
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True, check=False)
    if proc.returncode:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"benchmark exited {proc.returncode}"
        return None, detail
    try:
        summary = _eval_summary(benchmark_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"benchmark result is invalid: {exc}"
    metrics_path.write_text(json.dumps(summary, indent=2))
    return summary, None


def _write_summaries(root: Path, results: list[dict]) -> None:
    aggregates = _aggregate(results)
    payload = {"runs": results, "by_experiment": aggregates, "by_experiment_variant": aggregates}
    (root / "summary.json").write_text(json.dumps(payload, indent=2))
    aggregate_payload = {"groups": list(aggregates.values())}
    (root / "aggregate.json").write_text(json.dumps(aggregate_payload, indent=2))
    run_fields = [
        "id", "variant_id", "params", "seed", "name", "status", "training_status",
        "train_seconds", "training_identity_sha256", "evaluation_identity_sha256", "checkpoint",
        "mean_norm_return", "step_weighted_return", "failed", "clips", "failure_rate",
    ]
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "params": _canonical(row.get("params", {}))})
    aggregate_fields = [
        "id", "variant_id", "params", "seeds_expected", "seeds_evaluated", "missing_seeds",
        "mean_norm_return_mean", "mean_norm_return_std", "step_weighted_return_mean",
        "step_weighted_return_std", "failure_rate_mean", "failure_rate_std", "failed_mean", "failed_std",
    ]
    with (root / "aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields, extrasaction="ignore")
        writer.writeheader()
        for row in aggregates.values():
            writer.writerow({
                **row, "params": _canonical(row["params"]), "missing_seeds": _canonical(row["missing_seeds"]),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()
    spec = yaml.safe_load(args.cfg.read_text()) or {}
    cfg_path = args.cfg.resolve()
    rows = expand(spec, cfg_path)
    root = Path(spec.get("output_root", "runs"))
    group = spec.get("name", args.cfg.stem)
    root = (root / group).resolve()
    eval_cfg = spec.get("evaluation", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", False))
    allowed_eval_keys = {"enabled", "dataset", "num_envs", "device", "checkpoint", "clips", "clip_list", "manifest"}
    unknown_eval_keys = set(eval_cfg) - allowed_eval_keys
    if unknown_eval_keys:
        raise ValueError(f"unsupported evaluation parameter(s): {', '.join(sorted(unknown_eval_keys))}")
    if eval_enabled and not args.dry_run:
        dataset_value = eval_cfg.get("dataset")
        if not dataset_value:
            raise ValueError("evaluation.enabled requires evaluation.dataset")
        dataset = Path(dataset_value)
        if not dataset.exists() or not dataset.is_dir():
            raise FileNotFoundError(f"evaluation dataset directory does not exist: {dataset}")
        if not any(dataset.rglob("*.npz")):
            raise FileNotFoundError(f"evaluation dataset contains no .npz clips: {dataset}")
    for row in rows:
        print(f"{row['name']}: {row['cfg']} variant={row['variant_id']} params={_canonical(row['params'])} "
              + " ".join(f"--set {value}" for value in row["set"]))
    if args.dry_run:
        print(f"matrix: {len(rows)} runs; execution is sequential; output={root}")
        if eval_enabled:
            print(f"benchmark: {eval_cfg.get('dataset')} with {eval_cfg.get('num_envs', 256)} envs after successful training")
        return

    root.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        run_dir = root / row["name"]
        meta_path = run_dir / "metadata.json"
        requested_identity, requested_hash = _train_identity(row, root)
        metadata: dict = {}
        should_train = True
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                results.append({**row, "status": "skipped_unverified_metadata", "training_status": "invalid_metadata", "identity_error": str(exc)})
                continue
            if existing.get("state") != "completed":
                results.append({**row, "status": "skipped_incomplete", "training_status": existing.get("state", "unknown")})
                continue
            saved_hash = existing.get("training_identity_sha256")
            if not saved_hash:
                results.append({**row, "status": "skipped_unverified_legacy", "training_status": "completed", "identity_error": "training identity is absent"})
                continue
            if saved_hash != requested_hash:
                raise ValueError(
                    f"training identity mismatch for existing run {run_dir}; choose a new sweep name/output directory "
                    f"(stored={saved_hash}, requested={requested_hash})"
                )
            if requested_identity["code"]["dirty"]:
                raise ValueError(f"cannot reuse {run_dir} while the Git worktree is dirty")
            metadata = existing
            should_train = False
            try:
                checkpoint = _select_checkpoint(run_dir, metadata, eval_cfg if eval_enabled else {})
            except OSError as exc:
                results.append({**row, "status": "skipped_incomplete", "training_status": "completed", "identity_error": str(exc)})
                continue
            if not eval_enabled:
                results.append({
                    **row, "status": "skipped_completed", "training_status": "completed",
                    "train_seconds": metadata.get("elapsed_s"), "checkpoint": str(checkpoint),
                    "training_identity_sha256": requested_hash,
                })
                continue
            eval_identity, eval_hash = _evaluation_identity(checkpoint, eval_cfg)
            result_files_exist = (run_dir / "eval_metrics.json").is_file() and (run_dir / "benchmark.json").is_file()
            eval_done = (
                metadata.get("evaluation_state") == "completed"
                and metadata.get("evaluation_identity_sha256") == eval_hash
                and result_files_exist
            )
            if eval_done:
                try:
                    metrics = _eval_summary(run_dir / "benchmark.json")
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    eval_done = False
            if eval_done:
                results.append({
                    **row, "status": "skipped_completed", "training_status": "completed",
                    "train_seconds": metadata.get("elapsed_s"), "checkpoint": str(checkpoint),
                    "training_identity_sha256": requested_hash, "evaluation_identity_sha256": eval_hash,
                    **metrics,
                })
                continue
            old_eval_hash = metadata.get("evaluation_identity_sha256")
            _archive_evaluation(run_dir, old_eval_hash)
            metadata["evaluation_identity"] = eval_identity
            metadata["evaluation_identity_sha256"] = eval_hash
            metadata["evaluation_state"] = "pending"
            meta_path.write_text(json.dumps(metadata, indent=2))
        elif run_dir.exists() and any(run_dir.iterdir()):
            results.append({**row, "status": "skipped_unverified_directory", "training_status": "untracked_existing_directory"})
            continue

        train_seconds = metadata.get("elapsed_s")
        checkpoint = Path(metadata["checkpoint"]) if metadata.get("checkpoint") else None
        if should_train:
            command = [
                sys.executable, "-m", "supermimic.scripts.train", "--cfg", row["cfg"],
                "--set", f"run_name={row['name']}", "--set", f"log_root={root}",
            ]
            for override in row["set"]:
                command.extend(["--set", override])
            env = dict(os.environ)
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.gpu
            started = time.time()
            proc = subprocess.run(command, env=env, check=False)
            train_seconds = time.time() - started
            if proc.returncode:
                run_dir.mkdir(parents=True, exist_ok=True)
                failure = {**row, "status": "training_failed", "training_status": "failed", "returncode": proc.returncode, "train_seconds": train_seconds}
                (run_dir / "failure.json").write_text(json.dumps(failure, indent=2))
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                    metadata.update({"state": "failed", "finished_at": time.time(), "error_returncode": proc.returncode})
                    meta_path.write_text(json.dumps(metadata, indent=2))
                results.append(failure)
                continue
            if not meta_path.exists():
                results.append({**row, "status": "training_failed", "training_status": "missing_metadata", "returncode": proc.returncode, "train_seconds": train_seconds})
                continue
            metadata = json.loads(meta_path.read_text())
            if metadata.get("state") != "completed":
                results.append({**row, "status": "training_failed", "training_status": metadata.get("state", "unknown"), "train_seconds": train_seconds})
                continue
            checkpoint = Path(metadata.get("checkpoint") or run_dir / "checkpoints" / "ckpt_latest.pt")
            metadata.update({
                "elapsed_s": train_seconds,
                "training_identity": requested_identity,
                "training_identity_sha256": requested_hash,
                "evaluation_state": "pending" if eval_enabled else None,
            })
            meta_path.write_text(json.dumps(metadata, indent=2))

        if checkpoint is None:
            checkpoint = Path(metadata.get("checkpoint") or run_dir / "checkpoints" / "ckpt_latest.pt")
        if eval_enabled:
            try:
                checkpoint = _select_checkpoint(run_dir, metadata, eval_cfg)
                eval_identity, eval_hash = _evaluation_identity(checkpoint, eval_cfg)
            except OSError as exc:
                metrics, eval_error, eval_hash, eval_identity = None, str(exc), None, None
            else:
                metadata.update({
                    "evaluation_state": "running", "evaluation_identity": eval_identity,
                    "evaluation_identity_sha256": eval_hash,
                })
                meta_path.write_text(json.dumps(metadata, indent=2))
                metrics, eval_error = _benchmark(run_dir, checkpoint, eval_cfg, args.gpu)
            if eval_error:
                metadata.update({"evaluation_state": "failed", "evaluation_error": eval_error})
                meta_path.write_text(json.dumps(metadata, indent=2))
                results.append({
                    **row, "status": "training_completed_eval_failed", "training_status": "completed",
                    "train_seconds": train_seconds, "checkpoint": str(checkpoint),
                    "training_identity_sha256": requested_hash, "evaluation_identity_sha256": eval_hash,
                    "evaluation_error": eval_error,
                })
                continue
            metadata.update({
                "evaluation_state": "completed", "evaluation_error": None,
                "eval_metrics": str(run_dir / "eval_metrics.json"), "benchmark": str(run_dir / "benchmark.json"),
                "evaluation_identity": eval_identity, "evaluation_identity_sha256": eval_hash,
            })
            meta_path.write_text(json.dumps(metadata, indent=2))
        else:
            metrics = {}
        results.append({
            **row, "status": "completed", "training_status": "completed",
            "train_seconds": train_seconds, "checkpoint": str(checkpoint),
            "training_identity_sha256": requested_hash,
            "evaluation_identity_sha256": metadata.get("evaluation_identity_sha256"),
            **metrics,
        })
    _write_summaries(root, results)


if __name__ == "__main__":
    main()
