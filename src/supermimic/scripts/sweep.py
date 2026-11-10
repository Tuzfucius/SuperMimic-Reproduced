"""Sequential YAML sweep runner for reproducible ablations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


def expand(spec: dict, cfg_path: Path) -> list[dict]:
    exps = spec.get("experiments", [])
    grid = spec.get("grid", {})
    grid_keys = list(grid)
    combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*(grid[k] for k in grid_keys))] if grid_keys else [{}]
    rows = []
    for exp in exps:
        path = (cfg_path.parent / exp.get("cfg", "../experiments/baseline.yaml")).resolve()
        for seed in exp.get("seeds", spec.get("seeds", [0])):
            for combo in combos:
                overrides = list(exp.get("set", []))
                overrides.extend(f"{k}={v}" for k, v in combo.items())
                overrides.append(f"seed={seed}")
                label = exp["id"]
                if combo:
                    label += "_" + "_".join(f"{k.split('.')[-1]}{v}" for k, v in combo.items())
                name = f"{label}_seed{seed}"
                rows.append({"id": exp["id"], "seed": seed, "name": name, "cfg": str(path), "set": overrides})
    return rows


def _eval_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    clips = int(summary.get("clips", 0))
    failed = int(summary.get("failed", 0))
    return {
        "mean_norm_return": summary.get("mean_norm_return"),
        "step_weighted_return": summary.get("step_weighted_norm_return"),
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
        if result.get("mean_norm_return") is not None:
            grouped.setdefault(result["id"], []).append(result)
    summary = {}
    for exp_id, rows in grouped.items():
        values = {}
        for key in ("mean_norm_return", "step_weighted_return", "failure_rate", "failed"):
            samples = [float(row[key]) for row in rows if row.get(key) is not None]
            values[f"{key}_mean"] = statistics.mean(samples) if samples else None
            values[f"{key}_std"] = statistics.stdev(samples) if len(samples) > 1 else (0.0 if samples else None)
        values["seeds_evaluated"] = len(rows)
        summary[exp_id] = values
    return summary


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
        sys.executable,
        str(repo_root / "scripts" / "eval_benchmark.py"),
        "--ckpt", str(checkpoint.resolve()),
        "--npz-dir", str(dataset),
        "--num-envs", str(num_envs),
        "--device", str(device),
        "--out", str(benchmark_path),
    ]
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
        print(f"{row['name']}: {row['cfg']} " + " ".join(f"--set {x}" for x in row["set"]))
    if args.dry_run:
        print(f"matrix: {len(rows)} runs; execution is sequential; output={root}")
        if eval_enabled:
            print(f"benchmark: {eval_cfg.get('dataset')} with {eval_cfg.get('num_envs', 256)} envs after successful training")
        return
    results = []
    for row in rows:
        run_dir = root / row["name"]
        meta_path = run_dir / "metadata.json"
        metadata = {}
        should_train = True
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
                if existing.get("state") == "completed":
                    metadata = existing
                    should_train = False
                    eval_done = (
                        existing.get("evaluation_state") == "completed"
                        and (run_dir / "eval_metrics.json").is_file()
                        and (run_dir / "benchmark.json").is_file()
                    )
                    if not eval_enabled or eval_done:
                        metrics = _eval_summary(run_dir / "benchmark.json") if eval_done else {}
                        results.append({
                            **row,
                            "status": "skipped_completed",
                            "training_status": "completed",
                            "train_seconds": existing.get("elapsed_s"),
                            "checkpoint": existing.get("checkpoint"),
                            **metrics,
                        })
                        continue
                else:
                    results.append({**row, "status": "skipped_incomplete", "training_status": existing.get("state"), "returncode": None})
                    continue
            except (OSError, json.JSONDecodeError):
                results.append({**row, "status": "skipped_incomplete", "training_status": "invalid_metadata", "returncode": None})
                continue
        elif run_dir.exists() and any(run_dir.iterdir()):
            results.append({**row, "status": "skipped_incomplete", "training_status": "untracked_existing_directory", "returncode": None})
            continue

        train_seconds = metadata.get("elapsed_s")
        checkpoint = Path(metadata["checkpoint"]) if metadata.get("checkpoint") else None
        if should_train:
            cmd = [sys.executable, "-m", "supermimic.scripts.train", "--cfg", row["cfg"], "--set", f"run_name={row['name']}", "--set", f"log_root={root}"]
            for override in row["set"]:
                cmd.extend(["--set", override])
            env = dict(os.environ)
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.gpu
            started = time.time()
            proc = subprocess.run(cmd, env=env, check=False)
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
            checkpoint = Path(metadata.get("checkpoint") or run_dir / "checkpoints" / "ckpt_latest.pt")
            metadata.update({"elapsed_s": train_seconds, "evaluation_state": "pending" if eval_enabled else None})
            meta_path.write_text(json.dumps(metadata, indent=2))

        if checkpoint is None:
            checkpoint = Path(metadata.get("checkpoint") or run_dir / "checkpoints" / "ckpt_latest.pt")
        if eval_enabled:
            metadata["evaluation_state"] = "running"
            meta_path.write_text(json.dumps(metadata, indent=2))
            try:
                checkpoint = _select_checkpoint(run_dir, metadata, eval_cfg)
                metrics, eval_error = _benchmark(run_dir, checkpoint, eval_cfg, args.gpu)
            except OSError as exc:
                metrics, eval_error = None, str(exc)
            if eval_error:
                metadata.update({"evaluation_state": "failed", "evaluation_error": eval_error})
                meta_path.write_text(json.dumps(metadata, indent=2))
                results.append({**row, "status": "training_completed_eval_failed", "training_status": "completed", "train_seconds": train_seconds, "checkpoint": str(checkpoint), "evaluation_error": eval_error})
                continue
            metadata.update({"evaluation_state": "completed", "evaluation_error": None, "eval_metrics": str(run_dir / "eval_metrics.json"), "benchmark": str(run_dir / "benchmark.json")})
            meta_path.write_text(json.dumps(metadata, indent=2))
        else:
            metrics = {}
        results.append({
            **row,
            "status": "completed",
            "training_status": "completed",
            "train_seconds": train_seconds,
            "checkpoint": str(checkpoint),
            **metrics,
        })
    root.mkdir(parents=True, exist_ok=True)
    payload = {"runs": results, "by_experiment": _aggregate(results)}
    (root / "summary.json").write_text(json.dumps(payload, indent=2))
    keys = ["id", "seed", "name", "status", "training_status", "train_seconds", "checkpoint", "mean_norm_return", "step_weighted_return", "failed", "clips", "failure_rate"]
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
