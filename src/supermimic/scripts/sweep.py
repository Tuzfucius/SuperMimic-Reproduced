"""Sequential YAML sweep runner for reproducible ablations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()
    spec = yaml.safe_load(args.cfg.read_text()) or {}
    rows = expand(spec, args.cfg.resolve())
    root = Path(spec.get("output_root", "runs"))
    group = spec.get("name", args.cfg.stem)
    root = root / group
    for row in rows:
        print(f"{row['name']}: {row['cfg']} " + " ".join(f"--set {x}" for x in row["set"]))
    if args.dry_run:
        print(f"matrix: {len(rows)} runs; execution is sequential; output={root}")
        return
    results = []
    for row in rows:
        run_dir = root / row["name"]
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
                if existing.get("state") == "completed":
                    results.append({**row, "status": "skipped_completed", "returncode": 0, "checkpoint": existing.get("checkpoint")})
                    continue
                results.append({**row, "status": "skipped_incomplete", "returncode": None})
                continue
            except (OSError, json.JSONDecodeError):
                pass
        cmd = [sys.executable, "-m", "supermimic.scripts.train", "--cfg", row["cfg"], "--set", f"run_name={row['name']}", "--set", f"log_root={root}"]
        for override in row["set"]:
            cmd.extend(["--set", override])
        env = dict(os.environ)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        started = time.time()
        proc = subprocess.run(cmd, env=env, check=False)
        status = "completed" if proc.returncode == 0 else "failed"
        results.append({**row, "status": status, "returncode": proc.returncode, "elapsed_s": time.time() - started})
        if proc.returncode:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.json").write_text(json.dumps(results[-1], indent=2))
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                    metadata.update({"state": "failed", "finished_at": time.time(), "error_returncode": proc.returncode})
                    meta_path.write_text(json.dumps(metadata, indent=2))
                except (OSError, json.JSONDecodeError):
                    pass
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(results, indent=2))
    keys = ["id", "seed", "name", "status", "returncode", "elapsed_s", "checkpoint"]
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
