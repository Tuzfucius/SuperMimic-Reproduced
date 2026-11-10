"""Reproducibility metadata and collision-safe run directory setup."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml

from supermimic.config import TrainConfig
from supermimic.config_loader import config_dict


def git_info() -> dict[str, str | bool | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status or "clean"}


def create_run(cfg: TrainConfig) -> tuple[Path, dict]:
    path = cfg.log_root / cfg.run_name
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"experiment directory already exists and is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    resolved = config_dict(cfg)
    encoded = yaml.safe_dump(resolved, sort_keys=True).encode()
    metadata = {
        "experiment": cfg.run_name,
        "config_source": str(source) if (source := getattr(cfg, "_source_yaml", None)) else None,
        "git": git_info(),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "seed": cfg.seed,
        "dataset": {"npz_dir": str(cfg.data.npz_dir), "library": str(cfg.data.library) if cfg.data.library else None, "clips": cfg.data.clips, "manifest": str(cfg.data.manifest) if cfg.data.manifest else None},
        "device": cfg.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "started_at": time.time(),
        "state": "running",
        "checkpoint": None,
        "eval_metrics": "eval_metrics.json",
    }
    source = getattr(cfg, "_source_yaml", None)
    (path / "config.yaml").write_text(Path(source).read_text() if source else yaml.safe_dump(resolved, sort_keys=False))
    (path / "resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (path / "eval_metrics.json").write_text("{}\n")
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return path, metadata


def update_run(path: Path, metadata: dict, **updates) -> None:
    metadata.update(updates)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
