"""Reproducibility metadata and collision-safe run directory setup."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

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


def checkpoint_run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).resolve()
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _merge_defaults(defaults: dict, stored: dict) -> dict:
    merged = dict(defaults)
    for key, value in stored.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(mapping: dict, prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def validate_resume_config(cfg: TrainConfig, saved_config: dict) -> None:
    """Reject semantic drift while allowing logging limits and output location to change."""
    defaults = _normalize(config_dict(TrainConfig()))
    current = _normalize(config_dict(cfg))
    saved = _merge_defaults(defaults, _normalize(saved_config))
    current_flat, saved_flat = _flatten(current), _flatten(saved)
    allowed = {
        "run_name", "log_root", "resume", "init_from", "device",
        "schedule.log_interval", "schedule.max_hours", "schedule.total_iterations",
    }
    differences = [
        f"{key}: checkpoint={saved_flat.get(key)!r}, requested={current_flat.get(key)!r}"
        for key in sorted(set(current_flat) | set(saved_flat))
        if key not in allowed and current_flat.get(key) != saved_flat.get(key)
    ]
    if differences:
        detail = "\n  ".join(differences[:20])
        suffix = "\n  …" if len(differences) > 20 else ""
        raise ValueError("resume configuration conflicts with checkpoint:\n  " + detail + suffix)


def open_resume_run(cfg: TrainConfig, checkpoint: str | Path) -> tuple[Path, dict]:
    """Open an existing run without rewriting its original config or historical files."""
    path = checkpoint_run_dir(checkpoint)
    if not path.is_dir():
        raise FileNotFoundError(f"resume experiment directory does not exist: {path}")
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {
            "experiment": path.name,
            "git": git_info(),
            "config_sha256": None,
            "started_at": None,
            "state": "legacy",
            "checkpoint": str(checkpoint),
            "eval_metrics": "eval_metrics.json",
        }
    history = metadata.setdefault("resume_history", [])
    resume_number = len(history) + 1
    if metadata.get("evaluation_state") == "completed" or (path / "benchmark.json").exists():
        archive = path / "evaluation_history" / f"resume_{resume_number:03d}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("benchmark.json", "eval_metrics.json"):
            source = path / name
            if source.exists():
                shutil.copy2(source, archive / name)
        metadata["evaluation_state"] = "pending"
    effective = yaml.safe_dump(_normalize(config_dict(cfg)), sort_keys=True).encode()
    history.append({
        "at": time.time(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed": cfg.seed,
        "config_sha256": hashlib.sha256(effective).hexdigest(),
    })
    metadata.update({"state": "running", "resumed_at": time.time(), "resume_count": len(history)})
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return path, metadata


def update_run(path: Path, metadata: dict, **updates) -> None:
    metadata.update(updates)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
