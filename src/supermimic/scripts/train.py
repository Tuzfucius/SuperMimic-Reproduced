"""Train SuperMimic.  Example:

  uv run sm-train --run-name e1_walk --data.clips walk1_subject1 --env.num-envs 1024
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Under torchrun, pin each rank to its own GPU *before* CUDA/warp initialise: MuJoCo Warp and
# mjlab's simulation always run on the first visible device, so cfg.device alone is not enough.
# Skipped on a box with fewer GPUs than ranks (single-GPU smoke tests share cuda:0 via gloo).
def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


# A preset CUDA_VISIBLE_DEVICES list (e.g. "0,2,3" to skip a bad card) is indexed by LOCAL_RANK.
_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
if "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail) and (not _preset or len(_preset) > 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"
    # Give each rank its own slice of cores (the training loop is single-thread CPU bound;
    # sharing cores between ranks adds jitter that lockstep all-reduces turn into waiting).
    try:
        cores = sorted(os.sched_getaffinity(0))
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local
        if per >= 2:
            r = int(os.environ["LOCAL_RANK"])
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
    except (OSError, ValueError):
        pass

import tyro

from supermimic.config import TrainConfig, resolve_config
from supermimic.config_loader import config_dict, load_config


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    known, remaining = parser.parse_known_args()
    cfg = load_config(known.cfg, known.overrides)
    if remaining:
        # Tyro overlays explicit legacy flags on the resolved YAML object.
        cfg = tyro.cli(TrainConfig, default=cfg, args=remaining)
    cfg = resolve_config(cfg)
    cfg._source_yaml = known.cfg.resolve() if known.cfg else None
    if known.dry_run:
        import yaml

        print(yaml.safe_dump(config_dict(cfg), sort_keys=False))
        print(json.dumps({
            "dimensions": {
                "policy_observation": cfg.policy.history * 93,
                "goal": cfg.policy.goal_dim,
                "integral": cfg.policy.integral_dim,
                "actor_decoder_input": cfg.policy.latent_dim + cfg.policy.history * 93 + cfg.policy.integral_dim,
                "critic_integral": cfg.critic.integral_dim,
            },
            "integral_residual": {
                "enabled": cfg.integral.residual_enabled,
                "max_gain": cfg.integral.residual_max_gain,
                "learnable_parameters": 29 if cfg.integral.residual_enabled else 0,
            },
            "experiment_path": str(cfg.log_root / cfg.run_name),
            "validation": "configuration structure only; no GPU or simulator initialized",
        }, indent=2))
        return
    from supermimic.training.loop import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
