import copy
import json
from dataclasses import asdict

import pytest

from supermimic.config import TrainConfig, resolve_config
from supermimic.experiment import create_run, open_resume_run, validate_resume_config


def test_new_run_collision_and_resume_preserves_history(tmp_path):
    cfg = resolve_config(TrainConfig(run_name="exp", log_root=tmp_path))
    run_dir, _ = create_run(cfg)
    original_config = (run_dir / "config.yaml").read_text()
    (run_dir / "benchmark.json").write_text('{"summary": {"mean_norm_return": 0.5}}')
    (run_dir / "eval_metrics.json").write_text('{"mean_norm_return": 0.5}')
    (run_dir / "checkpoints").mkdir()
    checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")

    with pytest.raises(FileExistsError):
        create_run(cfg)
    reopened, metadata = open_resume_run(cfg, checkpoint)
    assert reopened == run_dir
    assert (run_dir / "config.yaml").read_text() == original_config
    assert json.loads((run_dir / "eval_metrics.json").read_text())["mean_norm_return"] == 0.5
    archived = run_dir / "evaluation_history" / "resume_001"
    assert json.loads((archived / "benchmark.json").read_text())["summary"]["mean_norm_return"] == 0.5
    assert json.loads((archived / "eval_metrics.json").read_text())["mean_norm_return"] == 0.5
    assert metadata["state"] == "running"
    assert metadata["resume_count"] == 1


def test_resume_config_strict_except_logging_limits(tmp_path):
    saved = resolve_config(TrainConfig(run_name="exp", log_root=tmp_path))
    checkpoint_config = asdict(saved)
    current = copy.deepcopy(saved)
    current.schedule.log_interval = 5
    current.schedule.max_hours = 2.0
    current.schedule.total_iterations += 100
    validate_resume_config(resolve_config(current), checkpoint_config)

    current = copy.deepcopy(saved)
    current.integral.tau_s = 4.0
    with pytest.raises(ValueError, match="integral.tau_s"):
        validate_resume_config(resolve_config(current), checkpoint_config)

    current = copy.deepcopy(saved)
    current.policy.history = 8
    with pytest.raises(ValueError, match="policy.history"):
        validate_resume_config(resolve_config(current), checkpoint_config)
