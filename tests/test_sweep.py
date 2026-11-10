import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from supermimic.scripts.sweep import _eval_summary, expand, main


def _write_sweep(tmp_path, *, evaluation=None, tau=2.0):
    train_cfg = tmp_path / "integral.yaml"
    dataset = tmp_path / "motions"
    dataset.mkdir(exist_ok=True)
    (dataset / "clip.npz").write_bytes(b"motion")
    train_cfg.write_text(yaml.safe_dump({
        "integral": {"enabled": True, "tau_s": tau},
        "data": {"npz_dir": str(dataset)},
    }))
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "eval", "output_root": str(tmp_path / "runs"), "seeds": [0],
        "evaluation": evaluation or {"enabled": False},
        "experiments": [{"id": "I1", "cfg": "integral.yaml"}],
    }))
    return cfg_path, train_cfg


def _install_clean_git(monkeypatch):
    from supermimic.scripts import sweep

    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "abc123", "dirty": False, "status": "clean"})
    return sweep


def test_eval_summary_preserves_metrics():
    path = Path("/tmp/benchmark-test.json")
    path.write_text(json.dumps({"summary": {
        "clips": 4, "failed": 1, "mean_norm_return": 0.7, "step_weighted_norm_return": 0.6,
    }}))
    try:
        metrics = _eval_summary(path)
    finally:
        path.unlink(missing_ok=True)
    assert metrics == {
        "mean_norm_return": 0.7, "step_weighted_return": 0.6,
        "failed": 1, "clips": 4, "failure_rate": 0.25,
    }


def test_eval_summary_rejects_missing_or_nonfinite_metrics(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"summary": {"clips": 1, "failed": 0, "mean_norm_return": float("nan"), "step_weighted_norm_return": 0.5}}))
    with pytest.raises(ValueError, match="not finite"):
        _eval_summary(path)
    path.write_text(json.dumps({"summary": {"clips": 1, "failed": 0}}))
    with pytest.raises(ValueError, match="missing"):
        _eval_summary(path)


def test_sweep_dry_run_does_not_check_dataset_or_launch_process(tmp_path, monkeypatch, capsys):
    cfg_path, _ = _write_sweep(tmp_path, evaluation={
        "enabled": True, "dataset": str(tmp_path / "missing"), "num_envs": 8,
    })
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path), "--dry-run"])
    real_run = subprocess.run

    def reject_non_git_process(command, *args, **kwargs):
        if command and command[0] == "git":
            return real_run(command, *args, **kwargs)
        raise AssertionError("launched process")

    monkeypatch.setattr("supermimic.scripts.sweep.subprocess.run", reject_non_git_process)
    main()
    output = capsys.readouterr().out
    assert "matrix: 1 runs" in output
    assert "variant=" in output and "benchmark:" in output


def test_sweep_rejects_unimplemented_evaluation_parameters(tmp_path, monkeypatch):
    cfg_path, _ = _write_sweep(tmp_path, evaluation={"enabled": False, "max_steps": 10})
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path), "--dry-run"])
    with pytest.raises(ValueError, match="unsupported evaluation parameter"):
        main()


def test_sweep_reuses_training_only_when_eval_identity_matches(tmp_path, monkeypatch):
    sweep = _install_clean_git(monkeypatch)
    dataset = tmp_path / "npz"
    dataset.mkdir()
    (dataset / "clip.npz").write_bytes(b"data-a")
    evaluation = {"enabled": True, "dataset": str(dataset), "num_envs": 2}
    cfg_path, _ = _write_sweep(tmp_path, evaluation=evaluation)
    output = tmp_path / "runs" / "eval"
    calls = []
    train_calls = 0
    eval_calls = 0
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        nonlocal train_calls, eval_calls
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        calls.append(command)
        if "supermimic.scripts.train" in command:
            train_calls += 1
            run_name = command[command.index("--set") + 1].split("=", 1)[1]
            log_root = Path(command[command.index("--set", command.index("--set") + 1) + 1].split("=", 1)[1])
            run_dir = log_root / run_name
            checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"weights")
            (run_dir / "metadata.json").write_text(json.dumps({
                "state": "completed", "checkpoint": str(checkpoint), "elapsed_s": 12.0,
            }))
        else:
            eval_calls += 1
            out_path = Path(command[command.index("--out") + 1])
            out_path.write_text(json.dumps({"summary": {
                "clips": 2, "failed": 1, "mean_norm_return": 0.75, "step_weighted_norm_return": 0.7,
            }}))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert (output / "summary.json").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "aggregate.json").is_file()
    assert (output / "aggregate.csv").is_file()
    run_dir = next(p for p in output.iterdir() if p.is_dir())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["evaluation_state"] == "completed"
    assert metadata["training_identity_sha256"]
    assert metadata["evaluation_identity_sha256"]
    assert json.loads((run_dir / "eval_metrics.json").read_text())["failure_rate"] == 0.5
    aggregate = json.loads((output / "aggregate.json").read_text())["groups"][0]
    assert aggregate["seeds_expected"] == aggregate["seeds_evaluated"] == 1
    assert aggregate["mean_norm_return_std"] is None
    assert train_calls == eval_calls == 1

    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert train_calls == eval_calls == 1

    # A changed evaluation protocol reuses the checkpoint but produces a new benchmark.
    changed = {**evaluation, "num_envs": 4}
    cfg_path.write_text(yaml.safe_dump({
        "name": "eval", "output_root": str(tmp_path / "runs"), "seeds": [0],
        "evaluation": changed,
        "experiments": [{"id": "I1", "cfg": "integral.yaml"}],
    }))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert train_calls == 1 and eval_calls == 2
    assert list((run_dir / "evaluation_history").glob("*/benchmark.json"))

    # Missing output is also a cache miss for evaluation, never for training.
    (run_dir / "benchmark.json").unlink()
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert train_calls == 1 and eval_calls == 3


def test_training_success_evaluation_failure_retries_only_evaluation(tmp_path, monkeypatch):
    sweep = _install_clean_git(monkeypatch)
    dataset = tmp_path / "npz"
    dataset.mkdir()
    (dataset / "clip.npz").write_bytes(b"data")
    cfg_path, _ = _write_sweep(tmp_path, evaluation={"enabled": True, "dataset": str(dataset)})
    run_dir = None
    train_calls = 0
    eval_calls = 0
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        nonlocal train_calls, eval_calls, run_dir
        if command and command[0] == "git":
            return real_run(command, **kwargs)
        if "supermimic.scripts.train" in command:
            train_calls += 1
            run_name = command[command.index("--set") + 1].split("=", 1)[1]
            log_root = Path(command[command.index("--set", command.index("--set") + 1) + 1].split("=", 1)[1])
            run_dir = log_root / run_name
            checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"weights")
            (run_dir / "metadata.json").write_text(json.dumps({"state": "completed", "checkpoint": str(checkpoint)}))
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        eval_calls += 1
        if eval_calls == 1:
            return SimpleNamespace(returncode=1, stderr="simulated benchmark failure", stdout="")
        out_path = Path(command[command.index("--out") + 1])
        out_path.write_text(json.dumps({"summary": {
            "clips": 1, "failed": 0, "mean_norm_return": 0.8, "step_weighted_norm_return": 0.8,
        }}))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert json.loads((run_dir / "metadata.json").read_text())["evaluation_state"] == "failed"

    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["evaluation_state"] == "completed"
    assert train_calls == 1 and eval_calls == 2


def test_sweep_does_not_reuse_corrupt_or_legacy_identity_metadata(tmp_path, monkeypatch):
    _install_clean_git(monkeypatch)
    cfg_path, _ = _write_sweep(tmp_path)
    output = tmp_path / "runs" / "eval"
    row_name = expand({"experiments": [{"id": "I1", "cfg": "integral.yaml", "seeds": [0]}]}, cfg_path)[0]["name"]
    run_dir = output / row_name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text("not-json")
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    result = json.loads((output / "summary.json").read_text())["runs"][0]
    assert result["status"] == "skipped_unverified_metadata"

    (run_dir / "metadata.json").write_text(json.dumps({"state": "completed", "checkpoint": "ckpt.pt"}))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    result = json.loads((output / "summary.json").read_text())["runs"][0]
    assert result["status"] == "skipped_unverified_legacy"
