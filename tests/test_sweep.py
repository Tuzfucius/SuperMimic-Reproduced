import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from supermimic.scripts.sweep import _aggregate, _eval_summary, main


def test_benchmark_summary_and_across_seed_aggregate(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"summary": {"clips": 4, "failed": 1, "mean_norm_return": 0.7, "step_weighted_norm_return": 0.6}}))
    metrics = _eval_summary(path)
    assert metrics == {
        "mean_norm_return": 0.7,
        "step_weighted_return": 0.6,
        "failed": 1,
        "clips": 4,
        "failure_rate": 0.25,
    }
    rows = [
        {"id": "I1", "seed": 0, **metrics},
        {"id": "I1", "seed": 1, **metrics, "mean_norm_return": 0.9, "failure_rate": 0.0},
    ]
    summary = _aggregate(rows)["I1"]
    assert summary["seeds_evaluated"] == 2
    assert summary["mean_norm_return_mean"] == 0.8
    assert summary["mean_norm_return_std"] > 0


def test_sweep_dry_run_does_not_check_dataset_or_launch_process(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "dry",
        "output_root": str(tmp_path / "runs"),
        "evaluation": {"enabled": True, "dataset": str(tmp_path / "missing"), "num_envs": 8},
        "experiments": [{"id": "B0", "cfg": "baseline.yaml"}],
    }))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path), "--dry-run"])
    monkeypatch.setattr("supermimic.scripts.sweep.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched process")))
    main()
    out = capsys.readouterr().out
    assert "matrix: 1 runs" in out
    assert "benchmark:" in out


def test_sweep_runs_benchmark_and_can_retry_eval_without_retraining(tmp_path, monkeypatch):
    dataset = tmp_path / "npz"
    dataset.mkdir()
    (dataset / "clip.npz").write_bytes(b"")
    output = tmp_path / "runs"
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "eval",
        "output_root": str(output),
        "seeds": [0],
        "evaluation": {"enabled": True, "dataset": str(dataset), "num_envs": 2},
        "experiments": [{"id": "I1", "cfg": "integral.yaml"}],
    }))
    run_dir = output / "eval" / "I1_seed0"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "supermimic.scripts.train" in command:
            checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"weights")
            (run_dir / "metadata.json").write_text(json.dumps({
                "state": "completed", "checkpoint": str(checkpoint), "elapsed_s": 12.0,
            }))
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        out_path = Path(command[command.index("--out") + 1])
        out_path.write_text(json.dumps({"summary": {
            "clips": 2, "failed": 1, "mean_norm_return": 0.75, "step_weighted_norm_return": 0.7,
        }}))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("supermimic.scripts.sweep.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["evaluation_state"] == "completed"
    assert json.loads((run_dir / "eval_metrics.json").read_text())["failure_rate"] == 0.5
    summary = json.loads((output / "eval" / "summary.json").read_text())
    assert summary["runs"][0]["mean_norm_return"] == 0.75
    assert summary["by_experiment"]["I1"]["mean_norm_return_std"] == 0.0

    # Complete training/evaluation is skipped until one of the result files is missing.
    before = len(calls)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert len(calls) == before

    # Missing evaluation output can be rebuilt without another training call.
    metadata["evaluation_state"] = "completed"
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    (run_dir / "benchmark.json").unlink()
    before = len(calls)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    assert len(calls) == before + 1
    assert "eval_benchmark.py" in calls[-1][1]


def test_sweep_keeps_training_checkpoint_when_evaluation_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "npz"
    dataset.mkdir()
    (dataset / "clip.npz").write_bytes(b"")
    output = tmp_path / "runs"
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "eval_failure",
        "output_root": str(output),
        "evaluation": {"enabled": True, "dataset": str(dataset)},
        "experiments": [{"id": "B0", "cfg": "baseline.yaml"}],
    }))
    run_dir = output / "eval_failure" / "B0_seed0"

    def fake_run(command, **kwargs):
        if "supermimic.scripts.train" in command:
            checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"weights")
            (run_dir / "metadata.json").write_text(json.dumps({"state": "completed", "checkpoint": str(checkpoint)}))
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        return SimpleNamespace(returncode=1, stderr="simulated benchmark failure", stdout="")

    monkeypatch.setattr("supermimic.scripts.sweep.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    main()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    result = json.loads((output / "eval_failure" / "summary.json").read_text())["runs"][0]
    assert (run_dir / "checkpoints" / "ckpt_latest.pt").is_file()
    assert metadata["evaluation_state"] == "failed"
    assert result["status"] == "training_completed_eval_failed"
