import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from supermimic.experiment import git_info
from supermimic.scripts import sweep


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _init_sweep_repo(tmp_path: Path, output_root: str = "runs") -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "--quiet")
    source_ignore = Path(__file__).resolve().parents[1] / ".gitignore"
    (repo / ".gitignore").write_text(source_ignore.read_text())

    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    (motion_dir / "clip.npz").write_bytes(b"temporary motion bytes")
    (repo / "train.yaml").write_text(yaml.safe_dump({"data": {"npz_dir": str(motion_dir)}}))
    (repo / "tracked.py").write_text("VALUE = 1\n")
    (repo / "sweep.yaml").write_text(yaml.safe_dump({
        "name": "git-identity",
        "output_root": output_root,
        "seeds": [0, 1],
        "experiments": [{"id": "B0", "cfg": "train.yaml"}],
    }))
    _run_git(repo, "add", ".gitignore", "train.yaml", "tracked.py", "sweep.yaml")
    _run_git(repo, "-c", "user.name=Codex Test", "-c", "user.email=codex@example.invalid", "commit", "--quiet", "-m", "fixture")
    return repo, motion_dir


def _install_fake_training(monkeypatch):
    real_run = subprocess.run
    calls: list[str] = []

    def run(command, *args, **kwargs):
        if command and command[0] == "git":
            return real_run(command, *args, **kwargs)
        assert command[1:3] == ["-m", "supermimic.scripts.train"]
        pairs = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--set"]
        options = dict(pair.split("=", 1) for pair in pairs)
        run_dir = Path(options["log_root"]) / options["run_name"]
        checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"trained model")
        (run_dir / "metadata.json").write_text(json.dumps({
            "state": "completed", "checkpoint": str(checkpoint), "elapsed_s": 1.0,
        }))
        calls.append(options["run_name"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    return calls


def test_default_runs_output_keeps_real_git_identity_stable_and_reusable(tmp_path, monkeypatch):
    repo, _ = _init_sweep_repo(tmp_path)
    monkeypatch.chdir(repo)
    calls = _install_fake_training(monkeypatch)
    before = git_info()
    assert before["dirty"] is False

    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(repo / "sweep.yaml")])
    sweep.main()
    after_first = git_info()
    assert after_first == before
    assert len(calls) == 2

    run_root = repo / "runs" / "git-identity"
    identities = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        metadata = json.loads((run_dir / "metadata.json").read_text())
        identities.append(metadata["training_identity"]["code"])
    assert identities == [before, before]

    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(repo / "sweep.yaml")])
    sweep.main()
    assert len(calls) == 2
    assert git_info() == before

    (repo / "tracked.py").write_text("VALUE = 2\n")
    changed = git_info()
    assert changed["dirty"] is True
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(repo / "sweep.yaml")])
    with pytest.raises(ValueError, match="training identity mismatch"):
        sweep.main()


def test_nonignored_custom_output_root_is_rejected_before_dry_run(tmp_path, monkeypatch):
    repo, _ = _init_sweep_repo(tmp_path, output_root="custom_runs")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(repo / "sweep.yaml"), "--dry-run"])
    with pytest.raises(ValueError, match="output_root.*not ignored"):
        sweep.main()
    assert not (repo / "custom_runs").exists()
