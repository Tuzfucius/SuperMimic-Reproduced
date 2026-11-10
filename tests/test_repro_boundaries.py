import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from supermimic.config import TrainConfig, resolve_config
from supermimic.policy.integral import IntegralPolicyRunner, IntegralStateTracker
from supermimic.scripts import sweep
from supermimic.scripts.sweep import _dataset_identity, _train_identity, expand


def _write_training_cfg(tmp_path: Path, *, init_from=None, npz_dir=None, base=False) -> Path:
    data_dir = npz_dir or (tmp_path / "motions")
    data_dir.mkdir(parents=True, exist_ok=True)
    clip = data_dir / "a.npz"
    if not clip.exists():
        clip.write_bytes(b"motion-a")
    payload = {"data": {"npz_dir": str(data_dir)}}
    if init_from is not None:
        payload["init_from"] = str(init_from)
    cfg_path = tmp_path / ("child.yaml" if base else "train.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if base:
        parent = tmp_path / "base.yaml"
        parent.write_text(yaml.safe_dump(payload))
        cfg_path.write_text("_base_: base.yaml\n")
    else:
        cfg_path.write_text(yaml.safe_dump(payload))
    return cfg_path


def _sweep_row(cfg_path: Path, *, seed=0, overrides=None) -> dict:
    spec = {"experiments": [{"id": "I1", "cfg": cfg_path.name, "seeds": [seed], "set": overrides or []}]}
    return expand(spec, cfg_path.parent / "sweep.yaml")[0]


def _clean_git(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "git_info",
        lambda: {"commit": "abc123", "dirty": False, "status": "clean"},
    )


def test_initialization_checkpoint_is_part_of_training_identity(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint_a = tmp_path / "model-a.pt"
    checkpoint_b = tmp_path / "model-b.pt"
    checkpoint_a.write_bytes(b"weights-a")
    checkpoint_b.write_bytes(b"weights-b")
    shared_data = tmp_path / "shared-motions"
    shared_data.mkdir()
    (shared_data / "clip.npz").write_bytes(b"motion")

    scratch_cfg = _write_training_cfg(tmp_path / "scratch", npz_dir=shared_data)
    init_cfg = _write_training_cfg(tmp_path / "init", init_from=checkpoint_a, npz_dir=shared_data)
    row_scratch = _sweep_row(scratch_cfg)
    row_init = _sweep_row(init_cfg)
    scratch_identity, scratch_hash = _train_identity(row_scratch, tmp_path / "runs")
    init_identity, init_hash = _train_identity(row_init, tmp_path / "runs")
    same_identity, same_hash = _train_identity(row_init, tmp_path / "runs")
    assert scratch_identity["initialization"] == {"mode": "scratch"}
    assert init_identity["initialization"]["mode"] == "init_from"
    assert init_identity["initialization"]["checkpoint_path"] == str(checkpoint_a.resolve())
    assert init_identity["initialization"]["checkpoint_sha256"]
    assert scratch_hash != init_hash
    assert init_hash == same_hash
    assert init_identity == same_identity
    assert _train_identity(row_init, tmp_path / "another" / "output")[1] == init_hash

    other_cfg = _write_training_cfg(tmp_path / "other", init_from=checkpoint_b, npz_dir=shared_data)
    _, other_hash = _train_identity(_sweep_row(other_cfg), tmp_path / "runs")
    assert other_hash != init_hash


def test_init_from_content_change_at_same_path_changes_training_identity(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"aaaa")
    original_stat = checkpoint.stat()
    cfg_path = _write_training_cfg(tmp_path / "project", init_from=checkpoint)
    row = _sweep_row(cfg_path)
    _, before = _train_identity(row, tmp_path / "runs")

    checkpoint.write_bytes(b"bbbb")
    os.utime(checkpoint, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    _, after = _train_identity(row, tmp_path / "runs")
    assert checkpoint.stat().st_size == original_stat.st_size
    assert checkpoint.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert after != before


def test_init_from_inheritance_and_cli_override_are_resolved(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    inherited_ckpt = tmp_path / "inherited.pt"
    override_ckpt = tmp_path / "override.pt"
    inherited_ckpt.write_bytes(b"inherited")
    override_ckpt.write_bytes(b"override")
    child = _write_training_cfg(tmp_path / "inherited", init_from=inherited_ckpt, base=True)
    inherited_row = _sweep_row(child)
    inherited_identity, _ = _train_identity(inherited_row, tmp_path / "runs")
    assert inherited_identity["initialization"]["checkpoint_path"] == str(inherited_ckpt.resolve())

    overridden_row = _sweep_row(child, overrides=[f"init_from={override_ckpt}"])
    overridden_identity, _ = _train_identity(overridden_row, tmp_path / "runs")
    assert overridden_identity["initialization"]["checkpoint_path"] == str(override_ckpt.resolve())
    assert overridden_identity["initialization"]["checkpoint_sha256"] != inherited_identity["initialization"]["checkpoint_sha256"]


def test_missing_init_from_checkpoint_is_rejected(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    cfg_path = _write_training_cfg(tmp_path / "project", init_from=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="init_from checkpoint does not exist"):
        _train_identity(_sweep_row(cfg_path), tmp_path / "runs")


def test_sweep_preflight_rejects_missing_init_from_checkpoint(tmp_path, monkeypatch):
    cfg_path = _write_training_cfg(tmp_path / "project", init_from=tmp_path / "missing.pt")
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "experiments": [{"id": "I1", "cfg": os.path.relpath(cfg_path, tmp_path)}],
    }))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(sweep_path), "--dry-run"])
    with pytest.raises(FileNotFoundError, match="init_from checkpoint does not exist"):
        sweep.main()


@pytest.mark.parametrize("resume_source", ["yaml", "set"])
def test_sweep_rejects_resume_even_in_dry_run(tmp_path, monkeypatch, resume_source):
    cfg_path = _write_training_cfg(tmp_path / "project")
    if resume_source == "yaml":
        payload = yaml.safe_load(cfg_path.read_text())
        payload["resume"] = str(tmp_path / "checkpoint.pt")
        cfg_path.write_text(yaml.safe_dump(payload))
        row_set = []
    else:
        row_set = [f"resume={tmp_path / 'checkpoint.pt'}"]
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "output_root": str(tmp_path / "runs"),
        "experiments": [{
            "id": "B0", "cfg": os.path.relpath(cfg_path, sweep_path.parent), "set": row_set,
        }],
    }))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(sweep_path), "--dry-run"])
    with pytest.raises(ValueError, match="does not support resume"):
        sweep.main()


def test_completed_sweep_run_rejects_changed_init_from_weights(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    dataset = tmp_path / "motions"
    dataset.mkdir()
    (dataset / "clip.npz").write_bytes(b"motion")
    init_from = tmp_path / "init.pt"
    init_from.write_bytes(b"aaaa")
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump({"data": {"npz_dir": str(dataset)}, "init_from": str(init_from)}))
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "name": "reuse", "output_root": str(tmp_path / "runs"),
        "experiments": [{"id": "I1", "cfg": cfg_path.name}],
    }))
    row = _sweep_row(cfg_path)
    _, training_hash = _train_identity(row, tmp_path / "runs" / "reuse")
    run_dir = tmp_path / "runs" / "reuse" / row["name"]
    checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trained")
    (run_dir / "metadata.json").write_text(json.dumps({
        "state": "completed", "checkpoint": str(checkpoint),
        "training_identity_sha256": training_hash,
    }))

    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(sweep_path)])
    sweep.main()
    init_from.write_bytes(b"bbbb")
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(sweep_path)])
    with pytest.raises(ValueError, match="training identity mismatch"):
        sweep.main()


def test_dataset_fingerprint_is_content_based_and_order_stable(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    a = dataset / "a.npz"
    b = dataset / "b.npz"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bbbb")
    first = _dataset_identity(dataset)
    second = _dataset_identity(dataset)
    assert first == second
    assert first["fingerprint_method"] == "sha256-content-v1"
    assert first["file_count"] == 2

    original_rglob = Path.rglob

    def reverse_rglob(path, pattern):
        return iter(reversed(list(original_rglob(path, pattern))))

    monkeypatch.setattr(Path, "rglob", reverse_rglob)
    reverse_order = _dataset_identity(dataset)
    assert reverse_order["files_digest"] == first["files_digest"]


def test_dataset_fingerprint_changes_for_same_size_and_restored_mtime_content(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    clip = dataset / "a.npz"
    clip.write_bytes(b"aaaa")
    before_stat = clip.stat()
    before = _dataset_identity(dataset)

    clip.write_bytes(b"bbbb")
    os.utime(clip, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    after = _dataset_identity(dataset)
    assert clip.stat().st_size == before_stat.st_size
    assert clip.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert after["files_digest"] != before["files_digest"]


def test_dataset_fingerprint_changes_when_files_are_added_or_removed(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    a = dataset / "a.npz"
    a.write_bytes(b"a")
    one = _dataset_identity(dataset)
    b = dataset / "b.npz"
    b.write_bytes(b"b")
    two = _dataset_identity(dataset)
    b.unlink()
    three = _dataset_identity(dataset)
    assert two["files_digest"] != one["files_digest"]
    assert three["files_digest"] == one["files_digest"]


def test_content_fingerprint_supports_pt_manifest_and_missing_paths(tmp_path):
    packed = tmp_path / "library.pt"
    packed.write_bytes(b"compact")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name,keep\na,1\n")
    assert _dataset_identity(packed)["file_count"] == 1
    assert _dataset_identity(manifest)["file_count"] == 1
    with pytest.raises(FileNotFoundError, match="dataset input does not exist"):
        _dataset_identity(tmp_path / "missing")


def test_training_identity_tracks_dataset_content_and_hashes_each_file_once(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    dataset = tmp_path / "motions"
    dataset.mkdir()
    (dataset / "a.npz").write_bytes(b"same-size")
    (dataset / "b.npz").write_bytes(b"other")
    cfg_path = _write_training_cfg(tmp_path / "project", npz_dir=dataset)
    rows = expand({
        "experiments": [{"id": "I1", "cfg": os.path.relpath(cfg_path, tmp_path), "seeds": [0, 1, 2]}],
        "grid": {"integral.tau_s": [1.0, 2.0]},
    }, tmp_path / "sweep.yaml")
    calls = 0
    original_hash = sweep._file_sha256

    def count_hash(path, cache=None):
        nonlocal calls
        if cache is None or str(path.expanduser().resolve()) not in cache:
            calls += 1
        return original_hash(path, cache)

    monkeypatch.setattr(sweep, "_file_sha256", count_hash)
    cache = {}
    identities = [_train_identity(row, tmp_path / "runs", file_sha256_cache=cache)[1] for row in rows]
    assert len(set(identities)) == len(rows)
    assert calls == 2

    target = dataset / "a.npz"
    before_stat = target.stat()
    target.write_bytes(b"new-same")
    os.utime(target, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    _, changed_identity = _train_identity(rows[0], tmp_path / "runs", file_sha256_cache={})
    assert changed_identity != identities[0]


class _PolicyProbe:
    def __init__(self, reset_on_episode, device):
        self.cfg = SimpleNamespace(history=1)
        self.control_dt_s = 0.02
        self.integral_options = {
            "enabled": True, "dimension": 29, "tau_s": 2.0, "clip_abs": 10.0,
            "actor_input": True, "reset_on_episode": reset_on_episode,
        }
        self.device = device
        self.received = []

    def __call__(self, policy_obs, goal, integral=None):
        self.received.append(integral.clone())
        return torch.zeros(policy_obs.shape[0], 29, device=self.device)


def _observation(error: float, device: str) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.zeros(1, 29, device=device),
        "goal": torch.full((1, 68), error, device=device),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def test_runner_reset_false_preserves_terminal_error_update_exactly_once(device):
    policy = _PolicyProbe(reset_on_episode=False, device=device)
    command = SimpleNamespace(
        frame=torch.tensor([0], device=device),
        library=SimpleNamespace(frame_clip=torch.zeros(8, dtype=torch.long, device=device)),
    )
    runner = IntegralPolicyRunner(policy, command)
    trainer_reference = IntegralStateTracker(1, 29, device)

    runner(_observation(1.25, device))  # action_t sees z_t; save e_t for after env.step
    runner.reset(torch.tensor([0], device=device))  # reset with carry-over configured
    trainer_reference.update(torch.full((1, 29), 1.25, device=device), 0.02, 2.0, 10.0)

    command.frame[:] = 1
    runner(_observation(-0.5, device))  # commits e_t before the next action
    assert torch.allclose(policy.received[-1], trainer_reference.state)

    trainer_reference.update(torch.full((1, 29), -0.5, device=device), 0.02, 2.0, 10.0)
    command.frame[:] = 2
    runner(_observation(0.75, device))
    assert torch.allclose(policy.received[-1], trainer_reference.state)
    assert torch.allclose(
        trainer_reference.state,
        torch.full(
            (1, 29), torch.exp(torch.tensor(-0.01, device=device)).item() * (0.02 * 1.25) - 0.02 * 0.5,
            device=device,
        ),
    )


def test_resolve_config_checks_history_and_world_model_replay_capacity():
    assert resolve_config(TrainConfig()).env.replay_capacity == 1000

    minimum = TrainConfig()
    minimum.policy.history = 5
    minimum.wm_trainer.window = 7
    minimum.env.replay_capacity = 9
    resolve_config(minimum)

    for history, window, capacity, limiting_name in (
        (8, 7, 9, "policy.history"),
        (5, 8, 9, "wm_trainer.window"),
        (5, 7, 8, "env.replay_capacity"),
    ):
        cfg = TrainConfig()
        cfg.policy.history = history
        cfg.wm_trainer.window = window
        cfg.env.replay_capacity = capacity
        with pytest.raises(ValueError, match=limiting_name):
            resolve_config(cfg)


@pytest.mark.parametrize("history,window", [(1, 1), (4, 9), (12, 3)])
def test_minimum_valid_replay_capacity_tracks_larger_window(history, window):
    cfg = TrainConfig()
    cfg.policy.history = history
    cfg.wm_trainer.window = window
    cfg.env.replay_capacity = max(history, window) + 2
    resolve_config(cfg)
