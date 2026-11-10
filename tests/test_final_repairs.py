import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from supermimic.config import TrainConfig, resolve_config
from supermimic.config_loader import config_dict, load_config
from supermimic.eval.benchmark import evaluation_env_args, load_policy_from_checkpoint
from supermimic.policy.integral import IntegralPolicyRunner
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.scripts.sweep import _aggregate, expand


@pytest.mark.parametrize("history", [5, 8, 10, 12])
@pytest.mark.parametrize("integral", [False, True])
def test_policy_forward_uses_configured_history(history, integral, tmp_path):
    cfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4,
        history=history, integral_dim=29 if integral else 0,
    )
    policy = Policy(cfg)
    obs = torch.zeros(2, history * 93)
    goal = torch.zeros(2, 680)
    z = torch.zeros(2, 29) if integral else None
    result = policy(obs, goal, integral=z)
    first = policy.decoder.mlp.net[0]
    assert first.in_features == 4 + history * 93 + (29 if integral else 0)
    assert result.shape == (2, 29)
    path = tmp_path / "policy.pt"
    checkpoint_cfg = {
        "policy": asdict(cfg), "env": {"future_frames": 10},
        "integral": {"enabled": integral, "dimension": 29, "actor_input": integral},
    }
    torch.save({"config": checkpoint_cfg, "policy": policy.state_dict()}, path)
    restored, loaded_cfg = load_policy_from_checkpoint(path, "cpu")
    assert restored.decoder.mlp.net[0].in_features == first.in_features
    assert evaluation_env_args(["clip.npz"], loaded_cfg).history_length == history


def test_default_policy_checkpoint_shape_is_legacy_compatible():
    legacy = Policy(PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,)))
    assert legacy.decoder.mlp.net[0].in_features == 962
    checkpoint_state = copy.deepcopy(legacy.state_dict())
    restored = Policy(PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,)))
    restored.load_state_dict(checkpoint_state, strict=True)


def test_dry_run_actor_dimension_matches_history_eight_model(tmp_path, monkeypatch, capsys):
    from supermimic.scripts.train import main as train_main

    cfg_path = tmp_path / "history.yaml"
    cfg_path.write_text("policy:\n  history: 8\n")
    monkeypatch.setattr(sys, "argv", ["sm-train", "--cfg", str(cfg_path), "--dry-run"])
    train_main()
    output = capsys.readouterr().out
    assert '"actor_decoder_input": 776' in output
    cfg = resolve_config(TrainConfig())
    cfg.policy.history = 8
    cfg = resolve_config(cfg)
    model = Policy(cfg.policy)
    assert model.decoder.mlp.net[0].in_features == 776


def test_eval_env_args_recover_checkpoint_history():
    args = evaluation_env_args(
        ["clip.npz"], {"policy": {"history": 8}, "env": {"future_frames": 10}},
    )
    assert args.history_length == 8


def test_compact_library_mode_is_resolved_before_trainer(tmp_path):
    cfg_path = tmp_path / "compact.yaml"
    cfg_path.write_text("data:\n  library: packed.pt\n")
    cfg = load_config(cfg_path)
    assert cfg.env.compact_library is True
    assert config_dict(cfg)["env"]["compact_library"] is True


def test_compact_library_explicit_conflict_is_rejected(tmp_path):
    cfg_path = tmp_path / "conflict.yaml"
    cfg_path.write_text("data:\n  library: packed.pt\nenv:\n  compact_library: false\n")
    with pytest.raises(ValueError, match="compact_library"):
        load_config(cfg_path)
    cfg_path.write_text("env:\n  compact_library: true\ndata:\n  library: null\n")
    with pytest.raises(ValueError, match="data.library"):
        load_config(cfg_path)


def test_resume_compact_library_uses_same_resolved_config(tmp_path):
    checkpoint_cfg = resolve_config(TrainConfig())
    checkpoint_cfg.data.library = Path("packed.pt")
    checkpoint_cfg.env.compact_library = True
    requested = copy.deepcopy(checkpoint_cfg)
    validate = __import__("supermimic.experiment", fromlist=["validate_resume_config"]).validate_resume_config
    validate(resolve_config(requested), config_dict(checkpoint_cfg))
    stored = config_dict(checkpoint_cfg)
    del stored["env"]["compact_library"]
    validate(checkpoint_cfg, stored)


def test_old_npz_checkpoint_without_compact_field_remains_compatible():
    saved = config_dict(resolve_config(TrainConfig()))
    del saved["env"]["compact_library"]
    validate = __import__("supermimic.experiment", fromlist=["validate_resume_config"]).validate_resume_config
    validate(resolve_config(TrainConfig()), saved)


def test_init_from_loads_only_weights_and_keeps_fresh_counters(tmp_path):
    from supermimic.training.loop import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.device = "cpu"
    trainer.wm = torch.nn.Linear(2, 2)
    trainer.policy = torch.nn.Linear(2, 2)
    trainer.critic = torch.nn.Linear(2, 2)
    trainer.iteration = 3
    trainer.num_updates = 2
    trainer._utd_acc = 0.25
    payload = {
        key: {name: torch.full_like(value, 7.0) for name, value in model.state_dict().items()}
        for key, model in (("wm", trainer.wm), ("policy", trainer.policy), ("critic", trainer.critic))
    }
    payload.update({"iteration": 100, "num_updates": 50, "utd_accumulator": 0.9})
    checkpoint = tmp_path / "init.pt"
    torch.save(payload, checkpoint)
    trainer.load(checkpoint, resume=False)
    assert trainer.iteration == 3 and trainer.num_updates == 2 and trainer._utd_acc == 0.25
    assert torch.all(trainer.wm.weight == 7) and torch.all(trainer.policy.weight == 7)


def test_sweep_expansion_has_stable_variant_ids_and_parameters(tmp_path):
    spec_a = {
        "experiments": [{"id": "I1", "seeds": [0, 1]}],
        "grid": {"integral.tau_s": [0.5, 2.0]},
    }
    spec_b = {"grid": {"integral.tau_s": [0.5, 2.0]}, "experiments": spec_a["experiments"]}
    rows_a = expand(spec_a, tmp_path / "sweep.yaml")
    rows_b = expand(spec_b, tmp_path / "sweep.yaml")
    assert len(rows_a) == 4
    assert all("variant_id" in row and "params" in row for row in rows_a)
    assert [(r["variant_id"], r["name"]) for r in rows_a] == [(r["variant_id"], r["name"]) for r in rows_b]
    assert len({r["variant_id"] for r in rows_a}) == 2


def test_sweep_matrix_expands_experiments_variants_and_seeds(tmp_path):
    spec = {
        "experiments": [{"id": "I1", "seeds": [0, 1, 2]}, {"id": "I2", "seeds": [0, 1, 2]}],
        "grid": {"integral.tau_s": [0.5, 2.0, 5.0]},
    }
    rows = expand(spec, tmp_path / "sweep.yaml")
    assert len(rows) == 18
    assert len({(r["id"], r["variant_id"]) for r in rows}) == 6


def test_aggregate_separates_variants_and_excludes_missing_runs():
    rows = []
    for exp_id in ("I1", "I2"):
        for variant, values in (("a", [0.5, 0.7, 0.9]), ("b", [0.2, 0.4, 0.6]), ("c", [0.8, None, None])):
            for seed, value in enumerate(values):
                row = {
                    "id": exp_id, "variant_id": variant, "params": {"tau_s": variant},
                    "seed": seed, "status": "completed" if value is not None else "training_failed",
                }
                if value is not None:
                    row.update(mean_norm_return=value, step_weighted_return=value, failure_rate=0.0, failed=0, clips=3)
                rows.append(row)
    summary = _aggregate(rows)
    assert len(summary) == 6
    first = summary["I1::a"]
    assert first["mean_norm_return_mean"] == pytest.approx(0.7)
    assert first["mean_norm_return_std"] == pytest.approx(0.2)
    assert first["seeds_expected"] == 3 and first["seeds_evaluated"] == 3
    single = summary["I1::c"]
    assert single["seeds_evaluated"] == 1 and single["mean_norm_return_std"] is None
    assert single["missing_seeds"] == [1, 2]


class _PolicyProbe:
    def __init__(self):
        self.cfg = SimpleNamespace(history=1)
        self.control_dt_s = 0.02
        self.integral_options = {"enabled": True, "tau_s": 2.0, "clip_abs": 0.5, "actor_input": True}
        self.received = []

    def __call__(self, policy_obs, goal, integral=None):
        self.received.append(integral.clone())
        return torch.zeros(policy_obs.shape[0], 29)


def test_integral_runner_holds_at_clip_end_and_explicit_reset_isolated():
    policy = _PolicyProbe()
    command = SimpleNamespace(
        frame=torch.tensor([1, 4]),
        library=SimpleNamespace(frame_clip=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])),
    )
    runner = IntegralPolicyRunner(policy, command)
    obs = {"policy": torch.zeros(2, 29), "goal": torch.ones(2, 68)}
    runner(obs)
    command.frame[:] = torch.tensor([2, 5])
    runner(obs)
    nonzero = policy.received[-1].clone()
    command.frame[:] = torch.tensor([2, 6])  # environment clamps the completed clip at frame 2
    runner(obs)
    assert torch.allclose(policy.received[-1][0], nonzero[0] * torch.exp(torch.tensor(-0.01)) + 0.02)
    assert torch.count_nonzero(policy.received[-1][1]) > 0


def test_integral_runner_uses_explicit_environment_reset_mask_for_same_frame():
    policy = _PolicyProbe()
    reset_buf = torch.zeros(2, dtype=torch.bool)
    command = SimpleNamespace(
        frame=torch.tensor([2, 2]),
        library=SimpleNamespace(frame_clip=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])),
        _env=SimpleNamespace(reset_buf=reset_buf),
    )
    runner = IntegralPolicyRunner(policy, command)
    obs = {"policy": torch.zeros(2, 29), "goal": torch.ones(2, 68)}
    runner(obs)
    command.frame[:] = 3
    runner(obs)
    assert torch.count_nonzero(policy.received[-1]) > 0

    # A true reset can return to the exact same frame and clip, so frame-delta
    # inference alone cannot find it. The environment reset mask is authoritative.
    reset_buf[0] = True
    runner(obs)
    assert torch.count_nonzero(policy.received[-1][0]) == 0
    assert torch.count_nonzero(policy.received[-1][1]) > 0

    runner.reset(torch.tensor([0]))
    runner(obs)
    assert torch.count_nonzero(policy.received[-1][0]) == 0
    assert torch.count_nonzero(policy.received[-1][1]) > 0


def test_sweep_grid_parameter_order_does_not_change_variant_id(tmp_path):
    rows_a = expand({"experiments": [{"id": "B0"}], "grid": {"b": [2], "a": [1]}}, tmp_path / "x.yaml")
    rows_b = expand({"experiments": [{"id": "B0"}], "grid": {"a": [1], "b": [2]}}, tmp_path / "x.yaml")
    assert rows_a[0]["variant_id"] == rows_b[0]["variant_id"]


def test_training_and_evaluation_identities_track_config_dataset_and_checkpoint(tmp_path, monkeypatch):
    from supermimic.scripts import sweep

    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "abc123", "dirty": False, "status": "clean"})
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("integral:\n  tau_s: 2.0\n")
    child.write_text("_base_: base.yaml\n")
    row = expand({"experiments": [{"id": "I1", "cfg": "child.yaml", "seeds": [0]}]}, child)[0]
    _, hash_a = sweep._train_identity(row, tmp_path / "runs")
    _, hash_same = sweep._train_identity(row, tmp_path / "runs")
    assert hash_a == hash_same

    base.write_text("integral:\n  tau_s: 5.0\n")
    _, hash_inherited = sweep._train_identity(row, tmp_path / "runs")
    assert hash_inherited != hash_a
    row_override = {**row, "set": [*row["set"], "integral.tau_s=5.0"]}
    _, hash_cli = sweep._train_identity(row_override, tmp_path / "runs")
    assert hash_cli == hash_inherited
    row_override = {**row, "set": [*row["set"], "integral.tau_s=3.0"]}
    _, hash_cli_changed = sweep._train_identity(row_override, tmp_path / "runs")
    assert hash_cli_changed != hash_inherited
    other_seed = {**row, "seed": 1, "set": ["seed=1"]}
    _, hash_seed = sweep._train_identity(other_seed, tmp_path / "runs")
    assert hash_seed != hash_inherited
    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "def456", "dirty": False, "status": "clean"})
    identity_commit, hash_commit = sweep._train_identity(row, tmp_path / "runs")
    assert hash_commit != hash_inherited
    assert identity_commit["code"]["commit"] == "def456"
    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "def456", "dirty": True, "status": " M src/example.py"})
    dirty_identity, dirty_hash = sweep._train_identity(row, tmp_path / "runs")
    assert dirty_identity["code"]["dirty"] is True and dirty_hash != hash_commit

    dataset = tmp_path / "eval_data"
    dataset.mkdir()
    clip = dataset / "clip.npz"
    clip.write_bytes(b"first")
    fingerprint = sweep._dataset_identity(dataset)
    assert fingerprint["file_count"] == 1 and "files" not in fingerprint and fingerprint["manifest_sha256"]
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"weights-a")
    eval_cfg = {"dataset": str(dataset), "num_envs": 2}
    _, eval_hash_a = sweep._evaluation_identity(checkpoint, eval_cfg)
    _, eval_hash_same = sweep._evaluation_identity(checkpoint, eval_cfg)
    assert eval_hash_a == eval_hash_same
    _, eval_hash_num_envs = sweep._evaluation_identity(checkpoint, {**eval_cfg, "num_envs": 4})
    assert eval_hash_num_envs != eval_hash_a
    clip.write_bytes(b"changed dataset")
    _, eval_hash_dataset = sweep._evaluation_identity(checkpoint, eval_cfg)
    assert eval_hash_dataset != eval_hash_a
    checkpoint.write_bytes(b"weights-b")
    _, eval_hash_checkpoint = sweep._evaluation_identity(checkpoint, eval_cfg)
    assert eval_hash_checkpoint != eval_hash_dataset
    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "ghi789", "dirty": False, "status": "clean"})
    _, eval_hash_code = sweep._evaluation_identity(checkpoint, eval_cfg)
    assert eval_hash_code != eval_hash_checkpoint


def test_existing_sweep_run_reuse_requires_exact_training_identity(tmp_path, monkeypatch):
    from supermimic.scripts import sweep

    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "abc123", "dirty": False, "status": "clean"})
    output = tmp_path / "runs"
    config = tmp_path / "train.yaml"
    config.write_text("integral:\n  tau_s: 2.0\n")
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "identity", "output_root": str(output),
        "experiments": [{"id": "I1", "cfg": "train.yaml", "seeds": [0]}],
    }))
    row = expand({"experiments": [{"id": "I1", "cfg": "train.yaml", "seeds": [0]}]}, cfg_path)[0]
    run_dir = output / "identity" / row["name"]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        checkpoint = run_dir / "checkpoints" / "ckpt_latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"weights")
        (run_dir / "metadata.json").write_text(json.dumps({
            "state": "completed", "checkpoint": str(checkpoint), "elapsed_s": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    sweep.main()
    sweep.main()
    assert len(calls) == 1

    config.write_text("integral:\n  tau_s: 5.0\n")
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    with pytest.raises(ValueError, match="training identity mismatch"):
        sweep.main()
    assert len(calls) == 1


def test_unverified_and_corrupt_sweep_metadata_are_never_reused(tmp_path, monkeypatch):
    from supermimic.scripts import sweep

    monkeypatch.setattr(sweep, "git_info", lambda: {"commit": "abc123", "dirty": False, "status": "clean"})
    output = tmp_path / "runs"
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text("seed: 0\n")
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "name": "legacy", "output_root": str(output),
        "experiments": [{"id": "B0", "cfg": "train.yaml", "seeds": [0]}],
    }))
    row = expand({"experiments": [{"id": "B0", "cfg": "train.yaml", "seeds": [0]}]}, cfg_path)[0]
    run_dir = output / "legacy" / row["name"]
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text("not-json")
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    sweep.main()
    result = json.loads((output / "legacy" / "summary.json").read_text())["runs"][0]
    assert result["status"] == "skipped_unverified_metadata"

    (run_dir / "metadata.json").write_text(json.dumps({"state": "completed", "checkpoint": "ckpt.pt"}))
    monkeypatch.setattr(sys, "argv", ["sm-sweep", "--cfg", str(cfg_path)])
    sweep.main()
    result = json.loads((output / "legacy" / "summary.json").read_text())["runs"][0]
    assert result["status"] == "skipped_unverified_legacy"
