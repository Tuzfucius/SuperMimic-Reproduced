import sys

import pytest

from supermimic.config_loader import config_dict, load_config


def test_defaults_overrides_and_nested_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("env:\n  num_envs: 16\nintegral:\n  tau_s: 1.5\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nrun_name: child\nintegral:\n  enabled: true\n")
    cfg = load_config(tmp_path / "child.yaml", ["integral.tau_s=3.0", "seed=2"])
    assert cfg.env.num_envs == 16
    assert cfg.integral.enabled and cfg.integral.tau_s == 3.0
    assert cfg.seed == 2
    assert cfg.policy.integral_dim == 29
    assert config_dict(cfg)["integral"]["enabled"] is True


def test_multilevel_inheritance_and_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\nseed: 1\n")
    (tmp_path / "b.yaml").write_text("_base_: c.yaml\nenv:\n  num_envs: 3\n")
    (tmp_path / "c.yaml").write_text("run_name: inherited\n")
    assert load_config(tmp_path / "a.yaml").env.num_envs == 3
    (tmp_path / "c.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_unknown_type_and_invalid_override(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("env:\n  num_envs: many\n")
    with pytest.raises(TypeError, match="env.num_envs"):
        load_config(p)
    p.write_text("mystery: true\n")
    with pytest.raises(ValueError, match="unknown parameter"):
        load_config(p)
    with pytest.raises(ValueError, match="unknown --set"):
        load_config(overrides=["env.no_such_field=1"])


def test_path_tuple_optional_and_boolean_conversion(tmp_path):
    p = tmp_path / "typed.yaml"
    p.write_text("log_root: /tmp/logs\npolicy:\n  decoder_hidden: [16, 8]\nenv:\n  obs_noise: false\ndata:\n  limit: null\n")
    cfg = load_config(p)
    assert str(cfg.log_root) == "/tmp/logs"
    assert cfg.policy.decoder_hidden == (16, 8)
    assert cfg.env.obs_noise is False
    assert cfg.data.limit is None


def test_derived_dimensions_schedule_fields_and_control_rate_guard(tmp_path):
    p = tmp_path / "derived.yaml"
    p.write_text(
        "policy:\n  history: 8\nenv:\n  future_frames: 12\n"
        "schedule:\n  total_iterations: 100\n  max_hours: 1.5\n"
        "integral:\n  enabled: true\n  actor_input: false\n  critic_input: true\n"
    )
    cfg = load_config(p)
    assert cfg.policy.history == cfg.policy_trainer.history == 8
    assert cfg.policy.goal_dim == cfg.env.future_frames * 68
    assert cfg.policy_trainer.future_frames == 12
    assert cfg.policy.integral_dim == 0 and cfg.critic.integral_dim == 29
    assert cfg.schedule.total_iterations == 100 and cfg.schedule.max_hours == 1.5

    p.write_text("env:\n  sim_timestep_s: 0.01\n  decimation: 4\n")
    with pytest.raises(ValueError, match="50 Hz"):
        load_config(p)


def test_resume_and_init_from_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(overrides=["resume=old.pt", "init_from=old.pt"])


def test_train_dry_run_applies_yaml_set_then_legacy_cli_without_trainer(tmp_path, monkeypatch, capsys):
    from supermimic.scripts.train import main

    cfg = tmp_path / "exp.yaml"
    cfg.write_text("run_name: yaml_name\nintegral:\n  enabled: true\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sm-train", "--cfg", str(cfg), "--set", "seed=2", "--run-name", "legacy_name",
            "--env.num-envs", "16", "--dry-run",
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "run_name: legacy_name" in output
    assert "seed: 2" in output
    assert "num_envs: 16" in output
    assert '"integral": 29' in output
    assert '"validation": "configuration structure only; no GPU or simulator initialized"' in output
