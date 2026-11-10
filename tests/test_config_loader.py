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
