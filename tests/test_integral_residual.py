"""I4 explicit integral-residual policy, lifecycle, and checkpoint tests."""

from __future__ import annotations

import copy
import io
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.config import IntegralCfg
from supermimic.config_loader import load_config
from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination import rollout as rollout_module
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.rollout import imagine
from supermimic.policy.checkpoint import is_b0_to_i4_init, prepare_state_dict
from supermimic.policy.integral import IntegralPolicyRunner
from supermimic.policy.nets import Critic, Policy, PolicyCfg
from supermimic.training.loop import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _small_cfg(*, residual: bool = False, max_gain: float = 0.4) -> PolicyCfg:
    return PolicyCfg(
        encoder_hidden=(16,),
        decoder_hidden=(16,),
        history=2,
        goal_dim=68,
        integral_residual_enabled=residual,
        integral_residual_max_gain=max_gain,
    )


def _paired_policies():
    torch.manual_seed(7)
    base = Policy(_small_cfg())
    i4 = Policy(_small_cfg(residual=True))
    state = prepare_state_dict(i4, base.state_dict(), allowed_missing=("integral_gain_raw",))
    i4.load_state_dict(state, strict=True)
    return base, i4


def test_zero_initialized_i4_action_matches_b0_for_nonzero_integral():
    base, i4 = _paired_policies()
    obs = torch.randn(4, 186)
    goal = torch.randn(4, 68)
    integral = torch.randn(4, 29)
    assert torch.count_nonzero(i4.integral_gain_raw) == 0
    assert torch.equal(i4(obs, goal, integral=integral), base(obs, goal))


def test_integral_gain_has_gradient_and_optimizer_updates_action():
    _, i4 = _paired_policies()
    obs = torch.randn(3, 186)
    goal = torch.randn(3, 68)
    integral = torch.ones(3, 29)
    optimizer = torch.optim.SGD(i4.parameters(), lr=0.1)

    optimizer.zero_grad(set_to_none=True)
    i4(obs, goal, integral=integral).sum().backward()
    grad = i4.integral_gain_raw.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    before = i4.integral_gain_raw.detach().clone()
    optimizer.step()
    assert torch.count_nonzero(i4.integral_gain_raw.detach() - before) > 0
    changed = i4(obs, goal, integral=integral)
    base_action, _ = i4.forward_components(obs, goal, integral=integral)
    assert torch.count_nonzero(changed - base_action) > 0


def test_residual_is_independent_per_joint_and_batch():
    _, i4 = _paired_policies()
    joint = 11
    with torch.no_grad():
        i4.integral_gain_raw.zero_()
        i4.integral_gain_raw[joint] = torch.atanh(torch.tensor(0.5))
    obs, goal = torch.zeros(2, 186), torch.zeros(2, 68)
    integral = torch.zeros(2, 29)
    integral[:, joint] = torch.tensor([0.3, -0.2])
    base, _ = i4.forward_components(obs, goal, integral=integral)
    actual = i4(obs, goal, integral=integral)
    delta = actual - base
    assert torch.count_nonzero(delta[:, :joint]) == 0
    assert torch.count_nonzero(delta[:, joint + 1 :]) == 0
    assert torch.allclose(delta[:, joint], 0.2 * integral[:, joint])


def test_gain_and_clipped_integral_bound_residual_magnitude():
    _, i4 = _paired_policies()
    with torch.no_grad():
        i4.integral_gain_raw.fill_(10.0)
    gain = i4.cfg.integral_residual_max_gain * torch.tanh(i4.integral_gain_raw)
    z = torch.full((2, 29), 0.5)
    correction = gain * z
    assert torch.all(gain.abs() <= 0.4)
    assert torch.all(correction.abs() <= 0.2)


def test_configs_keep_i4_decoder_critic_and_world_model_structure_compatible():
    b0 = load_config(ROOT / "configs/experiments/baseline.yaml")
    i1 = load_config(ROOT / "configs/experiments/integral_actor.yaml")
    i2 = load_config(ROOT / "configs/experiments/integral_actor_critic.yaml")
    i3 = load_config(ROOT / "configs/experiments/integral_detach.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        i4 = load_config(ROOT / "configs/experiments/integral_residual.yaml")
    assert not caught
    assert not b0.integral.enabled and b0.policy.integral_dim == 0
    assert i1.policy.integral_dim == 29 and i1.critic.integral_dim == 0
    assert i2.policy.integral_dim == 29 and i2.critic.integral_dim == 29
    assert i3.integral.gradient_mode == "detach"
    assert i4.run_name == "I4" and i4.integral.residual_enabled
    assert i4.integral.actor_input is False and i4.integral.critic_input is False
    assert i4.policy.integral_dim == i4.critic.integral_dim == 0
    assert i4.policy.integral_residual_enabled and i4.policy.integral_residual_max_gain == 0.4
    assert i4.world_model == b0.world_model

    b0_policy_cfg = replace(b0.policy, encoder_hidden=(16,), decoder_hidden=(16,))
    i4_policy_cfg = replace(i4.policy, encoder_hidden=(16,), decoder_hidden=(16,))
    b0_policy, i4_policy = Policy(b0_policy_cfg), Policy(i4_policy_cfg)
    assert b0_policy.decoder.mlp.net[0].in_features == 962
    assert i4_policy.decoder.mlp.net[0].in_features == 962
    assert sum(p.numel() for p in i4_policy.parameters()) - sum(p.numel() for p in b0_policy.parameters()) == 29

    critic_cfg = replace(i4.critic, hidden=(16,))
    b0_critic = Critic(critic_cfg, pcfg=b0_policy_cfg)
    i4_critic = Critic(critic_cfg, pcfg=i4_policy_cfg)
    assert sum(p.numel() for p in i4_critic.parameters()) == sum(p.numel() for p in b0_critic.parameters())


@pytest.mark.parametrize("yaml_text", [
    "integral:\n  enabled: false\n  residual_enabled: true\n",
    "integral:\n  enabled: true\n  residual_max_gain: 0\n",
    "integral:\n  enabled: true\n  residual_max_gain: .inf\n",
])
def test_invalid_integral_residual_configs_are_rejected(tmp_path, yaml_text):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError):
        load_config(path)


def test_b0_init_migration_is_init_from_only_and_strictly_limited():
    base, i4 = _paired_policies()
    source_cfg = {"enabled": False}
    assert is_b0_to_i4_init(source_cfg, resume=False, target_residual_enabled=True)
    assert not is_b0_to_i4_init(source_cfg, resume=True, target_residual_enabled=True)
    assert not is_b0_to_i4_init({"enabled": True}, resume=False, target_residual_enabled=True)
    assert not is_b0_to_i4_init(source_cfg, resume=False, target_residual_enabled=False)

    migrated = prepare_state_dict(i4, base.state_dict(), allowed_missing=("integral_gain_raw",))
    i4.load_state_dict(migrated, strict=True)
    assert torch.count_nonzero(i4.integral_gain_raw) == 0
    obs, goal, z = torch.randn(2, 186), torch.randn(2, 68), torch.randn(2, 29)
    assert torch.equal(i4(obs, goal, integral=z), base(obs, goal))

    bad_missing = copy.deepcopy(base.state_dict())
    bad_missing.pop("decoder.mlp.net.0.weight")
    with pytest.raises(RuntimeError, match="missing keys"):
        prepare_state_dict(i4, bad_missing, allowed_missing=("integral_gain_raw",))
    bad_shape = copy.deepcopy(base.state_dict())
    bad_shape["decoder.mlp.net.0.weight"] = bad_shape["decoder.mlp.net.0.weight"][:1]
    with pytest.raises(RuntimeError, match="checkpoint shape"):
        prepare_state_dict(i4, bad_shape, allowed_missing=("integral_gain_raw",))
    with pytest.raises(RuntimeError, match="missing keys"):
        prepare_state_dict(i4, base.state_dict())  # resume uses no migration exception

    extra = copy.deepcopy(i4.state_dict())
    extra["unexpected.key"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        prepare_state_dict(i4, extra)


def test_i4_state_dict_round_trip_is_strict_and_preserves_actions():
    _, i4 = _paired_policies()
    with torch.no_grad():
        i4.integral_gain_raw.copy_(torch.linspace(-0.5, 0.5, 29))
    obs, goal, z = torch.randn(2, 186), torch.randn(2, 68), torch.randn(2, 29)
    expected = i4(obs, goal, integral=z)
    stream = io.BytesIO()
    torch.save(i4.state_dict(), stream)
    stream.seek(0)
    restored = Policy(_small_cfg(residual=True))
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(restored(obs, goal, integral=z), expected)


def _bare_trainer(policy, wm, critic, integral_cfg):
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(integral=integral_cfg)
    trainer.device = "cpu"
    trainer.wm, trainer.policy, trainer.critic = wm, policy, critic
    trainer.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(wm.parameters(), lr=1e-3))
    trainer.policy_trainer = SimpleNamespace(
        opt=torch.optim.Adam(policy.parameters(), lr=1e-3),
        critic_opt=torch.optim.Adam(critic.parameters(), lr=1e-3),
    )
    return trainer


def test_trainer_load_migrates_only_init_from_and_resumes_i4_strictly(tmp_path):
    b0_policy, i4_policy = _paired_policies()
    b0_wm, b0_critic = torch.nn.Linear(3, 2), torch.nn.Linear(4, 1)
    target_wm, target_critic = copy.deepcopy(b0_wm), copy.deepcopy(b0_critic)
    b0_ckpt = tmp_path / "b0.pt"
    torch.save({
        "wm": b0_wm.state_dict(),
        "policy": b0_policy.state_dict(),
        "critic": b0_critic.state_dict(),
        "config": {"integral": {"enabled": False}},
    }, b0_ckpt)

    init_trainer = _bare_trainer(
        i4_policy, target_wm, target_critic,
        IntegralCfg(enabled=True, actor_input=False, residual_enabled=True),
    )
    with torch.no_grad():
        init_trainer.policy.integral_gain_raw.fill_(0.3)
    init_trainer.load(b0_ckpt, resume=False)
    assert torch.count_nonzero(init_trainer.policy.integral_gain_raw) == 0
    obs, goal, z = torch.randn(2, 186), torch.randn(2, 68), torch.randn(2, 29)
    assert torch.equal(init_trainer.policy(obs, goal, integral=z), b0_policy(obs, goal))

    resume_b0_trainer = _bare_trainer(
        Policy(_small_cfg(residual=True)), copy.deepcopy(b0_wm), copy.deepcopy(b0_critic),
        IntegralCfg(enabled=True, actor_input=False, residual_enabled=True),
    )
    with pytest.raises(RuntimeError, match="missing keys"):
        resume_b0_trainer.load(b0_ckpt, resume=True)

    resume_trainer = _bare_trainer(
        Policy(_small_cfg(residual=True)), copy.deepcopy(target_wm), copy.deepcopy(target_critic),
        IntegralCfg(enabled=True, actor_input=False, residual_enabled=True),
    )
    with torch.no_grad():
        resume_trainer.policy.integral_gain_raw.fill_(0.15)
    i4_ckpt = tmp_path / "i4.pt"
    torch.save({
        "wm": resume_trainer.wm.state_dict(),
        "policy": resume_trainer.policy.state_dict(),
        "critic": resume_trainer.critic.state_dict(),
        "wm_opt": resume_trainer.wm_trainer.opt.state_dict(),
        "policy_opt": resume_trainer.policy_trainer.opt.state_dict(),
        "critic_opt": resume_trainer.policy_trainer.critic_opt.state_dict(),
        "iteration": 12,
        "num_updates": 4,
        "utd_accumulator": 0.25,
        "config": {"integral": {"enabled": True, "residual_enabled": True}},
    }, i4_ckpt)
    resume_trainer.iteration = resume_trainer.start_iteration = 0
    resume_trainer.num_updates = 0
    resume_trainer._utd_acc = 0.0
    resume_trainer.load(i4_ckpt, resume=True)
    assert resume_trainer.iteration == 12 and resume_trainer.num_updates == 4
    assert resume_trainer._utd_acc == 0.25
    assert torch.all(resume_trainer.policy.integral_gain_raw == 0.15)


def test_eval_checkpoint_loader_constructs_i4_residual_policy(tmp_path):
    from dataclasses import asdict

    from supermimic.eval.benchmark import load_policy_from_checkpoint

    _, policy = _paired_policies()
    with torch.no_grad():
        policy.integral_gain_raw.copy_(torch.linspace(-0.2, 0.2, 29))
    ckpt = tmp_path / "i4_eval.pt"
    config = {
        "policy": asdict(policy.cfg),
        "integral": {
            "enabled": True,
            "dimension": 29,
            "actor_input": False,
            "residual_enabled": True,
            "residual_max_gain": 0.4,
            "tau_s": 2.0,
            "clip_abs": 0.5,
            "reset_on_episode": True,
        },
        "env": {"sim_timestep_s": 0.005, "decimation": 4},
    }
    torch.save({"config": config, "policy": policy.state_dict()}, ckpt)
    restored, restored_cfg = load_policy_from_checkpoint(ckpt, "cpu")
    obs, goal, z = torch.randn(2, 186), torch.randn(2, 68), torch.randn(2, 29)
    assert restored.integral_residual_enabled
    assert restored.integral_options["residual_enabled"]
    assert torch.equal(restored(obs, goal, integral=z), policy(obs, goal, integral=z))
    assert restored_cfg["integral"]["actor_input"] is False


class _ToyLibrary:
    default_joint_pos = torch.zeros(29)

    def last_frame_of(self, frame):
        return torch.full_like(frame, 100)

    def goal_window(self, frame, k):
        goal = torch.zeros(frame.shape[0], k, 68)
        goal[..., :29] = 1.0
        return goal.reshape(frame.shape[0], k * 68)


class _ToyHeads:
    anchor_index = 0

    def root_error_features(self, state, ref):
        return torch.zeros(state.shape[0], 4)


class _ToyWorldModel:
    heads = _ToyHeads()


class _ActionCaptureStep:
    def __init__(self):
        self.actions = []

    def __call__(self, state, action, prev_action, xi, ref):
        self.actions.append(action)
        n = state.shape[0]
        return (
            state,
            action.sum(-1) * 0.0,
            torch.zeros(len(REWARD_TERM_NAMES), n),
            torch.zeros(n),
            torch.zeros(n, 93),
            None,
        )


def _fake_reference_at(lib, frames):
    n = frames.numel()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 1, 4)
    return SimpleNamespace(
        body_pos_w=torch.zeros(n, 1, 3),
        body_quat_w=quat,
        body_lin_vel_w=torch.zeros(n, 1, 3),
        body_ang_vel_w=torch.zeros(n, 1, 3),
        root_pos_w=torch.zeros(n, 3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4),
    )


def _toy_start(z):
    return StartBatch(
        state=torch.zeros(1, 71),
        xi=torch.zeros(1, 33),
        frame=torch.zeros(1, dtype=torch.long),
        prev_action=torch.zeros(1, 29),
        hist_states=torch.zeros(2, 1, 71),
        hist_actions=torch.zeros(2, 1, 29),
        integral=z,
    )


def test_real_imagined_and_evaluation_paths_pass_integral_to_same_policy(monkeypatch):
    _, policy = _paired_policies()
    with torch.no_grad():
        policy.integral_gain_raw.fill_(0.2)
    z = torch.full((1, 29), 0.1)
    goal = torch.zeros(1, 68)
    goal[:, :29] = 1.0
    obs = {"policy": torch.zeros(1, 186), "goal": goal}

    trainer = Trainer.__new__(Trainer)
    trainer.policy = policy
    trainer.integral_tracker = SimpleNamespace(state=z)
    trainer.cfg = SimpleNamespace(integral=IntegralCfg(
        enabled=True, actor_input=False, critic_input=False, residual_enabled=True,
    ))
    trainer._real_action_abs_sums = torch.zeros(2)
    trainer._real_action_abs_max = torch.zeros(2)
    trainer._real_action_count = 0
    real_action = trainer.act(obs)

    monkeypatch.setattr(rollout_module, "analytic_observation", lambda s, a, xi, q0: torch.zeros(s.shape[0], 93))
    monkeypatch.setattr(
        rollout_module,
        "flatten_history_term_major",
        lambda hist: hist.permute(1, 0, 2).reshape(hist.shape[1], -1),
    )
    monkeypatch.setattr(rollout_module, "reference_at", _fake_reference_at)
    step = _ActionCaptureStep()
    imagined = imagine(
        _ToyWorldModel(), policy, _ToyLibrary(), _toy_start(z), horizon=2,
        future_frames=1, obs_noise=False, step_fn=step,
        integral_enabled=True, integral_dim=29, integral_clip_abs=0.5,
        integral_actor_input=False, integral_residual_enabled=True,
    )
    assert torch.equal(step.actions[0], real_action)
    assert torch.equal(imagined.actions[0], real_action)
    assert imagined.integral_action_metrics is not None
    imagined.actions[1].sum().backward()
    assert policy.integral_gain_raw.grad is not None
    assert torch.count_nonzero(policy.integral_gain_raw.grad) > 0

    runner_policy = Policy(_small_cfg(residual=True))
    runner_policy.load_state_dict(policy.state_dict(), strict=True)
    runner_policy.integral_options = {
        "enabled": True,
        "dimension": 29,
        "actor_input": False,
        "residual_enabled": True,
        "tau_s": 2.0,
        "clip_abs": 0.5,
        "reset_on_episode": True,
    }
    runner_policy.control_dt_s = 0.02
    command = SimpleNamespace(
        frame=torch.tensor([0]),
        library=SimpleNamespace(frame_clip=torch.zeros(8, dtype=torch.long)),
    )
    runner = IntegralPolicyRunner(runner_policy, command)
    eval_action = runner(obs)
    assert torch.equal(eval_action, runner_policy(obs["policy"], obs["goal"], integral=torch.zeros(1, 29)))
    runner.state.fill_(0.1)
    runner.previous_error.zero_()
    command.frame += 1
    eval_nonzero = runner(obs)
    assert torch.equal(eval_nonzero, runner_policy(obs["policy"], obs["goal"], integral=runner.state))
    runner.reset(torch.tensor([True]))
    assert torch.count_nonzero(runner.state) == 0
