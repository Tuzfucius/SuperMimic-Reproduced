"""CPU end-to-end PolicyTrainer update through the real I4 imagination objective."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from supermimic.config import IntegralCfg
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.imagination import policy_trainer as policy_trainer_module
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


class _TinyMotionLibrary:
    default_joint_pos = torch.zeros(29)

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.full_like(frame, 100)

    def goal_window(self, frame: torch.Tensor, future_frames: int) -> torch.Tensor:
        n = frame.numel()
        goal = torch.zeros(n, future_frames, 68)
        goal[..., :29] = 0.5
        return goal.reshape(n, future_frames * 68)

    def ref(self, frame: torch.Tensor):
        n = frame.numel()
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4)
        return SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=quat)

    def bodies(self, frame: torch.Tensor):
        n = frame.numel()
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 1, 4)
        return SimpleNamespace(
            pos_w=torch.zeros(n, 1, 3),
            quat_w=quat,
            lin_vel_w=torch.zeros(n, 1, 3),
            ang_vel_w=torch.zeros(n, 1, 3),
        )


class _DifferentiableTrackingHeads:
    """Small differentiable world-model heads that reward predicted joint motion."""

    anchor_index = 0

    def robot_bodies(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "pos": state[..., 0:3].unsqueeze(-2),
            "quat": state[..., 3:7].unsqueeze(-2),
            "lin_vel": state[..., 36:39].unsqueeze(-2),
            "ang_vel": state[..., 39:42].unsqueeze(-2),
        }

    def reward(self, state, action, previous_action, ref: ReferenceFrame, bodies=None):
        # Smooth, non-saturated tracking reward with a nonzero d(reward)/d(q).
        reward = 0.02 * torch.exp(-((state[..., 7:36] - 0.5) ** 2).mean(-1) / 0.09)
        terms = {name: torch.zeros_like(reward) for name in REWARD_TERM_NAMES}
        terms["motion_body_pos"] = reward
        terms["total"] = reward
        return terms

    def termination_features(self, root_pos, root_quat, ref: ReferenceFrame):
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        tilt = root_quat[..., 1] - ref.root_quat_w[..., 1]
        return torch.stack((height, tilt), dim=-1)

    def root_error_features(self, state: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        return torch.zeros(state.shape[0], 4, device=state.device, dtype=state.dtype)


def _populated_replay() -> ReplayBuffer:
    envs, capacity = 2, 24
    replay = ReplayBuffer(envs, capacity, 71, 29, 33, "cpu", integral_dim=29)
    ids = torch.arange(envs)
    state = torch.zeros(envs, 71)
    state[:, 3] = 1.0  # identity root quaternion
    xi = torch.zeros(envs, 33)
    xi[:, 0] = 0.75  # StateSpec's neutral friction context
    action = torch.zeros(envs, 29)
    integral = torch.full((envs, 29), 0.3)
    for t in range(12):
        replay.add(
            ids,
            state,
            action,
            action,
            xi,
            torch.zeros(envs),
            torch.zeros(envs),
            torch.full((envs,), t, dtype=torch.long),
            torch.zeros(envs, dtype=torch.long),
            torch.zeros(envs, dtype=torch.bool),
            torch.full((envs,), t == 0, dtype=torch.bool),
            integral=integral,
        )
    return replay


def test_policy_trainer_objective_updates_i4_gain_through_frozen_world_model(monkeypatch):
    torch.manual_seed(9)
    policy_cfg = PolicyCfg(
        encoder_hidden=(16,),
        decoder_hidden=(16,),
        history=2,
        goal_dim=68,
        integral_residual_enabled=True,
        integral_residual_max_gain=0.4,
    )
    policy = Policy(policy_cfg)
    assert torch.count_nonzero(policy.integral_gain_raw) == 0

    with torch.no_grad():
        # Nonzero base actions make the world-model transition action-sensitive.
        policy.decoder.mlp.net[-1].bias.fill_(0.1)

    heads = _DifferentiableTrackingHeads()
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu")
    dynamics_first = wm.dynamics.mlp.net[0]
    dynamics_last = wm.dynamics.mlp.net[-1]
    with torch.no_grad():
        # Dynamics input is [canonical state (71), action (29), normalized xi (33)].
        # Route its action sum through one hidden unit to each joint-position correction
        # (output indices 41:70), which StateSpec scales by 0.02.
        dynamics_first.weight.zero_()
        dynamics_first.bias.zero_()
        dynamics_first.weight[0, 71:100] = 1.0
        dynamics_last.weight.zero_()
        dynamics_last.bias.zero_()
        dynamics_last.weight[41:70, 0] = 0.5
    wm_before = {name: value.detach().clone() for name, value in wm.state_dict().items()}

    critic = Critic(CriticCfg(hidden=(16,), integral_dim=0), pcfg=policy_cfg)
    library = _TinyMotionLibrary()
    integral_cfg = IntegralCfg(
        enabled=True,
        actor_input=False,
        critic_input=False,
        residual_enabled=True,
        residual_max_gain=0.4,
    )
    trainer = PolicyTrainer(
        wm,
        policy,
        critic,
        library,
        PolicyTrainerCfg(
            horizon=3,
            batch_size=4,
            history=2,
            future_frames=1,
            lr=1e-2,
            grad_clip_norm=100.0,
            grad_clip_value=100.0,
            critic_lr=1e-3,
            critic_epochs=1,
            critic_minibatches=1,
            obs_noise=False,
            compile=None,
        ),
        integral_cfg=integral_cfg,
    )

    optimized_ids = {id(param) for group in trainer.opt.param_groups for param in group["params"]}
    assert id(policy.integral_gain_raw) in optimized_ids

    replay = _populated_replay()
    sampled_starts = []
    original_sample_starts = replay.sample_starts

    def capture_starts(batch_size, history):
        start = original_sample_starts(batch_size, history)
        sampled_starts.append(start)
        return start

    monkeypatch.setattr(replay, "sample_starts", capture_starts)
    imagined_rollouts = []
    original_imagine = policy_trainer_module.imagine

    def capture_rollout(*args, **kwargs):
        rollout = original_imagine(*args, **kwargs)
        imagined_rollouts.append(rollout)
        return rollout

    monkeypatch.setattr(policy_trainer_module, "imagine", capture_rollout)

    gain_before = policy.integral_gain_raw.detach().clone()
    metrics = trainer.update(replay)

    assert len(sampled_starts) == len(imagined_rollouts) == 1
    start = sampled_starts[0]
    rollout = imagined_rollouts[0]
    assert start.integral.shape == (4, 29)
    assert torch.allclose(start.integral, torch.full_like(start.integral, 0.3))
    assert rollout.integrals.shape == (4, 4, 29)
    assert torch.equal(rollout.integrals[0], start.integral)
    assert torch.isfinite(rollout.actions).all() and torch.isfinite(rollout.rewards).all()
    assert torch.count_nonzero(rollout.states[1:, :, 7:36]) > 0  # action changed predicted joint state
    assert torch.isfinite(torch.tensor(metrics["imag_objective"]))

    gain_grad = policy.integral_gain_raw.grad
    assert gain_grad is not None
    assert torch.isfinite(gain_grad).all()
    assert torch.count_nonzero(gain_grad) > 0
    assert torch.isfinite(policy.integral_gain_raw).all()
    assert torch.count_nonzero(policy.integral_gain_raw.detach() - gain_before) > 0
    assert torch.isfinite(policy(torch.zeros(4, 186), library.goal_window(torch.zeros(4, dtype=torch.long), 1), integral=start.integral)).all()

    # The actor objective differentiates through the frozen dynamics, without updating its weights.
    for name, param in wm.named_parameters():
        assert torch.equal(param.detach(), wm_before[name])
        assert param.grad is None
    assert all(param.requires_grad for param in wm.parameters())  # restored for subsequent WM training
