from types import SimpleNamespace

import torch
from torch import nn

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.imagination.rollout import imagine


class _ToyLibrary:
    default_joint_pos = torch.zeros(29)

    def last_frame_of(self, frame):
        return torch.full_like(frame, 100)

    def goal_window(self, frame, k):
        goal = torch.zeros(frame.shape[0], k, 68)
        goal[:, :, :29] = 1.0
        return goal.reshape(frame.shape[0], k * 68)

    def ref(self, frame):
        n = frame.numel()
        return SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4))

    def bodies(self, frame):
        n = frame.numel()
        return SimpleNamespace(
            pos_w=torch.zeros(n, 1, 3),
            quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 1, 4),
            lin_vel_w=torch.zeros(n, 1, 3),
            ang_vel_w=torch.zeros(n, 1, 3),
        )


class _ToyHeads:
    anchor_index = 0

    def root_error_features(self, state, ref):
        return torch.zeros(state.shape[0], 4)


class _ToyWorldModel:
    heads = _ToyHeads()


class _ToyStep:
    def __call__(self, state, action, prev_action, xi, ref):
        n = state.shape[0]
        # Keep the world state fixed to isolate the recurrent integral-state gradient path.
        return (
            state,
            action.sum(-1) * 0.0,
            torch.zeros(len(REWARD_TERM_NAMES), n),
            torch.zeros(n),
            torch.zeros(n, 93),
            None,
        )


class _ActionSensitiveToyStep(_ToyStep):
    def __call__(self, state, action, prev_action, xi, ref):
        result = super().__call__(state, action, prev_action, xi, ref)
        obs = result[4].clone()
        obs[:, :29] = action * 0.5
        return (*result[:4], obs, result[5])


class _ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.outputs = []

    def forward(self, obs, goal, interface="motion", integral=None):
        action = self.gain * integral
        self.outputs.append(action)
        return action


def _start(integral, hist_states=None):
    n = 1
    return StartBatch(
        state=torch.zeros(n, 71),
        xi=torch.zeros(n, 33),
        frame=torch.zeros(n, dtype=torch.long),
        prev_action=torch.zeros(n, 29),
        hist_states=hist_states if hist_states is not None else torch.zeros(2, n, 71),
        hist_actions=torch.zeros(2, n, 29),
        integral=integral,
    )


def _run_gradient(mode, step_fn=None, requires_history_grad=False):
    policy = _ToyPolicy()
    z0 = torch.ones(1, 29, requires_grad=True)
    hist_states = torch.zeros(2, 1, 71, requires_grad=requires_history_grad)
    rollout = imagine(
        _ToyWorldModel(), policy, _ToyLibrary(), _start(z0, hist_states), horizon=3,
        future_frames=1, obs_noise=False, step_fn=step_fn or _ToyStep(),
        integral_enabled=True, integral_dt_s=0.02, integral_tau_s=2.0,
        integral_clip_abs=10.0, integral_gradient_mode=mode,
    )
    future_action = rollout.actions[2].sum()
    z_grad = torch.autograd.grad(future_action, z0, allow_unused=True, retain_graph=True)[0]
    history_grad = (
        torch.autograd.grad(future_action, hist_states, allow_unused=True, retain_graph=True)[0]
        if requires_history_grad else None
    )
    action_grad = torch.autograd.grad(future_action, policy.outputs[0], allow_unused=True, retain_graph=True)[0]
    future_action.backward()
    return z_grad, history_grad, action_grad, policy.gain.grad, rollout


def test_full_rollout_preserves_integral_bptt_and_detach_cuts_recurrence():
    full_z_grad, full_history_grad, _, full_policy_grad, full = _run_gradient(
        "full", requires_history_grad=True
    )
    detach_z_grad, detach_history_grad, _, detach_policy_grad, detached = _run_gradient(
        "detach", requires_history_grad=True
    )
    assert full_z_grad is not None and torch.all(full_z_grad > 0)
    assert detach_z_grad is None or torch.count_nonzero(detach_z_grad) == 0
    assert full_history_grad is not None and torch.count_nonzero(full_history_grad) > 0
    assert detach_history_grad is None or torch.count_nonzero(detach_history_grad) == 0
    assert full_policy_grad is not None and full_policy_grad.abs() > 0
    assert detach_policy_grad is not None and detach_policy_grad.abs() > 0
    assert full.integrals.shape == (4, 1, 29)
    assert detached.integrals.shape == (4, 1, 29)


def test_imagination_keeps_historical_policy_action_gradient_through_toy_world_model():
    _, _, action_grad, policy_grad, _ = _run_gradient("full", step_fn=_ActionSensitiveToyStep())
    assert action_grad is not None and torch.count_nonzero(action_grad) > 0
    assert policy_grad is not None and policy_grad.abs() > 0
