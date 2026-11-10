"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.policy.integral import update_integral
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, 680)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    integrals: torch.Tensor | None  # (H+1, n, 29), z_t aligned with state/action at t
    joint_errors: torch.Tensor | None  # (H+1, n, 29), policy-visible ref-minus-observation error
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
    integral_enabled: bool = False,
    integral_dim: int = 29,
    integral_dt_s: float = 0.02,
    integral_tau_s: float = 2.0,
    integral_clip_abs: float = 0.5,
    integral_gradient_mode: str = "full",
    integral_actor_input: bool = True,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagStep(wm, q0)
    noise_half = OBS_NOISE_HALF.to(device)
    anchor_noise_half = ANCHOR_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_goal(frame_t, ref_t, anchor_pose):
        g = lib.goal_window(frame_t, future_frames)
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :])
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * anchor_noise_half
        return torch.cat([g, rel], dim=-1)

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = [add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    integral = start.integral if integral_enabled and start.integral is not None else (
        torch.zeros(n, integral_dim, device=device) if integral_enabled else None
    )
    c = torch.ones(n, device=device)

    states = [s]
    integrals = [integral] if integral is not None else None
    joint_errors = [torch.zeros(n, integral_dim, device=device)] if integral is not None else None
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references for every step in one batched call.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    ref = ref_at(0)
    if anchor_obs:
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [make_goal(frame, ref, anchor_pose)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)

    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0))
        a = policy(obs_flat, goals[-1], integral=integral if integral_actor_input else None)
        if integral is not None:
            # Match collection: action uses z_t, then the error from the same visible (o_t, g_t)
            # advances memory to z_{t+1}. The newest observation is term-major frame hist[-1].
            error = goals[-1][:, :integral_dim] - hist[-1][:, :integral_dim]
            integral, _ = update_integral(integral, error, integral_dt_s, integral_tau_s, integral_clip_abs, integral_gradient_mode)
            integrals.append(integral)
            joint_errors.append(error)
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, ref)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        obs_new = add_noise(obs_clean)
        hist = hist[1:] + [obs_new]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        next_goal = make_goal(frame, ref, anchor_pose)
        goals.append(next_goal)
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        integrals=torch.stack(integrals) if integrals is not None else None,
        joint_errors=torch.stack(joint_errors) if joint_errors is not None else None,
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )
