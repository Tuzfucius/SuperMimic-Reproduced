"""Leaky integral of policy-visible joint-position tracking errors."""

from __future__ import annotations

import torch


def update_integral(
    z: torch.Tensor,
    error: torch.Tensor,
    dt_s: float,
    tau_s: float,
    clip_abs: float,
    gradient_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped z[t+1] and elementwise saturation mask."""
    if dt_s <= 0 or tau_s <= 0 or clip_abs <= 0:
        raise ValueError("dt_s, tau_s and clip_abs must be positive")
    if gradient_mode not in {"full", "detach"}:
        raise ValueError("gradient_mode must be 'full' or 'detach'")
    if z.shape != error.shape:
        raise ValueError(f"z and error shapes must match; got {tuple(z.shape)} and {tuple(error.shape)}")
    previous = z if gradient_mode == "full" else z.detach()
    beta = torch.exp(torch.as_tensor(-dt_s / tau_s, device=z.device, dtype=z.dtype))
    raw = beta * previous + dt_s * error
    clipped = raw.clamp(-clip_abs, clip_abs)
    return clipped, raw.abs() >= clip_abs


class IntegralStateTracker:
    """Batched episode state used by real environment execution."""

    def __init__(self, num_envs: int, dimension: int, device: torch.device | str):
        self.state = torch.zeros(num_envs, dimension, device=device)
        self.saturated = torch.zeros_like(self.state, dtype=torch.bool)

    @torch.no_grad()
    def update(self, error: torch.Tensor, dt_s: float, tau_s: float, clip_abs: float) -> torch.Tensor:
        state_next, saturated = update_integral(self.state, error, dt_s, tau_s, clip_abs)
        # Recorder holds this tensor by reference; preserve storage identity across steps.
        self.state.copy_(state_next)
        self.saturated.copy_(saturated)
        return self.state

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = 0
        self.saturated[env_ids] = False

    def metrics(self) -> dict[str, float]:
        return {
            "integral_mean": float(self.state.mean()),
            "integral_abs_mean": float(self.state.abs().mean()),
            "integral_norm": float(self.state.norm(dim=-1).mean()),
            "integral_clip_fraction": float(self.saturated.float().mean()),
        }


class IntegralPolicyRunner:
    """Apply the same delayed error update used by collection to evaluation policies.

    ``act`` is called before each environment step. The current observation determines
    the action with ``z_t``; its error is carried forward and committed on the next call.
    Frame/clip discontinuities identify the auto-reset that happened inside ``env.step``.
    """

    def __init__(self, policy, command):
        self.policy = policy
        self.command = command
        self.options = getattr(policy, "integral_options", {"enabled": False})
        self.state: torch.Tensor | None = None
        self.previous_error: torch.Tensor | None = None
        self.previous_frame: torch.Tensor | None = None
        self.previous_clip: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs, goal = obs["policy"], obs["goal"]
        if not self.options.get("enabled", False):
            return self.policy(policy_obs, goal)

        num_envs = policy_obs.shape[0]
        if self.state is None or self.state.shape[0] != num_envs:
            self.state = torch.zeros(num_envs, self.options.get("dimension", 29), device=policy_obs.device)

        frame = self.command.frame
        clip = self.command.library.frame_clip[frame]
        if self.previous_error is not None:
            next_state, _ = update_integral(
                self.state,
                self.previous_error,
                getattr(self.policy, "control_dt_s", 0.02),
                self.options.get("tau_s", 2.0),
                self.options.get("clip_abs", 0.5),
            )
            reset = (frame != self.previous_frame + 1) | (clip != self.previous_clip)
            if self.options.get("reset_on_episode", True):
                next_state = torch.where(reset[:, None], torch.zeros_like(next_state), next_state)
            self.state = next_state

        history = self.policy.cfg.history
        error = goal[:, : self.state.shape[-1]] - policy_obs[:, (history - 1) * 29 : history * 29]
        self.previous_error = error
        self.previous_frame = frame.clone()
        self.previous_clip = clip.clone()
        integral = self.state if self.options.get("actor_input", True) else None
        return self.policy(policy_obs, goal, integral=integral)
