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
        self.state, self.saturated = update_integral(self.state, error, dt_s, tau_s, clip_abs)
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
