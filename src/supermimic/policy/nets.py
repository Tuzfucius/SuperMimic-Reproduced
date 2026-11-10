"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import OBS_TERMS

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # 680 (+9 with anchor_obs)
    integral_dim: int = 0
    integral_residual_enabled: bool = False
    integral_residual_max_gain: float = 0.4


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    integral_dim: int = 0


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = (29, 29, 3, 3, 3, 1)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    k, extra = divmod(goal_dim, 68)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        obs_dim = cfg.history * sum(dim for _, dim in OBS_TERMS)
        self.integral_dim = cfg.integral_dim
        self.mlp = MLP(cfg.latent_dim + obs_dim + cfg.integral_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor, integral: torch.Tensor | None = None) -> torch.Tensor:
        parts = [z, obs_hist * self.scale]
        if self.integral_dim:
            if integral is None or integral.shape[-1] != self.integral_dim:
                raise ValueError(f"policy requires integral input of dimension {self.integral_dim}")
            parts.append(integral)
        return self.mlp(torch.cat(parts, dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)
        self.integral_residual_enabled = cfg.integral_residual_enabled
        if self.integral_residual_enabled:
            self.integral_gain_raw = nn.Parameter(torch.zeros(ACT_DIM))
        else:
            self.register_parameter("integral_gain_raw", None)

    def forward_components(
        self,
        obs_hist: torch.Tensor,
        goal: torch.Tensor,
        interface: str = "motion",
        integral: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the unmodified decoder action and its optional integral residual."""
        base_action = self.decoder(self.encoders[interface](goal), obs_hist, integral)
        if not self.integral_residual_enabled:
            return base_action, None
        if integral is None:
            raise ValueError("policy integral residual requires an integral input")
        if integral.shape != base_action.shape:
            raise ValueError(
                "policy integral residual input must match the action batch and dimension; "
                f"got {tuple(integral.shape)} for action {tuple(base_action.shape)}"
            )
        if integral.device != base_action.device:
            raise ValueError(
                "policy integral residual input and action must be on the same device; "
                f"got {integral.device} and {base_action.device}"
            )
        gain = self.cfg.integral_residual_max_gain * torch.tanh(self.integral_gain_raw)
        return base_action, gain * integral

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion", integral: torch.Tensor | None = None) -> torch.Tensor:
        base_action, correction = self.forward_components(obs_hist, goal, interface, integral)
        return base_action if correction is None else base_action + correction


class Critic(nn.Module):
    """V(canonical state 71, goal 680, xi 33, root error 4) -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.integral_dim = cfg.integral_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4 + cfg.integral_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor, integral: torch.Tensor | None = None) -> torch.Tensor:
        parts = [feat, goal * self.goal_scale, xi_norm, e_root]
        if self.integral_dim:
            if integral is None or integral.shape[-1] != self.integral_dim:
                raise ValueError(f"critic requires integral input of dimension {self.integral_dim}")
            parts.append(integral)
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
