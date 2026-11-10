"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    sim_timestep_s: float = 0.005
    decimation: int = 4

    @property
    def control_dt_s(self) -> float:
        return self.sim_timestep_s * self.decimation


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class IntegralCfg:
    enabled: bool = False
    error_source: str = "joint_pos"
    dimension: int = 29
    tau_s: float = 2.0
    clip_abs: float = 0.5
    actor_input: bool = True
    critic_input: bool = False
    gradient_mode: str = "full"  # full | detach
    reset_on_episode: bool = True


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    integral: IntegralCfg = field(default_factory=IntegralCfg)
    resume: Path | None = None
    init_from: Path | None = None


def resolve_config(cfg: TrainConfig) -> TrainConfig:
    """Derive the single effective network/environment dimensions and validate dependencies."""
    if cfg.env.future_frames < 1:
        raise ValueError("env.future_frames must be >= 1")
    if cfg.env.decimation < 1 or cfg.env.sim_timestep_s <= 0:
        raise ValueError("env.decimation and env.sim_timestep_s must be positive")
    # The simulator's composed control interval is the authority for both the
    # explicit world-model integrator and the error-memory recurrence.
    cfg.world_model.state_spec.dt = cfg.env.control_dt_s
    if cfg.policy.history < 1:
        raise ValueError("policy.history must be >= 1")
    cfg.policy_trainer.history = cfg.policy.history
    cfg.policy_trainer.future_frames = cfg.env.future_frames
    cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0)
    integral = cfg.integral
    if integral.error_source != "joint_pos":
        raise ValueError("integral.error_source currently supports only 'joint_pos'")
    if integral.dimension != 29:
        raise ValueError("integral.dimension must be 29 for G1 joint position tracking")
    if integral.tau_s <= 0:
        raise ValueError("integral.tau_s must be > 0")
    if integral.clip_abs <= 0:
        raise ValueError("integral.clip_abs must be > 0")
    if integral.gradient_mode not in {"full", "detach"}:
        raise ValueError("integral.gradient_mode must be 'full' or 'detach'")
    cfg.policy.integral_dim = integral.dimension if integral.enabled and integral.actor_input else 0
    cfg.critic.integral_dim = integral.dimension if integral.enabled and integral.critic_input else 0
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    return cfg
