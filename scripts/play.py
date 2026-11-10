"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.policy.integral import update_integral


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy, command):
        self.policy = policy
        self.command = command
        opts = getattr(policy, "integral_options", {"enabled": False})
        self.opts = opts
        self.z = None
        self.prev_error = None
        self.prev_frame = None
        self.prev_clip = None

    @torch.no_grad()
    def __call__(self, obs):
        if not self.opts.get("enabled", False):
            return self.policy(obs["policy"], obs["goal"])
        x = obs["policy"]
        goal = obs["goal"]
        n = x.shape[0]
        if self.z is None or self.z.shape[0] != n:
            self.z = torch.zeros(n, 29, device=x.device)
        history = self.policy.cfg.history
        q_obs = x[:, (history - 1) * 29 : history * 29]
        err = goal[:, :29] - q_obs
        frame = self.command.frame
        clip = self.command.library.frame_clip[frame]
        # Viewer policy callbacks run once per control step. Apply the previous observation's
        # error now, so the action at this call receives z_t. The command frame and clip
        # identify auto-reset rows exactly, including resets to another clip.
        if self.prev_error is not None:
            reset = (frame != self.prev_frame + 1) | (clip != self.prev_clip)
            updated, _ = update_integral(self.z, self.prev_error, getattr(self.policy, "control_dt_s", 0.02), self.opts.get("tau_s", 2.0), self.opts.get("clip_abs", 0.5))
            if not self.opts.get("reset_on_episode", True):
                reset = torch.zeros_like(reset)
            self.z = torch.where(reset[:, None], torch.zeros_like(updated), updated)
        self.prev_error = err
        self.prev_frame = frame.clone()
        self.prev_clip = clip.clone()
        return self.policy(x, goal, integral=self.z if self.opts.get("actor_input", True) else None)


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        sim_timestep_s=cfg["env"].get("sim_timestep_s", 0.005),
        decimation=cfg["env"].get("decimation", 4),
        debug_vis=ghost,
    )
    base_env = make_env(args, device=device)
    command = base_env.command_manager.get_term("motion")
    env = RslRlVecEnvWrapper(base_env)
    pol = _DictPolicy(policy, command)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
