"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig, resolve_config
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.experiment import (
    checkpoint_run_dir,
    create_run,
    open_resume_run,
    update_run,
    validate_resume_config,
)
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.integral import IntegralStateTracker
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import (
    allreduce_mean,
    broadcast_params,
    broadcast_setup_error,
    finalize,
    init_distributed,
)
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        # Resume operates in the original run directory; it does not create a new experiment.
        if cfg.resume is not None:
            resume_dir = checkpoint_run_dir(cfg.resume)
            cfg.log_root = resume_dir.parent
            cfg.run_name = resume_dir.name
        self.cfg = resolve_config(cfg)
        cfg = self.cfg
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.run_metadata = {}
        setup_error = None
        if self.is_main:
            try:
                if cfg.resume is not None:
                    resume_ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
                    if "config" not in resume_ck:
                        raise ValueError(
                            "resume checkpoint has no saved configuration; strict resume validation is unavailable"
                        )
                    validate_resume_config(cfg, resume_ck["config"])
                    self.log_dir, self.run_metadata = open_resume_run(cfg, cfg.resume)
                else:
                    self.log_dir, self.run_metadata = create_run(cfg)
                (self.log_dir / "checkpoints").mkdir(exist_ok=True)
            except Exception as exc:  # noqa: BLE001 - all rank-zero failures must reach every rank
                setup_error = f"{type(exc).__name__}: {exc}"
        setup_error = broadcast_setup_error(setup_error)
        if setup_error is not None:
            finalize()
            raise RuntimeError(f"experiment directory initialization failed: {setup_error}")
        if self.world > 1:
            torch.distributed.barrier()
        if self.is_main and cfg.resume is None:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                history_length=cfg.policy.history,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                sim_timestep_s=cfg.env.sim_timestep_s,
                decimation=cfg.env.decimation,
                integral_dim=cfg.integral.dimension if cfg.integral.enabled else 0,
                integral_reset_on_episode=cfg.integral.reset_on_episode,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        self.integral_tracker = (
            IntegralStateTracker(env.num_envs, cfg.integral.dimension, self.device) if cfg.integral.enabled else None
        )
        if self.integral_tracker is not None:
            self.recorder.integral_state = self.integral_tracker.state
        robot = env.scene["robot"]

        fk = G1Kinematics(load_g1_model(), device=self.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(
            self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer,
            integral_cfg=cfg.integral, integral_dt_s=cfg.env.control_dt_s,
        )
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")
            update_run(self.log_dir, self.run_metadata, model_parameters={"world_model": n_params(self.wm), "actor": n_params(self.policy), "critic": n_params(self.critic)})

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.start_iteration = 0  # warm-up (buffer refill) counts from here, also after a resume
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume, resume=True)
        if cfg.init_from is not None:
            self.load(cfg.init_from, resume=False)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        payload = {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "utd_accumulator": self._utd_acc,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
            }
        checkpoint_path = self.log_dir / "checkpoints" / name
        torch.save(
            payload,
            checkpoint_path,
        )
        legacy_path = self.log_dir / name
        if legacy_path.exists() or legacy_path.is_symlink():
            legacy_path.unlink()
        try:
            import os

            os.link(checkpoint_path, legacy_path)
        except OSError:
            import shutil

            shutil.copy2(checkpoint_path, legacy_path)
        update_run(self.log_dir, self.run_metadata, checkpoint=str(checkpoint_path), iteration=self.iteration)

    def load(self, path: Path, resume: bool = True) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.wm.load_state_dict(ck["wm"])
            self.policy.load_state_dict(ck["policy"])
            self.critic.load_state_dict(ck["critic"])
        except RuntimeError as exc:
            mode = "resume" if resume else "init_from"
            raise RuntimeError(f"checkpoint {mode} is structurally incompatible with configured model: {exc}") from exc
        if resume:
            try:
                self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
                self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
                self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
                if "bin_return" in ck:
                    self.command.bin_return.copy_(ck["bin_return"])
                    self.command.bin_memory.copy_(ck["bin_memory"])
            except (KeyError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"checkpoint resume optimizer/sampler state is incompatible: {exc}") from exc
            self.iteration = ck["iteration"]
            self.start_iteration = self.iteration
            self.num_updates = ck["num_updates"]
            self._utd_acc = float(ck.get("utd_accumulator", 0.0))
            print(f"[INFO] resumed model/optimizer/sampler from {path} at iteration {self.iteration}; environment and replay restarted")
        else:
            print(f"[INFO] initialized weights from {path}; optimizer, counters, environment and replay start fresh")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        integral = self.integral_tracker.state if self.integral_tracker is not None and self.cfg.integral.actor_input else None
        return self.policy(obs["policy"], obs["goal"], integral=integral)

    def _log_online_error(self, iteration: int) -> torch.Tensor | None:
        """Log a replay-backed diagnostic without imposing a fixed WM window."""
        diagnostic_k = min(8, self.cfg.wm_trainer.window)
        if not self.buffer.has_sampleable_windows(diagnostic_k):
            self.writer.add_text(
                "wm/online_nse_skipped",
                f"No sampleable {diagnostic_k}-step replay windows; collect longer Episodes or more data.",
                iteration,
            )
            return None
        error = self.wm_trainer.online_error(self.buffer, n=1024, k=diagnostic_k)
        self.writer.add_scalar("wm/online_nse_1step", float(error[0]), iteration)
        if diagnostic_k > 1:
            self.writer.add_scalar(f"wm/online_nse_{diagnostic_k}step", float(error[-1]), iteration)
        return error

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()
        metric_history: list[dict[str, float]] = []
        metrics_path = self.log_dir / "metrics.json"
        if cfg.resume is not None and metrics_path.exists():
            try:
                saved_metrics = json.loads(metrics_path.read_text())
                metric_history = saved_metrics.get("history", saved_metrics.get("recent", []))
            except (OSError, json.JSONDecodeError):
                metric_history = []

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs_t = obs
            obs, rew, term, tout, _extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if self.integral_tracker is not None:
                # Recorder hooks run inside env.step and write the pre-update z_t beside a_t.
                # Commit the observation error only after env.step returns, producing z_(t+1).
                # Policy-visible term-major joint-position history; newest sample is frame history-1.
                h = cfg.policy.history
                q_obs = obs_t["policy"][:, (h - 1) * 29 : h * 29]
                q_ref = obs_t["goal"][:, :29]
                self.integral_tracker.update(
                    q_ref - q_obs, cfg.env.control_dt_s, cfg.integral.tau_s, cfg.integral.clip_abs
                )
                if done.any() and cfg.integral.reset_on_episode:
                    self.integral_tracker.reset(done.nonzero().squeeze(-1))
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = round(cnt)
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                    metric_history.append({k: v / (agg_n if k != "lr" else 1) for k, v in agg.items()})
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    err = self._log_online_error(it) if self.is_main else None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    if self.integral_tracker is not None:
                        for key, value in self.integral_tracker.metrics().items():
                            self.writer.add_scalar(f"policy/{key}", value, it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    diagnostic_k = min(8, cfg.wm_trainer.window)
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} "
                        f"nse{diagnostic_k} {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.is_main:
            metrics_path.write_text(json.dumps({"iterations": self.iteration, "updates": self.num_updates, "history": metric_history, "recent": metric_history[-100:]}, indent=2))
            update_run(self.log_dir, self.run_metadata, state="completed", finished_at=time.time(), elapsed_s=self.run_metadata.get("elapsed_s", 0.0) + time.time() - t_start)
        if self.writer is not None:
            self.writer.flush()
        finalize()
