# YAML experiments and integral policy input

## Configuration

`TrainConfig` remains the schema. YAML files use safe loading, can inherit from one or more
relative `_base_` files, and are type checked before the environment is created. Resolution order
is dataclass defaults, parent YAMLs, current YAML, repeated `--set`, then explicit legacy Tyro
flags. Derived values are written to `resolved.yaml`; the supplied YAML is preserved as
`config.yaml`.

```bash
uv run sm-train --cfg configs/experiments/integral_actor.yaml
uv run sm-train --cfg configs/experiments/integral_actor.yaml \
  --set integral.tau_s=3.0 --set seed=1
uv run sm-train --cfg configs/experiments/integral_actor.yaml --dry-run
```

Dry run validates configuration and reports dimensions and the output path. It does not validate
GPU availability, MuJoCo Warp initialization, model compilation, or a training step. The control
interval is derived from `env.sim_timestep_s * env.decimation` and is also used for the explicit
world-model integrator. Policy history and future-frame dimensions are propagated from their
single environment/policy sources.

Use `sm-sweep` for sequential runs. `--gpu 0` sets `CUDA_VISIBLE_DEVICES`; `--dry-run` prints the
matrix without training. A completed run is skipped by its metadata state. Failed runs are written
to the sweep summary and do not stop later experiments.

## Integral state

The first version integrates the policy-visible relative joint-position error:

```text
e_t = goal_t[:29] - policy_obs_t.newest_joint_pos[29:]
z_(t+1) = clamp(exp(-dt/tau_s) * z_t + dt * e_t, -clip_abs, clip_abs)
```

Both quantities use the observation's relative-to-default convention. The observation side retains
the configured encoder bias and sensor noise; the actor does not read simulator joint state.
Collection writes `z_t` beside the transition for action `a_t`. Replay starts restore that value,
and imagined rollouts apply the same tensor update after the action to produce `z_(t+1)`. `full`
keeps gradients through previous integral states; `detach` cuts the recurrent state edge while
allowing gradients through the current error. Actor and Critic inputs are independently controlled.

The default is disabled and preserves the original policy layer dimensions and checkpoint layout.
Integral policies add 29 inputs to the actor decoder, so parameter counts are recorded in run
metadata and B0/I1/I2/I3 use the same data and training settings unless explicitly overridden.
The joint-only error has no direct information about global root translation or yaw drift; this
mechanism does not imply that those errors are corrected.

## Run state and recovery

Run directories are collision checked and contain source/resolved YAML, metadata, TensorBoard logs,
metrics, and checkpoints. `resume` restores model, optimizers, and sampler statistics and restores
the integral configuration from the checkpoint. The simulator state and replay buffer are not
checkpointed: resumed training restarts the environment and buffer, so it is not step-for-step
continuation. `init_from` loads model weights into a fresh experiment without loading optimizer or
iteration state.

Benchmark evaluation restores integral settings from the checkpoint. The interactive play adapter
updates from consecutive policy observations and detects episode resets from the motion command's
frame and clip transition.
