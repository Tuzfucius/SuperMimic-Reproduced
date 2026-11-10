# I4 Explicit Learnable Integral Residual

## Design

I4 reuses the existing 29 dimensional leaky joint-position error accumulator:

```text
e_t = q_ref,t - q_obs,t
z_(t+1) = clamp(exp(-dt/tau_s) * z_t + dt * e_t, -clip_abs, clip_abs)
```

The policy computes `a_base = pi_theta(o_t, g_t)`, then applies
`a_t = a_base + (K_max * tanh(w_I)) * z_t`. The only new trainable values are
the 29 elements of `w_I`, initialized to zero. This preserves the original
Decoder and Critic shapes and makes the initial I4 action equal to B0, including
when the stored integral state is nonzero. The correction is added to normalized
policy action output before the existing environment action handling.

I1 consumes `z_t` as an extra Decoder input. I4 keeps `actor_input: false` and
uses the same state in a separate elementwise residual branch. `critic_input`
stays false. Real collection, imagined rollout, benchmark, interactive playback,
and recorded playback use the same `Policy` action formula. The current action
uses `z_t`; the existing delayed update generates `z_(t+1)` after that action.

## Checkpoint behavior

`init_from` from a checkpoint whose saved integral config explicitly disables
the integral can migrate to I4. All World Model, Actor, and Critic keys and
shapes are checked; the only allowed missing key is `policy.integral_gain_raw`,
which is filled with zeros. Unexpected keys, other missing keys, and shape
mismatches fail before model weights are changed. I4-to-I4 `resume` remains a
strict model and optimizer restore; B0-to-I4 `resume` is rejected by the saved
configuration check and does not receive the migration exception.

Use one B0 checkpoint for both sides of an initialization-controlled comparison:

```bash
B0_CKPT=logs/B0/checkpoints/ckpt_latest.pt
uv run sm-train --cfg configs/experiments/baseline.yaml \
  --set run_name=B0_continue --set init_from="$B0_CKPT"
uv run sm-train --cfg configs/experiments/integral_residual.yaml \
  --set run_name=I4_warmstart --set init_from="$B0_CKPT"
```

Set equal iteration budgets, data, Seed, and other training settings before
comparing. No such training comparison is part of this implementation change.

## Verification scope

The unit tests cover initial action equality, gain gradients and optimizer
updates, per-joint independence, the configured magnitude bound, config and
network dimensions, strict B0-to-I4 state migration, I4 state-dict round trips,
real sampling action construction, imagined action delivery to the step model,
and evaluation Runner integral delivery/reset. They are engineering checks only.
Formal LAFAN1/BONES-SEED training, convergence, benchmark performance, and any
scientific benefit remain unverified.

## Code Review Follow-up / Verification

### P1: complete I4 policy-objective gradient

An added CPU test uses the production `PolicyTrainer`, `ReplayBuffer`, `imagine`,
`ImagStep`, `WorldModel`, `DynamicsMLP`, `StateSpec`, `Critic`, and
`policy_objective`. The small differentiable tracking head deliberately gives
the World Model an action-sensitive joint-position transition and a smooth
reward derivative. The Replay sample supplies nonzero `(batch, 29)` integral
states. The test verifies the residual parameter is in Adam, the predicted
joint state changes, the complete reward-plus-critic objective yields finite,
nonzero gain gradients, Adam changes the gain, the models remain finite, and
the frozen World Model receives no gradients or parameter updates.

A separate real MuJoCo Warp smoke used `walk1_subject1.npz`, 16 environments,
history 2, future horizon 1, World Model hidden sizes `(32, 32)`, Replay window
2, policy batch 16, imagination horizon 3, and 120 environment iterations
(40 warmup). It completed 32 World Model and PolicyTrainer updates. On the final
update the gain gradient norm was `2.258332e-08` with all 29 elements nonzero;
the largest raw-gain change from initialization was `2.0637797e-04`. Integral
state reached the configured `0.5` clip. Model parameters and the I4 gain
gradient checked finite. A saved I4 checkpoint reloaded and produced actions
equal to the live policy on a fixed input. This short smoke confirms the
training chain and does not indicate learned motion quality.

### P2: multi-step state and action alignment

Deterministic CPU tests compare an independently computed leaky-integral
recurrence against five actions from imagined rollout, `Trainer.act()` plus
`IntegralStateTracker`, and `IntegralPolicyRunner`. They use nonzero gains and
three different per-environment errors, checking each action's correction
against the integral in force before that action. A second three-environment
sequence checks ordinary frame advance, repeated frame, return to clip start,
clip switch, Boolean-mask reset, and environment-ID reset. It verifies the
affected residual clears while unaffected environments retain their values.
Existing recorder/Replay tests continue to cover `(state_t, action_t, z_t)` row
alignment and reset isolation.

### P3: old TensorBoard tags

Repository search found no analysis or aggregation code that requires the old
flat integral tag names. `scripts/tb_tail.py` enumerates all TensorBoard tags
and filters by substring, so it accepts both old runs and the new
`policy/real/*` / `policy/imag/*` paths. No compatibility change was needed.

The only production adjustment in this follow-up is detaching the existing
`imag/joint_tracking_error` diagnostic before converting it to a Python scalar.
The full-objective test exposed PyTorch's requires-grad-to-scalar warning on
that metric; the detach keeps logging out of the training graph. No policy,
critic, reward, World Model, or integral recurrence design was changed. The
Critic still does not receive `z_t` by design; its value-estimation limitations
remain a research question.

Validation run for this implementation:

- `env -u PYTHONPATH uv run sm-train --cfg configs/experiments/integral_residual.yaml --dry-run`: passed; reports run `I4`, Decoder input 962, Critic integral input 0, residual enabled, and 29 learnable residual parameters.
- `env -u PYTHONPATH uv run pytest tests/test_integral.py tests/test_integral_lifecycle.py tests/test_integral_residual.py tests/test_i4_training_smoke.py -q`: 24 passed.
- `env -u PYTHONPATH uv run pytest tests -q`: 130 passed, 1 existing TorchScript deprecation warning.
- `env -u PYTHONPATH uv run ruff check src/supermimic/imagination/policy_trainer.py tests/test_integral_residual.py tests/test_i4_training_smoke.py`: passed.
- `env -u PYTHONPATH uv run python -m compileall -q src/supermimic tests/test_integral_residual.py tests/test_i4_training_smoke.py`: passed; `git diff --check`: passed.
- GPU smoke completed as described above. No formal training or benchmark was started.
