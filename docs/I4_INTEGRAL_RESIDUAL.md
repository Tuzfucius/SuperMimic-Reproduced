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

Validation run for this implementation:

- `env -u PYTHONPATH uv run sm-train --cfg configs/experiments/integral_residual.yaml --dry-run`: passed; reports run `I4`, Decoder input 962, Critic integral input 0, residual enabled, and 29 learnable residual parameters.
- `env -u PYTHONPATH uv run pytest tests/test_integral.py tests/test_integral_lifecycle.py tests/test_integral_residual.py -q`: 21 passed.
- `env -u PYTHONPATH uv run pytest tests -q`: 127 passed.
- Modified Python files pass Ruff and `compileall`.
- No GPU smoke was run: the RTX 4080 showed 100% utilization and 13.1 GiB of 32 GiB in use during the check. No formal training was started.
