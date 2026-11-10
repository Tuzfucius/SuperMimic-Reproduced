"""Strict model-state validation with narrowly scoped checkpoint migrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def is_b0_to_i4_init(source_integral_config: Mapping, *, resume: bool, target_residual_enabled: bool) -> bool:
    """Identify the sole checkpoint migration allowed by training initialization."""
    return (
        not resume
        and target_residual_enabled
        and source_integral_config.get("enabled") is False
        and not source_integral_config.get("residual_enabled", False)
    )


def prepare_state_dict(
    module: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    allowed_missing: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    """Validate every key and shape, filling only explicitly allowed missing keys.

    This returns a complete state dict suitable for a subsequent strict load. It does
    not ignore unexpected keys or shape differences as ``strict=False`` would.
    """
    expected = module.state_dict()
    source_keys, expected_keys = set(checkpoint_state), set(expected)
    missing = expected_keys - source_keys
    unexpected = source_keys - expected_keys
    allowed = set(allowed_missing)
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected keys: {sorted(unexpected)}")
    disallowed_missing = missing - allowed
    if disallowed_missing:
        errors.append(f"missing keys: {sorted(disallowed_missing)}")
    invalid_allowed = allowed - expected_keys
    if invalid_allowed:
        errors.append(f"migration keys are not present in target model: {sorted(invalid_allowed)}")
    for key in source_keys & expected_keys:
        source_value, target_value = checkpoint_state[key], expected[key]
        if not isinstance(source_value, torch.Tensor):
            errors.append(f"{key}: checkpoint value is not a tensor")
        elif source_value.shape != target_value.shape:
            errors.append(
                f"{key}: checkpoint shape {tuple(source_value.shape)} != model shape {tuple(target_value.shape)}"
            )
    if errors:
        raise RuntimeError("incompatible model state dict (" + "; ".join(errors) + ")")

    prepared = dict(expected)
    prepared.update(checkpoint_state)
    for key in missing:
        prepared[key] = torch.zeros_like(expected[key])
    return prepared
