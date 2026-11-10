"""Safe YAML loading, inheritance and typed overrides for TrainConfig."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from supermimic.config import TrainConfig, resolve_config


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"cyclic YAML inheritance: {chain}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: YAML root must be a mapping")
    bases = raw.pop("_base_", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(x, str) for x in bases):
        raise ValueError(f"{path}: _base_ must be a path or list of paths")
    merged: dict[str, Any] = {}
    for base in bases:
        merged = _merge(merged, _read_yaml(path.parent / base, (*stack, path)))
    return _merge(merged, raw)


def _convert(value: Any, typ: Any, where: str) -> Any:
    origin = get_origin(typ)
    args = get_args(typ)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        options = [t for t in args if t is not type(None)]
        for option in options:
            try:
                return _convert(value, option, where)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"{where}: value {value!r} does not match {typ}")
    if typ is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{where}: expected path string, got {type(value).__name__}")
        return Path(value)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected {origin.__name__}, got {type(value).__name__}")
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise TypeError(f"{where}: expected tuple of length {len(args)}, got {len(value)}")
            converted = [_convert(v, item_type, f"{where}[{i}]") for i, (v, item_type) in enumerate(zip(value, args))]
        else:
            item_type = args[0] if args else Any
            converted = [_convert(v, item_type, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(converted) if origin is tuple else converted
    if typ is Any:
        return value
    if typ is bool:
        if type(value) is not bool:
            raise TypeError(f"{where}: expected bool, got {type(value).__name__}")
        return value
    if typ is int:
        if type(value) is not int:
            raise TypeError(f"{where}: expected int, got {type(value).__name__}")
        return value
    if typ is float:
        if type(value) not in (int, float):
            raise TypeError(f"{where}: expected float, got {type(value).__name__}")
        return float(value)
    if not isinstance(value, typ):
        raise TypeError(f"{where}: expected {typ.__name__}, got {type(value).__name__}")
    return value


def _construct(cls: type, values: dict[str, Any], prefix: str = ""):
    if not isinstance(values, dict):
        raise TypeError(f"{prefix or cls.__name__}: expected mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"{prefix or cls.__name__}: unknown parameter(s): {', '.join(sorted(unknown))}")
    hints = get_type_hints(cls)
    kwargs = {}
    for name, val in values.items():
        typ = hints[name]
        child_path = f"{prefix}.{name}" if prefix else name
        if is_dataclass(typ):
            kwargs[name] = _construct(typ, val, child_path)
        else:
            kwargs[name] = _convert(val, typ, child_path)
    return cls(**kwargs)


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    if not all(keys):
        raise ValueError(f"invalid --set path: {dotted!r}")
    node = data
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise ValueError(f"unknown --set parameter: {dotted}")
        node = node[key]
    if keys[-1] not in node:
        raise ValueError(f"unknown --set parameter: {dotted}")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    values = config_dict(TrainConfig())
    if path:
        values = _merge(values, _read_yaml(Path(path)))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid --set value for {key}: {exc}") from exc
        _set_path(values, key, value)
    return resolve_config(_construct(TrainConfig, values))


def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    def conv(v):
        if is_dataclass(v):
            return {f.name: conv(getattr(v, f.name)) for f in fields(v)}
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple):
            return [conv(x) for x in v]
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return conv(cfg)
