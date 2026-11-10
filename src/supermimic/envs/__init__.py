"""Environment layer, imported lazily so policy and CPU utilities stay mjlab-free."""

__all__ = ["SuperMimicEnvCfgArgs", "make_env", "make_env_cfg"]


def __getattr__(name: str):
    if name in __all__:
        from supermimic.envs import tracking_env_cfg

        return getattr(tracking_env_cfg, name)
    raise AttributeError(name)
