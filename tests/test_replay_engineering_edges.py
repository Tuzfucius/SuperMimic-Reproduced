from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig, resolve_config
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.training.loop import Trainer


def _buffer(envs: int, capacity: int) -> ReplayBuffer:
    return ReplayBuffer(envs, capacity, state_dim=1, act_dim=1, xi_dim=1, device="cpu")


def _add_row(buffer: ReplayBuffer, env: int, marker: float, *, start=False, terminal=False) -> None:
    device = buffer.device
    env_ids = torch.tensor([env], device=device)
    buffer.add(
        env_ids,
        s=torch.tensor([[marker]], dtype=torch.float32, device=device),
        a=torch.zeros(1, 1, device=device),
        a_exec=torch.zeros(1, 1, device=device),
        xi=torch.zeros(1, 1, device=device),
        r=torch.zeros(1, device=device),
        done=torch.tensor([float(terminal)], device=device),
        frame=torch.tensor([int(marker)], device=device),
        clip=torch.zeros(1, dtype=torch.long, device=device),
        terminal=torch.tensor([terminal], device=device),
        ep_start=torch.tensor([start], device=device),
    )


def _add_episode(buffer: ReplayBuffer, env: int, transitions: int, base: float) -> None:
    for step in range(transitions):
        _add_row(buffer, env, base + step, start=step == 0)
    _add_row(buffer, env, base + transitions, terminal=True)
    buffer.end_episode(torch.tensor([env], device=buffer.device))


def test_replay_minimum_capacity_with_terminal_row_and_history_boundary():
    # For H=8, T=10 yields dist_to_ptr < 1 after a terminal row: no nonterminal
    # row can be a start. T=11 makes the last transition eligible (dist=1).
    too_small = _buffer(1, 10)
    _add_episode(too_small, 0, transitions=8, base=0)
    with pytest.raises(RuntimeError, match="no sampleable.*start"):
        too_small.sample_starts(1, hist=8)

    minimum = _buffer(1, 11)
    _add_episode(minimum, 0, transitions=8, base=0)
    batch = minimum.sample_starts(32, hist=8)
    assert batch.hist_states.shape == (8, 32, 1)


def test_replay_minimum_capacity_supports_wm_window_but_data_must_be_long_enough():
    # K=7 needs eight stored rows for seven transitions and the target state.
    minimum = _buffer(1, 8)
    _add_episode(minimum, 0, transitions=7, base=0)
    batch = minimum.sample_windows(4, k=7)
    assert batch.states.shape == (8, 4, 1)

    short = _buffer(1, 8)
    _add_episode(short, 0, transitions=6, base=0)
    with pytest.raises(RuntimeError, match="no sampleable.*window"):
        short.sample_windows(1, k=7)

    fragmented = _buffer(1, 16)
    _add_episode(fragmented, 0, transitions=3, base=0)
    _add_episode(fragmented, 0, transitions=3, base=100)
    with pytest.raises(RuntimeError, match="no sampleable.*window"):
        fragmented.sample_windows(1, k=7)


def test_replay_terminal_rows_multiple_envs_wrap_and_interleaved_episodes():
    buf = _buffer(2, 11)
    _add_episode(buf, 0, transitions=8, base=0)
    _add_episode(buf, 1, transitions=5, base=100)
    last_rows = (buf.ptr - 1) % buf.T
    assert buf.terminal[torch.arange(buf.E), last_rows].all()
    assert buf.ptr[0] != buf.ptr[1]
    assert buf.sample_starts(16, hist=8).state.shape == (16, 1)
    assert buf.sample_windows(16, k=7).states.shape == (8, 16, 1)

    # Add enough interleaved episodes to wrap each environment's ring pointer.
    for episode in range(2):
        for env in (1, 0):
            _add_episode(buf, env, transitions=8, base=1000 + episode * 100 + env * 10)
    assert (buf.ptr > 0).any() and (buf.ptr < buf.T).any()
    assert buf.sample_starts(32, hist=8).state.shape == (32, 1)
    assert buf.sample_windows(32, k=7).states.shape == (8, 32, 1)


def test_replay_reports_no_sampleable_rows_for_empty_and_short_episode_data():
    empty = _buffer(1, 12)
    with pytest.raises(RuntimeError, match="no sampleable.*window"):
        empty.sample_windows(1, k=4)
    short = _buffer(1, 12)
    _add_episode(short, 0, transitions=2, base=0)
    with pytest.raises(RuntimeError, match="no sampleable.*window"):
        short.sample_windows(1, k=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_replay_terminal_sampling_smoke():
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    device = torch.device("cuda")
    buf = ReplayBuffer(1, 10, state_dim=71, act_dim=29, xi_dim=33, device=device)
    env_ids = torch.tensor([0], device=device)
    for step in range(5):
        state = torch.zeros(1, 71, device=device)
        state[:, 3] = 1
        buf.add(
            env_ids, s=state, a=torch.zeros(1, 29, device=device),
            a_exec=torch.zeros(1, 29, device=device), xi=torch.zeros(1, 33, device=device),
            r=torch.zeros(1, device=device), done=torch.zeros(1, device=device),
            frame=torch.tensor([step], device=device), clip=torch.zeros(1, dtype=torch.long, device=device),
            terminal=torch.zeros(1, dtype=torch.bool, device=device),
            ep_start=torch.tensor([step == 0], device=device),
        )
    state = torch.zeros(1, 71, device=device)
    state[:, 3] = 1
    buf.add(
        env_ids, s=state, a=torch.zeros(1, 29, device=device),
        a_exec=torch.zeros(1, 29, device=device), xi=torch.zeros(1, 33, device=device),
        r=torch.zeros(1, device=device), done=torch.ones(1, device=device),
        frame=torch.tensor([5], device=device), clip=torch.zeros(1, dtype=torch.long, device=device),
        terminal=torch.ones(1, dtype=torch.bool, device=device),
        ep_start=torch.zeros(1, dtype=torch.bool, device=device),
    )
    buf.end_episode(env_ids)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads=object(), device=device)
    wm_trainer = WorldModelTrainer(wm, lib=None, cfg=WMTrainerCfg(window=4, compile=None))

    class Writer:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (value, step)

        def add_text(self, *args):
            raise AssertionError("valid CUDA diagnostic window should not be skipped")

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(wm_trainer=SimpleNamespace(window=4))
    trainer.buffer = buf
    trainer.wm_trainer = wm_trainer
    trainer.writer = Writer()
    error = trainer._log_online_error(5)
    assert error.is_cuda and torch.isfinite(error).all()
    assert "wm/online_nse_4step" in trainer.writer.scalars


@pytest.mark.parametrize("window,expected", [(1, 1), (4, 4), (8, 8), (12, 8)])
def test_online_diagnostic_uses_configured_window_and_dynamic_metric(window, expected):
    buffer = _buffer(1, 16)
    _add_episode(buffer, 0, transitions=13, base=0)

    class FakeWMTrainer:
        def __init__(self):
            self.requested = None

        def online_error(self, replay, n, k):
            self.requested = k
            assert replay.sample_windows(1, k).states.shape[0] == k + 1
            return torch.arange(1, k + 1, dtype=torch.float32)

    class Writer:
        def __init__(self):
            self.scalars = {}
            self.text = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (value, step)

        def add_text(self, tag, value, step):
            self.text[tag] = (value, step)

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(wm_trainer=SimpleNamespace(window=window))
    trainer.buffer = buffer
    trainer.wm_trainer = FakeWMTrainer()
    trainer.writer = Writer()
    result = trainer._log_online_error(17)

    assert result.shape == (expected,)
    assert trainer.wm_trainer.requested == expected
    assert trainer.writer.scalars["wm/online_nse_1step"] == (1.0, 17)
    if expected > 1:
        assert trainer.writer.scalars[f"wm/online_nse_{expected}step"] == (float(expected), 17)
    else:
        assert not any(tag.endswith("step") and tag != "wm/online_nse_1step" for tag in trainer.writer.scalars)


def test_online_diagnostic_skips_when_no_legal_window_without_masking_errors():
    buffer = _buffer(1, 16)
    _add_episode(buffer, 0, transitions=2, base=0)

    class Writer:
        def __init__(self):
            self.text = {}

        def add_text(self, tag, value, step):
            self.text[tag] = (value, step)

    class MustNotRun:
        def online_error(self, *args, **kwargs):
            raise AssertionError("online_error should be skipped without valid windows")

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(wm_trainer=SimpleNamespace(window=4))
    trainer.buffer = buffer
    trainer.wm_trainer = MustNotRun()
    trainer.writer = Writer()
    assert trainer._log_online_error(23) is None
    assert "wm/online_nse_skipped" in trainer.writer.text
    assert "4-step" in trainer.writer.text["wm/online_nse_skipped"][0]


@pytest.mark.parametrize(
    "history,window,capacity,valid",
    [
        (8, 7, 11, True),
        (8, 7, 10, False),
        (1, 1, 4, True),
        (12, 3, 15, True),
        (4, 12, 13, True),
        (4, 12, 12, False),
    ],
)
def test_config_replay_capacity_structural_lower_bound(history, window, capacity, valid):
    cfg = TrainConfig()
    cfg.policy.history = history
    cfg.wm_trainer.window = window
    cfg.env.replay_capacity = capacity
    if valid:
        resolve_config(cfg)
    else:
        with pytest.raises(ValueError, match="env.replay_capacity"):
            resolve_config(cfg)
