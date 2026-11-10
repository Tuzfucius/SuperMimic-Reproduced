"""CPU integration tests for policy -> recorder -> replay integral alignment."""

from types import SimpleNamespace

import torch

from supermimic.data.recorder import ReplayRecorder
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.policy.integral import IntegralPolicyRunner, IntegralStateTracker


class _StateIO:
    def __init__(self, state):
        self.state = state

    def read(self):
        return self.state


class _Context:
    def read(self):
        return torch.zeros(2, 33)


def _recorder_harness():
    recorder = ReplayRecorder.__new__(ReplayRecorder)
    tracker = IntegralStateTracker(2, 29, "cpu")
    command = SimpleNamespace(
        frame=torch.tensor([0, 0]),
        library=SimpleNamespace(frame_clip=torch.zeros(32, dtype=torch.long)),
    )
    action_term = SimpleNamespace(executed_raw=torch.zeros(2, 29))
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        action_manager=SimpleNamespace(action=torch.zeros(2, 29), get_term=lambda _: action_term),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_buf=torch.zeros(2),
        reset_buf=torch.zeros(2, dtype=torch.bool),
        reset_terminated=torch.zeros(2, dtype=torch.bool),
    )
    recorder._env = env
    recorder.state_io = _StateIO(torch.zeros(2, 71))
    recorder.ctx = _Context()
    recorder.command_name = "motion"
    recorder.action_name = "joint_pos"
    recorder.buffer = ReplayBuffer(2, 16, 71, 29, 33, "cpu", integral_dim=29)
    recorder.integral_state = tracker.state
    recorder.integral_reset_on_episode = True
    recorder.prev_s = torch.zeros(2, 71)
    recorder.prev_xi = torch.zeros(2, 33)
    recorder.prev_frame = torch.zeros(2, dtype=torch.long)
    recorder.prev_clip = torch.zeros(2, dtype=torch.long)
    recorder.ep_start = torch.ones(2, dtype=torch.bool)
    recorder.enabled = True
    return recorder, tracker, env, command


def test_tracker_shared_state_identity_is_stable():
    tracker = IntegralStateTracker(2, 29, "cpu")
    recorder_state = tracker.state
    ptr = recorder_state.data_ptr()
    tracker.update(torch.ones_like(tracker.state), 0.02, 2.0, 0.5)
    assert tracker.state is recorder_state
    assert recorder_state.data_ptr() == ptr
    assert torch.allclose(recorder_state, torch.full_like(recorder_state, 0.02))


def test_recorder_tracks_actor_z_t_reset_and_terminal_rows():
    recorder, tracker, env, command = _recorder_harness()
    actor_inputs = []

    def actor_step(z_t):
        actor_inputs.append(z_t.clone())
        return torch.zeros(2, 29)

    ids = torch.tensor([0, 1])
    recorder.state_io.state[:, 0] = torch.tensor([10.0, 20.0])
    recorder.record_post_reset(ids)

    # t=0: the actor and the row written inside env.step both see z_0 = 0.
    actor_step(tracker.state)
    actor_z0 = actor_inputs[-1]
    env.action_manager.action[:] = torch.zeros(2, 29)
    recorder.state_io.state[:, 0] = torch.tensor([11.0, 21.0])
    command.frame += 1
    recorder.record_post_step()
    assert torch.equal(recorder.buffer.integral[:, 0, :], actor_z0)
    tracker.update(torch.stack([torch.ones(29), torch.full((29,), 2.0)]), 0.02, 2.0, 0.5)

    # t=1: actor and recorder use the same nonzero z_1 before z_2 is committed.
    actor_step(tracker.state)
    actor_z1 = actor_inputs[-1]
    assert torch.allclose(actor_z1[0], torch.full((29,), 0.02))
    assert torch.allclose(actor_z1[1], torch.full((29,), 0.04))
    env.action_manager.action[:] = torch.ones(2, 29)
    recorder.state_io.state[:, 0] = torch.tensor([12.0, 22.0])
    command.frame += 1
    recorder.record_post_step()
    assert torch.equal(recorder.buffer.integral[:, 1, :], actor_z1)
    tracker.update(torch.ones_like(tracker.state), 0.02, 2.0, 0.5)

    # Env 0 terminates: its transition retains z_t, its terminal state has zero by definition,
    # and only that env is reset. Env 1 keeps accumulating independently.
    env.reset_buf[0] = True
    env.reset_terminated[0] = True
    env.action_manager.action[0] = 3
    recorder.state_io.state[0, 0] = 13
    command.frame[0] += 1
    z_before_done = tracker.state.clone()
    recorder.record_pre_reset(torch.tensor([0]))
    terminal_idx = int(recorder.buffer.ptr[0] - 1)
    transition_idx = terminal_idx - 1
    assert torch.equal(recorder.buffer.integral[0, transition_idx], z_before_done[0])
    assert torch.count_nonzero(recorder.buffer.integral[0, terminal_idx]) == 0
    recorder.state_io.state[0, 0] = 100
    command.frame[0] = 0
    recorder.record_post_reset(torch.tensor([0]))
    assert torch.count_nonzero(tracker.state[0]) == 0
    assert torch.equal(tracker.state[1], z_before_done[1])

    # First transition of the new episode is ep_start and records zero; replay starts return
    # the same integral that was recorded beside each sampled state.
    env.reset_buf.zero_()
    env.reset_terminated.zero_()
    env.action_manager.action[0] = 4
    recorder.state_io.state[:, 0] = torch.tensor([101.0, 23.0])
    command.frame[1] += 1
    recorder.record_post_step()
    new_ep_row = int((recorder.buffer.s[0, :, 0] == 100).nonzero()[0])
    assert recorder.buffer.ep_start[0, new_ep_row]
    assert torch.count_nonzero(recorder.buffer.integral[0, new_ep_row]) == 0
    starts = recorder.buffer.sample_starts(256, hist=2)
    for i in range(starts.state.shape[0]):
        marker = float(starts.state[i, 0])
        env_id = 0 if marker in {10.0, 11.0, 12.0, 100.0} else 1
        row_values = recorder.buffer.s[env_id, :, 0]
        row = int((row_values == starts.state[i, 0]).nonzero()[0])
        assert torch.equal(starts.integral[i], recorder.buffer.integral[env_id, row])


class _PolicyProbe:
    def __init__(self):
        self.cfg = SimpleNamespace(history=1)
        self.control_dt_s = 0.02
        self.integral_options = {"enabled": True, "tau_s": 2.0, "clip_abs": 0.5, "actor_input": True}
        self.received = []

    def __call__(self, policy_obs, goal, integral=None):
        self.received.append(integral.clone())
        return torch.zeros(policy_obs.shape[0], 29)


def test_eval_policy_runner_uses_delayed_error_and_resets_each_environment():
    policy = _PolicyProbe()
    command = SimpleNamespace(
        frame=torch.tensor([0, 4]),
        library=SimpleNamespace(frame_clip=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])),
    )
    runner = IntegralPolicyRunner(policy, command)
    observation = {"policy": torch.zeros(2, 29), "goal": torch.ones(2, 68)}

    runner(observation)
    assert torch.count_nonzero(policy.received[-1]) == 0

    command.frame[:] = torch.tensor([1, 5])
    runner(observation)
    expected = torch.full((2, 29), 0.02)
    assert torch.allclose(policy.received[-1], expected)

    # Env 0 auto-resets to frame 0; env 1 advances normally and keeps its own memory.
    command.frame[:] = torch.tensor([0, 6])
    runner(observation)
    beta = torch.exp(torch.tensor(-0.01))
    assert torch.count_nonzero(policy.received[-1][0]) == 0
    assert torch.allclose(policy.received[-1][1], torch.full((29,), float(0.02 * beta + 0.02)))
