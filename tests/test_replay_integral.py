import torch

from supermimic.data.replay_buffer import ReplayBuffer


def test_sampled_integral_is_aligned_and_episode_local():
    buf = ReplayBuffer(1, 12, 1, 1, 1, "cpu", integral_dim=1)

    def add(marker, z, start=False, terminal=False):
        buf.add(
            torch.tensor([0]),
            s=torch.tensor([[marker]], dtype=torch.float),
            a=torch.zeros(1, 1),
            a_exec=torch.zeros(1, 1),
            xi=torch.zeros(1, 1),
            r=torch.zeros(1),
            done=torch.zeros(1),
            frame=torch.tensor([marker]),
            clip=torch.tensor([0]),
            terminal=torch.tensor([terminal]),
            ep_start=torch.tensor([start]),
            integral=torch.tensor([[z]], dtype=torch.float),
        )

    add(10, 0.1, start=True)
    add(11, 0.2)
    add(12, 0.0, terminal=True)
    buf.end_episode(torch.tensor([0]))
    add(20, 0.0, start=True)
    batch = buf.sample_starts(128, hist=2)
    markers = batch.state[:, 0]
    expected = torch.where(markers < 20, (markers - 10) * 0.1 + 0.1, torch.zeros_like(markers))
    assert torch.equal(batch.integral[:, 0], expected)
    # Any history backfill at an episode start uses that same episode's state row.
    for i, marker in enumerate(markers.tolist()):
        if marker == 20:
            assert batch.hist_states[:, i, 0].eq(20).all()
