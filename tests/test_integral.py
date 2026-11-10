import pytest
import torch

from supermimic.policy.integral import IntegralStateTracker, update_integral
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg


def test_leaky_integral_formula_and_clipping():
    z = torch.zeros(2, 29)
    e = torch.ones_like(z)
    first, _ = update_integral(z, e, 0.02, 2.0, 10.0)
    beta = torch.exp(torch.tensor(-0.01))
    second, _ = update_integral(first, e, 0.02, 2.0, 10.0)
    assert torch.allclose(first, torch.full_like(first, 0.02))
    assert torch.allclose(second, torch.full_like(second, float(0.02 * beta + 0.02)))
    clipped, sat = update_integral(second, e * 100, 0.02, 2.0, 0.1)
    assert torch.all(clipped <= 0.1) and sat.all()


def test_tracker_independent_reset():
    tracker = IntegralStateTracker(3, 29, "cpu")
    tracker.update(torch.ones_like(tracker.state), 0.02, 2.0, 1.0)
    tracker.reset(torch.tensor([1]))
    assert tracker.state[1].count_nonzero() == 0
    assert tracker.state[0].count_nonzero() == 29
    assert tracker.state[2].count_nonzero() == 29


def test_full_and_detach_gradients_differ():
    err_full = torch.ones(1, 1, requires_grad=True)
    z_full, _ = update_integral(torch.zeros_like(err_full), err_full, 0.02, 2.0, 5.0, "full")
    e2_full = torch.ones_like(z_full, requires_grad=True)
    z_full2, _ = update_integral(z_full, e2_full, 0.02, 2.0, 5.0, "full")
    grad_full = torch.autograd.grad(z_full2.sum(), err_full)[0]
    err_detach = torch.ones(1, 1, requires_grad=True)
    z_detach, _ = update_integral(torch.zeros_like(err_detach), err_detach, 0.02, 2.0, 5.0, "detach")
    e2_detach = torch.ones_like(z_detach, requires_grad=True)
    z_detach2, _ = update_integral(z_detach, e2_detach, 0.02, 2.0, 5.0, "detach")
    grad_detach = torch.autograd.grad(z_detach2.sum(), err_detach, allow_unused=True)[0]
    assert grad_full.item() > 0
    assert grad_detach is None or torch.equal(grad_detach, torch.zeros_like(err_detach))


def test_network_dimensions_and_backward():
    base = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,)))
    enabled = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,), integral_dim=29))
    obs, goal, z = torch.zeros(2, 930), torch.zeros(2, 680), torch.ones(2, 29)
    assert base(obs, goal).shape == (2, 29)
    assert enabled(obs, goal, integral=z).shape == (2, 29)
    enabled(obs, goal, integral=z).sum().backward()
    assert enabled.decoder.mlp.net[0].weight.grad is not None
    critic = Critic(CriticCfg(hidden=(16,), integral_dim=29), pcfg=PolicyCfg())
    y = critic(torch.zeros(2, 71), goal, torch.zeros(2, 33), torch.zeros(2, 4), z)
    assert y.shape == (2,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_integral_agree():
    z = torch.randn(4, 29)
    e = torch.randn_like(z)
    cpu, _ = update_integral(z, e, 0.02, 2.0, 0.5)
    gpu, _ = update_integral(z.cuda(), e.cuda(), 0.02, 2.0, 0.5)
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-6)
