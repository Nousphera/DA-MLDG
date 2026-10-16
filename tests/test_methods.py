"""Numerical and state-isolation checks for the paper's outer objective."""

import copy
import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call
from damldg.method import domain_loss, energy_gap, meta_gradients, train_step, rsc_step
from damldg.model import build_model


def toy():
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, 0.7], [-0.2, 0.1]], dtype=torch.double))
    return model


def batch():
    return (
        torch.tensor(
            [[0.2, 0.7], [1.0, -0.5], [1.3, 0.1], [-0.3, 0.8]], dtype=torch.double
        ),
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0, 0, 1, 1]),
    )


def test_domain_mean_and_energy():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    assert domain_loss(
        logits, torch.tensor([0, 0, 0]), torch.tensor([0, 0, 1])
    ).item() == pytest.approx((np.log(2) + np.log1p(np.exp(-2))) / 2)
    assert energy_gap(
        torch.zeros(1, 2), torch.full((1, 2), 2.0)
    ).item() == pytest.approx(2.0)


@pytest.mark.parametrize("method", ["mldg", "damldg"])
def test_full_meta_gradient_matches_finite_difference(method):
    model = toy()
    x, y, d = batch()
    grads, _, _ = meta_gradients(
        model, x, y, d, 1, inner_lr=0.3, method=method, training=False
    )
    numeric = torch.zeros_like(model.weight)
    original = model.weight.detach().clone()
    eps = 1e-5
    for i in range(2):
        for j in range(2):
            vals = []
            for sign in [1, -1]:
                with torch.no_grad():
                    model.weight.copy_(original)
                    model.weight[i, j] += sign * eps
                vals.append(
                    meta_gradients(model, x, y, d, 1, 0.3, method, training=False)[2][
                        "loss"
                    ].item()
                )
            numeric[i, j] = (vals[0] - vals[1]) / (2 * eps)
    with torch.no_grad():
        model.weight.copy_(original)
    torch.testing.assert_close(grads[0], numeric, rtol=1e-6, atol=1e-8)


def test_meta_gradient_contains_inner_hessian_and_target_labels():
    model = toy()
    x, y, d = batch()
    full, _, _ = meta_gradients(model, x, y, d, 1, 0.3, "mldg", training=False)
    first, _, _ = meta_gradients(
        model, x, y, d, 1, 0.3, "mldg", training=False, first_order=True
    )
    changed, _, _ = meta_gradients(
        model, x, torch.tensor([0, 1, 0, 1]), d, 1, 0.3, "mldg", training=False
    )
    assert (full[0] - first[0]).abs().max() > 0.001
    assert (full[0] - changed[0]).abs().max() > 0.01
    before = model.weight.detach().clone()
    train_step(
        model, torch.optim.SGD(model.parameters(), lr=0.05), x, y, d, 1, 0.3, "mldg"
    )
    torch.testing.assert_close(model.weight, before - 0.05 * full[0])


def test_energy_margin_derivative_contributes_to_outer_gradient():
    model = toy()
    x, y, d = batch()
    full, _, _ = meta_gradients(model, x, y, d, 1, 0.3, "damldg", training=False)
    p = dict(model.named_parameters())
    train_loss = domain_loss(functional_call(model, p, (x[:2],)), y[:2], d[:2])
    inner = torch.autograd.grad(train_loss, tuple(p.values()), create_graph=True)
    adapted = {k: v - 0.3 * g for (k, v), g in zip(p.items(), inner)}
    a = functional_call(model, adapted, (x[:2],))
    b = functional_call(model, adapted, (x[2:],))
    gamma = energy_gap(a, b)
    test_loss = domain_loss(b, y[2:], d[2:])
    detached = torch.autograd.grad(
        train_loss + gamma.detach() * test_loss, tuple(p.values()), retain_graph=True
    )
    margin_grad = torch.autograd.grad(gamma, tuple(p.values()))
    torch.testing.assert_close(
        full[0] - detached[0], test_loss.detach() * margin_grad[0]
    )
    assert (full[0] - detached[0]).abs().max() > 0.001


def test_gradient_calls_are_pure_and_commit_only_source_bn_state():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2)).double()
    model.eval()
    model[1].train()  # Preserve deliberately mixed modes.
    x, y, d = batch()
    before = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    grads, state, _ = meta_gradients(model, x, y, d, 1, 0.01, "damldg")
    for k, v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    assert [m.training for m in model.modules()] == modes
    expected = copy.deepcopy(model)
    expected.train()
    expected(x[:2])
    for k, v in expected.named_buffers():
        torch.testing.assert_close(state[k], v, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    train_step(
        model, torch.optim.SGD(model.parameters(), lr=0.001), x, y, d, 1, 0.01, "damldg"
    )
    for k, v in model.named_buffers():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    assert int(model[0].num_batches_tracked) == 1


@pytest.mark.parametrize("method", ["deepall", "mldg", "damldg", "rsc"])
def test_full_resnet_updates_are_finite(method):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = build_model(num_classes=5, rsc=method == "rsc")
    x = torch.randn(4, 1, 64, 64)
    y = torch.tensor([0, 1, 2, 3])
    d = torch.tensor([0, 0, 1, 1])
    before = model.encoder.conv1.weight.detach().clone()
    result = train_step(
        model, torch.optim.SGD(model.parameters(), lr=0.001), x, y, d, 1, 0.01, method
    )
    assert set(result) == {"loss", "train_loss", "test_loss", "gamma"}
    assert all(isinstance(v, float) and np.isfinite(v) for v in result.values())
    assert not torch.equal(before, model.encoder.conv1.weight)


def test_reject_empty_meta_partition():
    x, y, d = batch()
    with pytest.raises(ValueError, match="nonempty"):
        meta_gradients(toy(), x, y, torch.zeros_like(d), 0)


def test_rsc_uses_matching_labels_and_only_challenged_subset():
    # A deterministic, zero-LR run exposes each stage's actual CE without updates.
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = build_model(num_classes=5, rsc=True)
    expected = copy.deepcopy(model)
    x = torch.randn(6, 1, 64, 64)
    y = torch.tensor([4, 3, 2, 1, 0, 4])
    expected.train()
    plain = torch.nn.functional.cross_entropy(expected(x[1:]), y[1:])
    expected.eval()
    features = expected.forward_features(x[:1])
    sensitivity = torch.autograd.grad(
        expected.classify(features).gather(1, y[:1, None]).sum(), features
    )[0]
    draw = int(np.random.default_rng(5).integers(9))
    importance = (
        sensitivity.mean(dim=1, keepdim=True)
        if draw <= 4
        else sensitivity.mean(dim=(2, 3), keepdim=True)
    )
    threshold = torch.quantile(importance.flatten(1), 0.66, dim=1).view(-1, 1, 1, 1)
    mask = (importance < threshold).to(features.dtype)
    expected.train()
    challenged = torch.nn.functional.cross_entropy(
        expected.classify(expected.forward_features(x[:1]) * mask), y[:1]
    )
    result = rsc_step(
        model, torch.optim.SGD(model.parameters(), lr=0), x, y, np.random.default_rng(5)
    )
    assert result["train_loss"] == pytest.approx(plain.item())
    assert result["test_loss"] == pytest.approx(challenged.item())
    assert all(np.isfinite(v) for v in result.values())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_full_resnet_second_order_update():
    torch.manual_seed(7)
    model = build_model().to("mps")
    x = torch.randn(4, 1, 64, 64, device="mps")
    before = model.encoder.conv1.weight.detach().clone()
    metrics = train_step(
        model,
        torch.optim.SGD(model.parameters(), lr=0.001),
        x,
        torch.tensor([0, 1, 2, 3], device="mps"),
        torch.tensor([0, 0, 1, 1], device="mps"),
        1,
        0.01,
        "damldg",
    )
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, model.encoder.conv1.weight)


def test_outer_gradient_norm_clipping_is_explicit():
    from damldg.method import _apply

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    _apply(model, optimizer, [torch.tensor([[30.0, 40.0]])], max_grad_norm=5.0)
    assert torch.linalg.vector_norm(model.weight).item() == pytest.approx(5.0, rel=1e-6)
    with torch.no_grad():
        model.weight.zero_()
    _apply(model, optimizer, [torch.tensor([[30.0, 40.0]])], max_grad_norm=0.0)
    assert torch.linalg.vector_norm(model.weight).item() == pytest.approx(50.0)
