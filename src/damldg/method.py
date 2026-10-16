"""Native PyTorch second-order MLDG, DA-MLDG, Deep-All and source RSC.

Meta-test is a held-out *source* domain, never the deployment target. The
functional inner step differentiates through adaptation and the energy margin.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F
from torch.func import functional_call


@contextmanager
def _mode(model, training):
    modes = [(module, module.training) for module in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def _batch(model, x, y, domains=None):
    parameter = next(model.parameters())
    x = torch.as_tensor(x, device=parameter.device, dtype=parameter.dtype)
    y = torch.as_tensor(y, device=parameter.device, dtype=torch.long)
    if len(x) == 0 or y.ndim != 1 or len(y) != len(x):
        raise ValueError("Expected a nonempty batch with one label per example")
    if domains is None:
        return x, y
    domains = torch.as_tensor(domains, device=parameter.device)
    if domains.ndim != 1 or len(domains) != len(x):
        raise ValueError("Expected one domain per example")
    return x, y, domains


def domain_loss(logits, labels, domains):
    """Equation 2: mean CE within each domain, then equal mean over domains."""
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    domains = torch.as_tensor(domains, device=logits.device)
    if (
        len(logits) == 0
        or labels.shape != domains.shape
        or labels.ndim != 1
        or len(labels) != len(logits)
    ):
        raise ValueError("Domain loss needs aligned, nonempty logits/labels/domains")
    losses = F.cross_entropy(logits, labels, reduction="none")
    return torch.stack(
        [losses[domains == domain].mean() for domain in domains.unique()]
    ).mean()


def energy_gap(train_logits, test_logits):
    """Equation 3 at T=1: absolute difference of mean log-sum-exp logits."""
    return (
        torch.logsumexp(train_logits, dim=-1).mean()
        - torch.logsumexp(test_logits, dim=-1).mean()
    ).abs()


def _copy_buffers(buffers):
    return {name: value.detach().clone() for name, value in buffers.items()}


def meta_gradients(
    model,
    x,
    y,
    domains,
    meta_test_domain,
    inner_lr=0.01,
    method="damldg",
    training=True,
    first_order=False,
):
    """Return (gradients, source-train buffers, tensor metrics), without mutation.

    Gradients follow the order of trainable ``model.named_parameters()``. BN
    buffers are cloned for every functional forward. Only source-train running
    state is returned; temporary meta-test updates cannot leak into it. Energy
    uses the adapted weights with source-train running statistics in eval mode,
    matching the source implementation. ``first_order`` is an audit ablation.
    Original parameter .grad fields, buffers, and per-module modes are preserved.
    """
    if method not in {"mldg", "damldg"}:
        raise ValueError("Unknown meta method")
    x, y, domains = _batch(model, x, y, domains)
    if meta_test_domain is None:
        unique = domains.unique()
        meta_test_domain = unique[torch.randint(len(unique), ())]
    heldout = domains == meta_test_domain
    if not heldout.any() or heldout.all():
        raise ValueError(
            "Meta-train and meta-test need disjoint nonempty source domains"
        )
    params = dict(model.named_parameters())
    trainable = {name: value for name, value in params.items() if value.requires_grad}
    initial_buffers = _copy_buffers(dict(model.named_buffers()))
    source_buffers = _copy_buffers(initial_buffers)
    xt, yt, dt = x[~heldout], y[~heldout], domains[~heldout]
    xv, yv, dv = x[heldout], y[heldout], domains[heldout]
    with torch.enable_grad():
        with _mode(model, training):
            logits = functional_call(
                model, (params, source_buffers), (xt,), strict=True
            )
        train_loss = domain_loss(logits, yt, dt)
        inner = torch.autograd.grad(
            train_loss,
            tuple(trainable.values()),
            create_graph=not first_order,
            retain_graph=True,
        )
        adapted = dict(params)
        adapted.update(
            {
                name: value - inner_lr * gradient
                for (name, value), gradient in zip(trainable.items(), inner)
            }
        )
        train_state = _copy_buffers(source_buffers)
        if method == "damldg":
            with _mode(model, False):
                a = functional_call(
                    model, (adapted, _copy_buffers(train_state)), (xt,), strict=True
                )
                b = functional_call(
                    model, (adapted, _copy_buffers(train_state)), (xv,), strict=True
                )
            gamma = energy_gap(a, b)
        else:
            gamma = train_loss.new_tensor(1.0)
        with _mode(model, training):
            test_logits = functional_call(
                model, (adapted, _copy_buffers(train_state)), (xv,), strict=True
            )
        test_loss = domain_loss(test_logits, yv, dv)
        loss = train_loss + gamma * test_loss
        gradients = torch.autograd.grad(loss, tuple(trainable.values()))
    return (
        list(gradients),
        train_state,
        {
            "loss": loss,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "gamma": gamma,
        },
    )


def _metrics(details):
    if not all(torch.isfinite(value.detach()).all() for value in details.values()):
        raise FloatingPointError("Nonfinite training objective")
    return {name: float(value.detach()) for name, value in details.items()}


def _apply(model, optimizer, gradients, max_grad_norm=0.0):
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Nonfinite training gradient")
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(
        (p for p in model.parameters() if p.requires_grad), gradients
    ):
        parameter.grad = gradient.detach()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
    optimizer.step()


def train_step(
    model,
    optimizer,
    x,
    y,
    domains,
    meta_test_domain=None,
    inner_lr=0.01,
    method="damldg",
    max_grad_norm=0.0,
):
    """One optimization step; meta methods commit only source-train BN state."""
    if method in {"mldg", "damldg"}:
        gradients, state, details = meta_gradients(
            model, x, y, domains, meta_test_domain, inner_lr, method
        )
        result = _metrics(details)
        _apply(model, optimizer, gradients, max_grad_norm)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(state[name])
        return result
    if method == "rsc":
        return rsc_step(model, optimizer, x, y, max_grad_norm=max_grad_norm)
    if method != "deepall":
        raise ValueError("Unknown method")
    x, y = _batch(model, x, y)
    with _mode(model, True):
        loss = F.cross_entropy(model(x), y)
    zero = loss.new_zeros(())
    result = _metrics(
        {"loss": loss, "train_loss": loss, "test_loss": zero, "gamma": zero}
    )
    gradients = torch.autograd.grad(
        loss, tuple(p for p in model.parameters() if p.requires_grad)
    )
    _apply(model, optimizer, gradients, max_grad_norm)
    return result


def rsc_step(
    model,
    optimizer,
    x,
    y,
    rng=None,
    percentile=66.0,
    batch_fraction=0.33,
    max_grad_norm=0.0,
):
    """Port the authors' two-stage RSC baseline with corrected label alignment.

    First update the ordinary remainder, then challenge the first fraction with
    a spatial (5/9 probability) or channel sensitivity mask. The source used an
    extra linear layer and a 66th-percentile threshold. This preserves that
    variant rather than claiming equivalence to every RSC implementation.
    """
    x, y = _batch(model, x, y)
    if len(x) < 2 or not 0 < batch_fraction < 1 or not 0 <= percentile <= 100:
        raise ValueError(
            "RSC needs >=2 examples, a fraction in (0,1), and a percentile in [0,100]"
        )
    n = min(len(x) - 1, max(1, int(len(x) * batch_fraction)))
    with _mode(model, True):
        plain = F.cross_entropy(model(x[n:]), y[n:])
    _metrics({"loss": plain})
    _apply(
        model,
        optimizer,
        torch.autograd.grad(
            plain, tuple(p for p in model.parameters() if p.requires_grad)
        ),
        max_grad_norm,
    )
    with _mode(model, False):
        features = model.forward_features(x[:n])
        score = model.classify(features).gather(1, y[:n, None]).sum()
        sensitivity = torch.autograd.grad(score, features)[0].detach()
    draw = int(rng.integers(9)) if rng is not None else int(torch.randint(9, ()).item())
    if draw <= 4:
        importance = sensitivity.mean(dim=1, keepdim=True)
        threshold = torch.quantile(importance.flatten(1), percentile / 100, dim=1).view(
            -1, 1, 1, 1
        )
    else:
        importance = sensitivity.mean(dim=(2, 3), keepdim=True)
        threshold = torch.quantile(importance.flatten(1), percentile / 100, dim=1).view(
            -1, 1, 1, 1
        )
    mask = (importance < threshold).to(features.dtype)
    with _mode(model, True):
        challenged = F.cross_entropy(
            model.classify(model.forward_features(x[:n]) * mask), y[:n]
        )
    result = _metrics(
        {
            "loss": (plain.detach() + challenged) / 2,
            "train_loss": plain.detach(),
            "test_loss": challenged,
            "gamma": challenged.new_zeros(()),
        }
    )
    _apply(
        model,
        optimizer,
        torch.autograd.grad(
            challenged, tuple(p for p in model.parameters() if p.requires_grad)
        ),
        max_grad_norm,
    )
    return result
