"""Paper objectives on source corpora; the final target is evaluated once."""

import hashlib
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from .config import Config


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_run(cfg):
    from .data import load_manifest

    manifest = load_manifest(cfg.manifest)
    sources = [r for r in manifest["records"] if r["split"] == "source"]
    targets = [r for r in manifest["records"] if r["split"] == "target"]
    domains = sorted({r["domain"] for r in sources})
    if not sources or cfg.batch_size < len(domains):
        raise ValueError("Batch size must cover all nonempty source domains")
    if cfg.method in ("mldg", "damldg") and len(domains) < 2:
        raise ValueError("Meta-learning needs at least two source domains")
    if cfg.method == "rsc" and cfg.batch_size < 2:
        raise ValueError("RSC requires at least two examples per batch")
    if not targets and manifest["representation"] != "spectrogram":
        raise ValueError("The held-out target is required for evaluation")
    if cfg.initialization == "pretrained" and not Path(cfg.encoder).is_file():
        raise FileNotFoundError(
            f"Prepare the encoder first: python scripts/prepare_model.py (missing {cfg.encoder})"
        )
    if {r["domain"] for r in targets} & set(domains):
        raise ValueError("A final target domain cannot enter source training")
    from .data import load_features

    example = load_features(
        sources[0], manifest["representation"], False, np.random.default_rng(0), False
    )
    if example.ndim != 3 or example.shape[0] not in (1, 3):
        raise ValueError(
            "ResNet-18 expects one-channel audio/features or three-channel images"
        )
    singleton = example.shape[1] <= 32 and example.shape[2] <= 32
    if singleton and cfg.method == "rsc":
        raise ValueError(
            "RSC spatial challenges require more than a 1x1 feature map; use Deep-All or DA-MLDG for TRILLsson"
        )
    inner_count = (
        cfg.batch_size - __import__("math").ceil(cfg.batch_size / len(domains))
        if cfg.method in ("mldg", "damldg")
        else cfg.batch_size
    )
    if singleton and (inner_count < 2 or cfg.batch_size // len(domains) < 2):
        raise ValueError(
            "Increase batch size: small feature grids require at least two examples per source domain for BatchNorm"
        )
    return manifest, sources, targets, domains


@torch.no_grad()
def calibrate_source_bn(model, records, representation, batch_size, seed, device):
    """Refresh running statistics for the final weights using sources only.

    A complete domain-balanced source pass uses cumulative BatchNorm averages,
    deterministically seeded crops and no augmentation. Parameters and gradients never change;
    target examples are rejected before any state is changed. This optional
    inference safeguard applies identically to every method.
    """
    from .data import load_features, make_batches

    rng = np.random.default_rng(seed)
    batches = make_batches(records, batch_size, rng)
    layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not layers:
        return {"batches": 0, "source_examples": len(records)}
    modes = [(module, module.training) for module in model.modules()]
    momenta = [(module, module.momentum) for module in layers]
    previous = {name: value.clone() for name, value in model.named_buffers()}
    succeeded = False
    try:
        model.eval()
        for module in layers:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        for rows in batches:
            x = np.stack(
                [load_features(row, representation, True, rng, False) for row in rows]
            )
            logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite source calibration logits")
        succeeded = True
    finally:
        for module, momentum in momenta:
            module.momentum = momentum
        for module, training in modes:
            module.training = training
        if not succeeded:
            for name, value in model.named_buffers():
                value.copy_(previous[name])
    return {"batches": len(batches), "source_examples": len(records)}


@torch.no_grad()
def evaluate(model, records, representation, device="cpu"):
    from .data import load_features

    if not records:
        return None
    model.eval()
    confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.long)
    total_loss = 0.0
    for row in records:
        x = load_features(row, representation, False, np.random.default_rng(0), False)
        x = torch.as_tensor(
            np.ascontiguousarray(x), dtype=torch.float32, device=device
        ).unsqueeze(0)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        target = torch.tensor([row["label"]], device=device)
        total_loss += float(torch.nn.functional.cross_entropy(logits, target))
        confusion[row["label"], int(logits.argmax(1))] += 1
    present = confusion.sum(1) > 0
    recalls = confusion.diag()[present] / confusion.sum(1)[present]
    return {
        "accuracy": float(confusion.diag().sum() / len(records)),
        "balanced_accuracy": float(recalls.mean()),
        "loss": total_loss / len(records),
        "num_examples": len(records),
        "confusion_matrix": confusion.tolist(),
    }


def run_trial(cfg: Config, trial_id: int):
    from .data import load_features, make_batches
    from .model import build_model, save_encoder
    from .method import train_step, rsc_step

    if type(trial_id) is not int or not 0 <= trial_id < cfg.trials:
        raise ValueError("Invalid trial ID")
    seed = cfg.seed + trial_id
    seed_everything(seed, cfg.threads)
    manifest, source, target, domains = prepare_run(cfg)
    output = Path(cfg.output) / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=False)
    representation = manifest["representation"]
    rng = np.random.default_rng(seed)
    sample = load_features(source[0], representation, False, rng, False)
    in_channels = sample.shape[0]
    model = build_model(
        num_classes=len(manifest["classes"]),
        in_channels=in_channels,
        rsc=cfg.method == "rsc",
        encoder_path=cfg.encoder if cfg.initialization == "pretrained" else None,
    ).to(cfg.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum
    )
    history = []
    started = time.monotonic()
    provenance = {
        "manifest_sha256": digest(cfg.manifest),
        "manifest": manifest,
        "encoder_sha256": digest(cfg.encoder)
        if cfg.initialization == "pretrained"
        else None,
    }
    write_json(
        output / "config.json",
        cfg.as_dict() | {"trial_id": trial_id, "trial_seed": seed},
    )
    write_json(output / "data.json", provenance)
    for epoch in range(cfg.epochs):
        totals = {"loss": 0.0, "train_loss": 0.0, "test_loss": 0.0, "gamma": 0.0}
        steps = make_batches(source, cfg.batch_size, rng)
        for rows in steps:
            inputs = [
                load_features(r, representation, True, rng, cfg.augment) for r in rows
            ]
            x = torch.as_tensor(
                np.stack(inputs), dtype=torch.float32, device=cfg.device
            )
            y = torch.tensor([r["label"] for r in rows], device=cfg.device)
            d = torch.tensor(
                [domains.index(r["domain"]) for r in rows], device=cfg.device
            )
            if cfg.method == "rsc":
                values = rsc_step(
                    model, optimizer, x, y, rng, max_grad_norm=cfg.max_grad_norm
                )
            else:
                heldout = (
                    int(rng.integers(len(domains)))
                    if cfg.method in ("damldg", "mldg")
                    else None
                )
                values = train_step(
                    model,
                    optimizer,
                    x,
                    y,
                    d,
                    meta_test_domain=heldout,
                    inner_lr=cfg.inner_lr,
                    method=cfg.method,
                    max_grad_norm=cfg.max_grad_norm,
                )
            for name in totals:
                totals[name] += float(values[name])
        history.append(
            {
                "epoch": epoch + 1,
                "steps": len(steps),
                **{k: v / len(steps) for k, v in totals.items()},
            }
        )
        write_json(output / "history.json", history)
        print(json.dumps({"trial": trial_id, "seed": seed, **history[-1]}), flush=True)
    calibration = (
        calibrate_source_bn(
            model, source, representation, cfg.batch_size, seed, cfg.device
        )
        if cfg.calibrate_bn
        else None
    )
    metrics = {
        "trial_id": trial_id,
        "seed": seed,
        "method": cfg.method,
        "epochs": cfg.epochs,
        "source_domains": domains,
        "source_examples": len(source),
        "representation": representation,
        "initialization": cfg.initialization,
        "batchnorm_calibration": calibration,
        "target": evaluate(model, target, representation, cfg.device),
        "seconds": time.monotonic() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": cfg.device,
        },
    }
    checkpoint = {
        "format_version": 1,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": {
            "num_classes": len(manifest["classes"]),
            "in_channels": in_channels,
            "rsc": cfg.method == "rsc",
        },
        "config": cfg.as_dict(),
        "trial_id": trial_id,
        "seed": seed,
        "classes": manifest["classes"],
        "provenance": provenance,
        "metrics": metrics,
        "history": history,
    }
    temporary = output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "checkpoint.pt")
    save_encoder(model, output / "encoder.pt")
    metrics["checkpoint_sha256"] = digest(output / "checkpoint.pt")
    write_json(output / "metrics.json", metrics)
    return metrics


def load_checkpoint(path, device="cpu"):
    from .model import build_model

    content = torch.load(path, map_location="cpu", weights_only=True)
    if content.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = build_model(**content["model_config"])
    model.load_state_dict(content["model"], strict=True)
    return model.to(device).eval(), content
