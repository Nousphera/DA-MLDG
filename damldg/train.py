"""Seeded source-domain training; held-out targets are evaluated only at the end."""

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
import tensorflow as tf
from .data import EMOTIONS, load_features, make_batches
from .model import build_model
from .method import train_step, rsc_step


@dataclass(frozen=True)
class Config:
    dataset: str = "synthetic"
    manifest: str = "data/scn1.json"
    representation: str = "audio"
    method: str = "damldg"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.05
    momentum: float = 0.9
    inner_lr: float = 0.01
    augment: bool = True
    seed: int = 2021
    steps_per_epoch: int | None = None
    pretrained_encoder: str | None = None
    synthetic_per_domain: int = 2
    output: str = "runs/experiment"

    def __post_init__(self):
        if self.dataset not in {"synthetic", "manifest"}:
            raise ValueError("dataset must be synthetic or manifest")
        if self.representation not in {"audio", "spectrogram", "features", "image"}:
            raise ValueError("Unknown representation")
        if self.method not in {"deepall", "rsc", "mldg", "damldg"}:
            raise ValueError("Unknown method")
        if (
            min(self.epochs, self.batch_size, self.synthetic_per_domain) < 1
            or self.seed < 0
        ):
            raise ValueError("Invalid counts")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        if self.learning_rate <= 0 or self.inner_lr < 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid optimizer parameters")


def load_config(path, **overrides):
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text())
    # Shipped configs live under configs/; keep their data paths tied to the
    # checkout independently of the caller's working directory.
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for key in ("manifest", "pretrained_encoder", "output"):
        if data.get(key) and not Path(data[key]).is_absolute():
            data[key] = str((base / data[key]).resolve())
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data)


def make_data(config):
    if config.dataset == "synthetic":
        rng = np.random.default_rng(config.seed)
        rows = []
        for i in range(4):
            for j in range(config.synthetic_per_domain):
                label = j % 5
                feature = rng.normal(i * 0.2 + label * 0.1, 1.0, (32, 32, 1)).astype(
                    np.float32
                )
                rows.append(
                    {
                        "id": f"domain-{i}/{j}",
                        "domain": f"domain-{i}",
                        "label": label,
                        "split": "target" if i == 3 else "source",
                        "_features": feature,
                    }
                )
        classes = EMOTIONS
    else:
        manifest = json.loads(Path(config.manifest).read_text())
        from .preparation import check_records
        rows = manifest["records"]
        check_records(rows, manifest["classes"], ("source",) if config.representation == "spectrogram" else ("source", "target"))
        classes = manifest["classes"]
        if manifest["representation"] != config.representation:
            raise ValueError("Manifest/config representation mismatch")
        if len(classes) < 2 or any(
            r["split"] not in {"source", "target"} or not 0 <= r["label"] < len(classes)
            for r in rows
        ):
            raise ValueError("Invalid manifest class or split")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Manifest repeats example IDs")
        paths = [str(Path(r["path"]).resolve()) for r in rows]
        if len(set(paths)) != len(paths):
            raise ValueError("Manifest repeats files across splits")
    source = [r for r in rows if r["split"] == "source"]
    target = [r for r in rows if r["split"] == "target"]
    domains = sorted({r["domain"] for r in source})
    if {r["domain"] for r in source} & {r["domain"] for r in target}:
        raise ValueError("Target domain leaked into source domains")
    if not source or (config.method in {"damldg", "mldg"} and len(domains) < 2):
        raise ValueError("Meta-learning requires >=2 source domains")
    return source, target, classes


def evaluate(model, records, representation):
    if not records:
        return None
    correct = 0
    losses = []
    for row in records:
        x = load_features(
            row, representation, False, np.random.default_rng(0), augment=False
        )
        logits = model(x[None], training=False)
        correct += int(int(tf.argmax(logits[0])) == row["label"])
        losses.append(
            float(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=[row["label"]], logits=logits
                )[0]
            )
        )
    return {
        "accuracy": correct / len(records),
        "loss": float(np.mean(losses)),
        "num_examples": len(records),
    }


def run(config):
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    source, target, classes = make_data(config)
    domains = sorted({r["domain"] for r in source})
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "partitions.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "records": [
                    {k: v for k, v in r.items() if k != "_features"}
                    for r in source + target
                ],
            },
            indent=2,
        )
        + "\n"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [d.name for d in tf.config.list_physical_devices()],
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }
    for name in ["manifest", "pretrained_encoder"]:
        path = getattr(config, name)
        if path and (name != "manifest" or config.dataset != "synthetic"):
            environment[name + "_sha256"] = hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    rng = np.random.default_rng(config.seed)
    example = load_features(source[0], config.representation, False, rng, False)
    model = build_model(
        (None, None, example.shape[-1]),
        len(classes),
        config.pretrained_encoder,
        config.method == "rsc",
    )
    optimizer = tf.keras.optimizers.SGD(config.learning_rate, momentum=config.momentum)
    history = []
    start = time.monotonic()
    for epoch in range(config.epochs):
        batches = make_batches(source, config.batch_size, rng, config.steps_per_epoch)
        steps = []
        for rows in batches:
            x = np.stack(
                [
                    load_features(r, config.representation, True, rng, config.augment)
                    for r in rows
                ]
            )
            y = np.asarray([r["label"] for r in rows], np.int32)
            d = np.asarray([domains.index(r["domain"]) for r in rows], np.int32)
            if config.method == "rsc":
                result = rsc_step(model, optimizer, x, y, rng)
            else:
                heldout = (
                    int(rng.integers(len(domains)))
                    if config.method != "deepall"
                    else None
                )
                result = train_step(
                    model, optimizer, x, y, d, heldout, config.inner_lr, config.method
                )
                result["meta_test_domain"] = (
                    domains[heldout] if heldout is not None else None
                )
            steps.append(result)
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean([r["loss"] for r in steps])),
            "steps": steps,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    metrics = {
        "dataset": config.dataset,
        "representation": config.representation,
        "initialization": "pretrained" if config.pretrained_encoder else "random",
        "method": config.method,
        "seconds": time.monotonic() - start,
        "epochs": history,
        "target": evaluate(model, target, config.representation),
    }
    model.save(output / "model.keras")
    model.get_layer("encoder").save(output / "encoder.keras")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
