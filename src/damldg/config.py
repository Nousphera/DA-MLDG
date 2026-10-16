"""Validated experiment settings shared by local and Flower trials."""

from dataclasses import asdict, dataclass
import math

METHODS = ("damldg", "mldg", "deepall", "rsc")


@dataclass(frozen=True)
class Config:
    manifest: str = "data/scn7.json"
    method: str = "damldg"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.05
    inner_lr: float = 0.01
    momentum: float = 0.9
    max_grad_norm: float = 1.0
    calibrate_bn: bool = True
    seed: int = 2021
    trials: int = 10
    augment: bool = True
    initialization: str = "pretrained"
    encoder: str = "checkpoints/voxceleb-resnet18.pt"
    output: str = "results/damldg-scn7"
    backend: str = "flower"
    device: str = "cpu"
    threads: int = 2
    registration_timeout: float = 120.0
    reply_timeout: float = 604800.0

    def __post_init__(self):
        if self.method not in METHODS or self.backend not in ("flower", "sequential"):
            raise ValueError("Unknown method or execution backend")
        if self.initialization not in ("pretrained", "random"):
            raise ValueError("Initialization must be pretrained or random")
        if type(self.calibrate_bn) is not bool:
            raise ValueError("calibrate_bn must be a boolean")
        for key in ("epochs", "batch_size", "trials", "threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - self.trials:
            raise ValueError("Seeds must fit the NumPy seed range")
        for key in (
            "learning_rate",
            "inner_lr",
            "momentum",
            "max_grad_norm",
            "registration_timeout",
            "reply_timeout",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate == 0 or self.inner_lr == 0 or self.momentum >= 1:
            raise ValueError(
                "Learning rates must be positive and momentum less than one"
            )
        if min(self.registration_timeout, self.reply_timeout) <= 0:
            raise ValueError("Timeouts must be positive")
        if self.initialization == "pretrained" and not self.encoder:
            raise ValueError(
                "Prepare a pretrained encoder or explicitly choose random initialization"
            )
        if not self.manifest or not self.output:
            raise ValueError("Manifest and output paths are required")
        if self.backend == "flower" and self.device.startswith("cuda:"):
            raise ValueError(
                "Use --device cuda and CUDA_VISIBLE_DEVICES with Flower GPU workers"
            )

    def as_dict(self):
        return asdict(self)


def resolve_device(device):
    import torch

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    parsed = torch.device(device)
    if parsed.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose cpu, cuda, or mps")
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if parsed.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return str(parsed)
