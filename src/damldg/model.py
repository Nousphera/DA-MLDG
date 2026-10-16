"""ResNet-18 and strict, classifier-independent encoder checkpoints.

The original convolution/block topology is the standard torchvision ResNet-18.
BatchNorm uses the source Keras epsilon and running-statistic update rate;
ReLUs are out of place so full second-order meta-gradients remain valid.
"""

from collections import OrderedDict
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class _HigherOrderMaxPool2d(nn.MaxPool2d):
    """Retain pooling indices: MPS's indices-free kernel has no double backward."""

    def forward(self, x):
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )[0]


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, rsc=False):
        super().__init__()
        if num_classes < 2 or in_channels < 1:
            raise ValueError("Need at least two classes and one input channel")
        backbone = resnet18(
            weights=None,
            norm_layer=partial(nn.BatchNorm2d, eps=1.001e-5, momentum=0.01),
        )
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, 7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(
                backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
            )
        backbone.maxpool = _HigherOrderMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(
            OrderedDict(
                (name, module)
                for name, module in backbone.named_children()
                if name not in {"avgpool", "fc"}
            )
        )
        layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        if rsc:
            # The source RSC baseline adds a linear 512-unit layer, without ReLU.
            layers.append(nn.Linear(512, 512))
        layers.append(nn.Linear(512, num_classes))
        self.head = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.rsc = rsc
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward_features(self, x):
        return self.encoder(x)

    def classify(self, features):
        return self.head(features)

    def forward(self, x):
        return self.classify(self.forward_features(x))


def build_model(num_classes=5, in_channels=1, rsc=False, encoder_path=None):
    """Build a real ResNet-18 accepting NCHW tensors; no weights are downloaded."""
    model = ResNet18Classifier(num_classes, in_channels, rsc)
    if encoder_path is not None:
        load_encoder(model, encoder_path)
    return model


def save_encoder(model, path):
    """Save only encoder parameters and BN buffers, never the task classifier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "damldg.encoder.v1",
            "architecture": "resnet18",
            "in_channels": model.in_channels,
            "state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.encoder.state_dict().items()
            },
        },
        path,
    )
    return path


def load_encoder(model, path):
    """Validate the whole artifact before changing any encoder state.

    Full-model checkpoints, partial keys, and channel/shape mismatches are errors.
    A speaker-classification encoder can therefore transfer to any emotion head.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "damldg.encoder.v1"
        or checkpoint.get("architecture") != "resnet18"
    ):
        raise ValueError("Expected a damldg.encoder.v1 ResNet-18 encoder checkpoint")
    if checkpoint.get("in_channels") != model.in_channels:
        raise ValueError("Encoder input channel mismatch")
    state = checkpoint.get("state_dict")
    expected = model.encoder.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Encoder checkpoint keys must match exactly")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Encoder tensor shape or dtype mismatch: {name}")
    model.encoder.load_state_dict(state, strict=True)
    return model
