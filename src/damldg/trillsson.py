"""TRILLsson3 waveform frontend and frozen speech embedding network."""

from __future__ import annotations

import math
import torch
from torch import nn


class TrillssonFrontend(nn.Module):
    """16 kHz waveform to the official 195 × 80 log-mel contexts."""

    def __init__(self):
        super().__init__()
        frequencies = torch.linspace(0, 8000, 257, dtype=torch.float64)
        mel = 1127.0 * torch.log1p(frequencies / 700.0)
        edges = torch.linspace(
            1127.0 * math.log1p(125.0 / 700.0),
            1127.0 * math.log1p(7500.0 / 700.0),
            82,
            dtype=torch.float64,
        )
        lower = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
        upper = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
        filters = torch.minimum(lower, upper).clamp_min(0)
        filters[0] = 0
        self.mel_filters = filters
        self.window = torch.hann_window(400, periodic=True, dtype=torch.float64)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2 or audio.shape[1] == 0:
            raise ValueError("Expected nonempty audio with shape [batch, samples]")
        if not torch.isfinite(audio).all():
            raise ValueError("Audio contains nonfinite samples")
        # The reference frontend casts to float32 before its float64 spectrum.
        audio = audio.float().double()
        n = audio.shape[1]
        if n < 32000:
            indices = torch.arange(32000, device=audio.device) % (2 * n)
            indices = torch.where(indices < n, indices, 2 * n - indices - 1)
            audio = audio[:, indices]
        frames = audio.unfold(1, 400, 160)
        spectrum = torch.fft.rfft(frames * self.window.to(audio.device), n=512).abs()
        mel = spectrum @ self.mel_filters.to(audio.device)
        log_mel = torch.log(mel.clamp_min(1e-12) + 0.001)
        # [batch, contexts, frequency, time] -> NCHW.
        contexts = log_mel.unfold(1, 195, 25).transpose(-1, -2)
        return contexts.unsqueeze(2).float()


class Trillsson3(nn.Module):
    """Official TRILLsson3 encoder, including its learned 1024-D projection."""

    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=False, in_chans=1, num_classes=0
        )
        self.projection = nn.Linear(1280, 1024)
        self.frontend = TrillssonFrontend()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        device = self.projection.weight.device
        # MPS does not implement float64 FFT; preserve frontend precision on CPU.
        frontend_audio = audio.cpu() if audio.device.type == "mps" else audio
        contexts = self.frontend(frontend_audio)
        batch, count = contexts.shape[:2]
        contexts = contexts.reshape(-1, 1, 195, 80).to(device)
        encoded = self.backbone(contexts / 128.0 - 1.0)
        return self.projection(encoded.reshape(batch, count, 1280).mean(1))
