"""Pretrained encoders and verified TRILLsson3 feature preparation."""

from pathlib import Path
import shutil
from urllib.request import urlopen
import numpy as np
import torch
from .training import digest, write_json

TRILLSSON_URL = (
    "https://github.com/Nousphera/FedLN/releases/download/v0.2.0/trillsson3.pt"
)
TRILLSSON_SHA256 = "256fc6a004f1ff285b86941c7b0f519db1991c0856acd13dd01b15ef91d9cdda"


def download_verified(url, path, sha256):
    path = Path(path)
    if path.is_file():
        if digest(path) != sha256:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if digest(temporary) != sha256:
            raise ValueError("Downloaded checkpoint checksum mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_imagenet(root):
    from torchvision.models import ResNet18_Weights, resnet18
    from .model import build_model, load_encoder, save_encoder

    path = Path(root) / "imagenet-resnet18.pt"
    model = build_model(in_channels=3, num_classes=7)
    if path.exists():
        load_encoder(model, path)
        return path
    source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.encoder.load_state_dict(
        {k: v for k, v in source.state_dict().items() if not k.startswith("fc.")},
        strict=True,
    )
    save_encoder(model, path)
    write_json(
        path.with_suffix(".json"),
        {"source": ResNet18_Weights.IMAGENET1K_V1.url, "sha256": digest(path)},
    )
    return path


def load_trillsson(root, device="cpu"):
    from .trillsson import Trillsson3

    path = download_verified(
        TRILLSSON_URL, Path(root) / "trillsson3.pt", TRILLSSON_SHA256
    )
    model = Trillsson3()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval().requires_grad_(False).to(device)


def embedding_image(embedding):
    """Keep all 1024 dimensions in a deterministic 32×32 single-channel grid."""
    if embedding.shape != (1024,) or not torch.isfinite(embedding).all():
        raise ValueError("Expected a finite 1024-dimensional TRILLsson3 embedding")
    return embedding.detach().float().cpu().numpy().reshape(1, 32, 32)


def prepare_embeddings(
    manifest_path, model_root="checkpoints", feature_root="features", device="cpu"
):
    from .data import load_manifest, read_wave

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if manifest["representation"] != "audio":
        raise ValueError("TRILLsson preparation requires audio")
    model = load_trillsson(model_root, device)
    root = Path(feature_root).resolve() / manifest_path.stem
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in manifest["records"]:
        source_hash = digest(row["path"])
        key = (
            __import__("hashlib")
            .sha256((row["id"] + source_hash + TRILLSSON_SHA256).encode())
            .hexdigest()
        )
        path = root / f"{key}.npy"
        if path.exists():
            data = np.load(path, allow_pickle=False)
            if data.shape != (1, 32, 32) or not np.isfinite(data).all():
                raise ValueError(f"Invalid cached embedding: {path}")
        else:
            wave = torch.from_numpy(read_wave(row["path"])).to(device)
            with torch.inference_mode():
                data = embedding_image(model(wave)[0])
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as stream:
                np.save(stream, data, allow_pickle=False)
            temporary.replace(path)
        records.append(
            {
                **row,
                "path": str(path),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "source_path": row["path"],
                "source_sha256": source_hash,
            }
        )
    output = manifest_path.with_name(manifest_path.stem + "-trillsson.json")
    result = {
        **manifest,
        "representation": "features",
        "records": records,
        "feature_extractor": {
            "name": "TRILLsson3",
            "checkpoint_sha256": TRILLSSON_SHA256,
            "backbone": "EfficientNetV2-S (official TRILLsson3 artifact)",
            "layout": "1024 dimensions reshaped to 1x32x32",
            "source_manifest_sha256": digest(manifest_path),
        },
    }
    write_json(output, result)
    return output
