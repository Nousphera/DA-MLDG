"""Validated corpus manifests and NumPy CHW inputs for PyTorch.

Audio uses the original 25 ms periodic Hamming / 10 ms STFT with a 1024
point FFT and HTK mel triangles. Training crops two seconds; evaluation
keeps the complete utterance (short utterances are zero padded).
"""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear"]
SCENARIOS = {
    "scn1": (["crema_d", "savee", "ravdess"], "iemocap"),
    "scn2": (["crema_d", "savee", "tess"], "ravdess"),
    "scn3": (["crema_d", "savee", "ravdess"], "tess"),
    "scn4": (["savee", "ravdess", "emodb"], "cafe"),
    "scn5": (["emovo", "shemo", "emodb"], "aesdd"),
    "scn6": (["crema_d", "savee", "ravdess"], "shemo"),
    "scn7": (["tess", "emodb"], "shemo"),
}
PACS = ["art_painting", "cartoon", "photo", "sketch"]
IMAGE_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load_manifest(path):
    """Read and validate paths, labels, identities and source/target separation."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return validate_manifest(value, path.parent)


def validate_manifest(value, base_path="."):
    """Validate in memory so preparation can fail before publishing a file."""
    from .preparation import check_records, RELEASE_COUNTS, PACS_COUNTS, VOXCELEB_COUNTS

    if not isinstance(value, dict):
        raise ValueError("Manifest must be an object")
    value = dict(value)
    representation = value.get("representation")
    if representation not in {"audio", "image", "spectrogram", "features"}:
        raise ValueError("Unknown representation")
    classes = value.get("classes", [])
    root = (Path(base_path) / value.get("root", ".")).resolve()
    rows = check_records(
        value.get("records", []),
        classes,
        None,
        root=root,
        decode=verify_image if representation == "image" else None,
        verify_hashes=True,
    )
    domains = {}
    for row in rows:
        if row.get("split") not in {"source", "target"}:
            raise ValueError("Manifest split must be source or target")
        domain = row.get("domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError("Every record needs a domain")
        if domain in domains and domains[domain] != row["split"]:
            raise ValueError("Source/target domain leakage")
        domains[domain] = row["split"]
    sources = {d for d, split in domains.items() if split == "source"}
    targets = {d for d, split in domains.items() if split == "target"}
    if not sources:
        raise ValueError("Manifest has no source domain")
    if not targets and value.get("dataset") != "voxceleb":
        raise ValueError("Manifest has no held-out target domain")
    if "sources" in value and set(value["sources"]) != sources:
        raise ValueError("Declared source domains disagree with records")
    if "target" in value and targets != {value["target"]}:
        raise ValueError("Declared target domain disagrees with records")
    scenario = value.get("scenario")
    if scenario is not None:
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        expected_sources, expected_target = SCENARIOS[scenario]
        if (
            sources != set(expected_sources)
            or targets != {expected_target}
            or classes != EMOTIONS
        ):
            raise ValueError(
                "Scenario classes or source/target domains disagree with Table I"
            )
        inventory = value.get("inventory", {})
        if set(inventory) != sources | targets:
            raise ValueError("Scenario requires a complete original corpus inventory")
        for domain in sources | targets:
            entry = inventory[domain]
            selected = [r for r in rows if r["domain"] == domain]
            required = set(range(len(EMOTIONS))) - (
                {EMOTIONS.index("disgust")} if domain == "shemo" else set()
            )
            if (
                entry.get("original_recordings") != RELEASE_COUNTS[domain]
                or entry.get("expected_original_recordings") != RELEASE_COUNTS[domain]
                or entry.get("selected_recordings") != len(selected)
                or {r["label"] for r in selected} != required
            ):
                raise ValueError(
                    f"Incomplete or inconsistent scenario inventory: {domain}"
                )
    if representation == "image" and value.get("dataset") == "pacs":
        if (
            classes != IMAGE_CLASSES
            or sources | targets != set(PACS)
            or len(targets) != 1
        ):
            raise ValueError("Invalid PACS classes or domains")
        inventory = value.get("inventory", {})
        for domain in PACS:
            selected = [r for r in rows if r["domain"] == domain]
            if {r["label"] for r in selected} != set(range(len(IMAGE_CLASSES))):
                raise ValueError(f"Incomplete PACS class coverage: {domain}")
            entry = inventory.get(domain)
            if entry is None:
                if len(selected) != PACS_COUNTS[domain]:
                    raise ValueError(f"Incomplete PACS domain: {domain}")
            elif entry.get("original_recordings") != PACS_COUNTS[domain] or entry.get(
                "selected_recordings"
            ) != len(selected):
                raise ValueError(f"Incomplete PACS inventory: {domain}")
    if value.get("dataset") == "voxceleb":
        if (
            len(rows) != VOXCELEB_COUNTS["recordings"]
            or len(classes) != VOXCELEB_COUNTS["speakers"]
        ):
            raise ValueError("Incomplete VoxCeleb1 manifest")
    value["records"] = rows
    return value


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(wave) or rate <= 0 or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    wave = wave.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    return np.asarray(wave, dtype=np.float32)


@lru_cache(maxsize=1)
def _mel_weights():
    # TensorFlow's original linear_to_mel_weight_matrix uses HTK mel scale
    # and zeros its DC bin. This explicit equivalent needs no TensorFlow.
    mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    edges = np.linspace(mel(60.0), mel(7800.0), 66)
    bins = mel(np.linspace(0, 8000, 513))[:, None]
    lower = (bins - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - bins) / (edges[2:] - edges[1:-1])
    weights = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
    weights[0] = 0
    return weights


def verify_image(path):
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()


def features(wave, training, rng, augment=True, spectrogram=False):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("Expected a finite nonempty mono waveform")
    if training and augment:
        import librosa

        wave = librosa.effects.time_stretch(wave, rate=float(rng.uniform(0.8, 1.25)))
        shift = int(rng.uniform(-0.1, 0.1) * 32000)
        if shift >= 0:
            wave = np.pad(wave, (shift, 0))[: len(wave)]
        else:
            wave = np.pad(wave[-shift:], (0, -shift))[: len(wave)]
    if len(wave) < 32000:
        extra = 32000 - len(wave)
        wave = np.pad(wave, (extra // 2, extra - extra // 2))
    if training:
        start = int(rng.integers(len(wave) - 32000 + 1))
        wave = wave[start : start + 32000]
    frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::160]
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 400)
    spectrum = np.abs(np.fft.rfft(frames * window, n=1024)).astype(np.float32)
    value = spectrum if spectrogram else spectrum @ _mel_weights()
    return np.ascontiguousarray(np.log(value + 1e-6).T[None], dtype=np.float32)


def image_features(path, training, rng, augment=True):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        scale = 256 / min(w, h)
        image = image.resize(
            (round(w * scale), round(h * scale)), Image.Resampling.BILINEAR
        )
        w, h = image.size
        if training and augment:
            left, top = int(rng.integers(w - 223)), int(rng.integers(h - 223))
        else:
            left, top = (w - 224) // 2, (h - 224) // 2
        x = (
            np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32)
            / 255
        )
    if training and augment and rng.random() < 0.5:
        x = x[:, ::-1]
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def load_features(row, representation, training, rng, augment=True):
    if representation == "image":
        return image_features(row["path"], training, rng, augment)
    if representation == "features":
        x = np.load(row["path"], allow_pickle=False)
        if x.ndim == 2:
            x = x[None]
        if (
            x.ndim != 3
            or not all(x.shape)
            or x.dtype.kind not in "fiu"
            or not np.isfinite(x).all()
        ):
            raise ValueError("Feature file must be a finite nonempty [C,H,W] array")
        return np.ascontiguousarray(x, dtype=np.float32)
    if representation not in {"audio", "spectrogram"}:
        raise ValueError("Unknown representation")
    return features(
        read_wave(row["path"]),
        training,
        rng,
        augment,
        spectrogram=representation == "spectrogram",
    )


def make_batches(records, batch_size, rng, steps=None):
    """Balance domains, reshuffling each domain when its complete order cycles."""
    if any(r["split"] != "source" for r in records):
        raise ValueError("Training received held-out target data")
    domains = sorted({r["domain"] for r in records})
    if type(batch_size) is not int or not domains or batch_size < len(domains):
        raise ValueError("Batch size must cover every source domain")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    groups = [[r for r in records if r["domain"] == d] for d in domains]
    sizes = [
        batch_size // len(domains) + (i < batch_size % len(domains))
        for i in range(len(domains))
    ]
    steps = (
        steps
        if steps is not None
        else max(math.ceil(len(g) / n) for g, n in zip(groups, sizes))
    )
    orders = [rng.permutation(len(g)) for g in groups]
    positions = [0] * len(groups)
    batches = []
    for _ in range(steps):
        batch = []
        for i, (group, n) in enumerate(zip(groups, sizes)):
            for _ in range(n):
                if positions[i] == len(group):
                    orders[i] = rng.permutation(len(group))
                    positions[i] = 0
                batch.append(group[int(orders[i][positions[i]])])
                positions[i] += 1
        rng.shuffle(batch)
        batches.append(batch)
    return batches
