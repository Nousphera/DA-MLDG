"""Prepared-corpus adapters with explicit source/target boundaries."""

from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tensorflow as tf

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


def read_audio_manifest(root, scenario):
    sources, target = SCENARIOS[scenario]
    return index_folders(root, sources, target, EMOTIONS, (".wav", ".flac"))


def index_folders(root, sources, target, classes, extensions):
    root = Path(root).resolve()
    rows = []
    seen = set()
    if target in sources:
        raise ValueError("Target must be excluded from source domains")
    for domain in sources + [target]:
        domain_rows = []
        for label, emotion in enumerate(classes):
            for path in sorted((root / domain / emotion).rglob("*")):
                if path.suffix.lower() not in extensions:
                    continue
                resolved = path.resolve()
                if not path.is_file() or not resolved.is_relative_to(root):
                    raise ValueError("Dataset files must stay inside the corpus root")
                if resolved in seen:
                    raise ValueError("Duplicate source/target path")
                seen.add(resolved)
                domain_rows.append(
                    {
                        "id": str(path.relative_to(root)),
                        "path": str(resolved),
                        "domain": domain,
                        "label": label,
                        "split": "target" if domain == target else "source",
                    }
                )
        if not domain_rows:
            raise ValueError(f"No recognized input files in {root/domain}")
        rows.extend(domain_rows)
    return rows


def read_wave(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    return wave.astype(np.float32)


def features(wave, training, rng, augment=True, spectrogram=False):
    wave = np.asarray(wave, dtype=np.float32)
    if training and augment:
        import librosa

        wave = librosa.effects.time_stretch(wave, rate=float(rng.uniform(0.8, 1.25)))
        # Zero-filled time shift, not a cyclic wrap; factor is relative to 2 seconds.
        shift = int(rng.uniform(-0.1, 0.1) * 32000)
        if shift >= 0:
            wave = np.r_[np.zeros(shift, np.float32), wave][: len(wave)]
        else:
            wave = np.r_[wave[-shift:], np.zeros(-shift, np.float32)][: len(wave)]
    if len(wave) < 32000:
        extra = 32000 - len(wave)
        wave = np.pad(wave, (extra // 2, extra - extra // 2))
    if training:
        start = int(rng.integers(len(wave) - 32000 + 1))
        wave = wave[start : start + 32000]
    spectrum = tf.abs(
        tf.signal.stft(
            wave, 400, 160, fft_length=1024, window_fn=tf.signal.hamming_window
        )
    )
    if spectrogram:
        return tf.math.log(spectrum + 1e-6)[..., None].numpy()
    mel = tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 60.0, 7800.0)
    return tf.math.log(tf.tensordot(spectrum, mel, 1) + 1e-6)[..., None].numpy()


def image_features(path, training, rng):
    from PIL import Image

    with Image.open(path) as image:
        x = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    if training:
        a, b = rng.integers(33, size=2)
        x = x[a : a + 224, b : b + 224]
        if rng.random() < 0.5:
            x = x[:, ::-1]
    else:
        x = x[16:240, 16:240]
    return x - np.array([123.68, 116.779, 103.939], np.float32)


def load_features(row, representation, training, rng, augment=True):
    if "_features" in row:
        return row["_features"]
    if representation == "image":
        return image_features(row["path"], training, rng)
    if representation == "features":
        x = np.load(row["path"], allow_pickle=False)
        if x.ndim == 2:
            x = x[..., None]
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError("Feature file must be a finite [H,W,C] array")
        return x.astype(np.float32)
    return features(
        read_wave(row["path"]),
        training,
        rng,
        augment,
        spectrogram=representation == "spectrogram",
    )


def make_batches(records, batch_size, rng, steps=None):
    """Domain-balanced batches, cycling smaller domains; no target access."""
    if any(r["split"] != "source" for r in records):
        raise ValueError("Training received held-out target data")
    domains = sorted({r["domain"] for r in records})
    if not domains or batch_size < len(domains):
        raise ValueError("Batch size must cover every source domain")
    groups = [[r for r in records if r["domain"] == d] for d in domains]
    sizes = [
        batch_size // len(domains) + (i < batch_size % len(domains))
        for i in range(len(domains))
    ]
    steps = steps or max(math.ceil(len(g) / n) for g, n in zip(groups, sizes))
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
                batch.append(group[orders[i][positions[i]]])
                positions[i] += 1
        batches.append(batch)
    return batches
