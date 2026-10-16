import json
import os
import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from damldg.data import load_manifest, read_wave, load_features, make_batches


def test_real_wave_resampling_crop_variable_eval(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.arange(48000 * 3) / 48000
    wave = np.sin(2 * np.pi * 440 * t).astype("float32")
    sf.write(path, np.column_stack([wave, wave]), 48000)
    assert read_wave(path).shape == (48000,)
    rng = np.random.default_rng(2)
    train = load_features({"path": path}, "audio", True, rng, augment=False)
    test = load_features({"path": path}, "audio", False, rng)
    spectrum = load_features({"path": path}, "spectrogram", True, rng, augment=False)
    assert train.shape == (1, 64, 198)
    assert test.shape == (1, 64, 298)
    assert spectrum.shape == (1, 513, 198)
    assert train.dtype == np.float32 and np.isfinite(train).all()


def test_image_imagenet_normalization_and_features_chw(tmp_path):
    path = tmp_path / "white.jpg"
    Image.new("RGB", (300, 250), "white").save(path)
    x = load_features({"path": path}, "image", False, np.random.default_rng(2))
    assert x.shape == (3, 224, 224)
    np.testing.assert_allclose(
        x[:, 0, 0],
        (1 - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225],
        rtol=1e-5,
    )
    np.save(tmp_path / "x.npy", np.ones((1, 8, 12), dtype="float32"))
    assert load_features(
        {"path": tmp_path / "x.npy"}, "features", False, None
    ).shape == (1, 8, 12)


def test_manifest_domain_file_leakage(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.ones((1, 8, 8)))
    second = tmp_path / "y.npy"
    os.link(path, second)
    rows = [
        {"id": "a", "path": str(path), "domain": "a", "label": 0, "split": "source"},
        {"id": "b", "path": str(second), "domain": "b", "label": 1, "split": "target"},
    ]
    manifest = tmp_path / "m.json"
    value = {"classes": ["a", "b"], "representation": "features", "records": rows}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate|leakage"):
        load_manifest(manifest)
    second.unlink()
    np.save(second, np.ones((1, 8, 8)))
    rows[1]["domain"] = "a"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="domain|leakage"):
        load_manifest(manifest)


def test_balanced_training_boundaries():
    rows = [
        {"id": str(i), "domain": d, "split": "source"}
        for i, d in enumerate(["a"] * 5 + ["b"])
    ]
    batches = make_batches(rows, 4, np.random.default_rng(2))
    assert all([r["domain"] for r in b].count("a") == 2 for b in batches)
    with pytest.raises(ValueError, match="target|held-out"):
        make_batches(
            rows + [{"domain": "c", "split": "target"}], 4, np.random.default_rng(1)
        )
    with pytest.raises(ValueError, match="steps"):
        make_batches(rows, 4, np.random.default_rng(2), steps=0)


def test_manifest_rejects_changed_recorded_hash(tmp_path):
    from damldg.preparation import checksum

    path, other = tmp_path / "x.npy", tmp_path / "y.npy"
    np.save(path, np.ones((1, 8, 8)))
    np.save(other, np.zeros((1, 8, 8)))
    value = {
        "representation": "features",
        "classes": ["a", "b"],
        "records": [
            {
                "id": "a",
                "path": str(path),
                "domain": "a",
                "label": 0,
                "split": "source",
                "sha256": checksum(path),
            },
            {
                "id": "b",
                "path": str(other),
                "domain": "b",
                "label": 1,
                "split": "target",
                "sha256": checksum(other),
            },
        ],
    }
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(value))
    load_manifest(manifest)
    np.save(path, np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match="checksum changed"):
        load_manifest(manifest)
