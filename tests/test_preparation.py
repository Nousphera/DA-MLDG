import json
import numpy as np
import pytest
import soundfile as sf
from damldg.import_data import import_corpus
from damldg.data import read_audio_manifest
from damldg.preparation import check_records, write_json
from damldg.train import Config, run


def test_explicit_import_preserves_labels_hashes_and_trains(tmp_path):
    rng = np.random.default_rng(3)
    for corpus in ("tess", "emodb", "shemo"):
        raw = tmp_path / "raw" / corpus
        raw.mkdir(parents=True)
        for label in ("anger", "happiness"):
            sf.write(raw / f"{label}.wav", rng.normal(0, .1, 16000), 16000)
        csv = raw / "annotations.csv"
        csv.write_text("path,emotion\nanger.wav,anger\nhappiness.wav,happiness\n")
        rows = import_corpus(raw, csv, corpus, tmp_path / "normalized" / corpus)
        assert all(len(r["sha256"]) == 64 for r in rows)
        with pytest.raises(FileExistsError):
            import_corpus(raw, csv, corpus, tmp_path / "normalized" / corpus)
    rows = read_audio_manifest(tmp_path / "normalized", "scn7")
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"classes": ["anger", "happiness", "sadness", "disgust", "fear"],
                         "records": rows, "representation": "audio"})
    result = run(Config(dataset="manifest", manifest=str(manifest), epochs=1, batch_size=4,
                        steps_per_epoch=1, augment=False, learning_rate=.001, output=str(tmp_path / "run")))
    assert result["target"]["num_examples"] == 2 and np.isfinite(result["target"]["loss"])


def test_import_rejects_invalid_labels_escape_and_duplicate_annotations(tmp_path):
    for content in ("x.wav,neutral", "../x.wav,anger", "x.wav,anger\nx.wav,anger"):
        sf.write(tmp_path / "x.wav", np.zeros(100), 16000)
        csv = tmp_path / "a.csv"
        csv.write_text("path,emotion\n" + content + "\n")
        with pytest.raises(ValueError):
            import_corpus(tmp_path, csv, "tess", tmp_path / "out")
        assert not (tmp_path / "out").exists()


def test_index_rejects_symlink_outside_root(tmp_path):
    outside = tmp_path / "outside.wav"
    sf.write(outside, np.zeros(100), 16000)
    root = tmp_path / "root"
    (root / "tess/anger").mkdir(parents=True)
    (root / "tess/anger/a.wav").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        read_audio_manifest(root, "scn7")


def test_config_paths_stay_with_checkout_when_working_directory_changes(tmp_path, monkeypatch):
    from damldg.train import load_config
    root = tmp_path / "checkout with spaces"
    (root / "configs").mkdir(parents=True)
    config = root / "configs/check.json"
    config.write_text(json.dumps({"manifest": "data/input.json", "output": "runs/check"}))
    monkeypatch.chdir(tmp_path)
    loaded = load_config(config)
    assert loaded.manifest == str(root / "data/input.json")
    assert loaded.output == str(root / "runs/check")


def test_manifest_rejects_same_inode_under_different_names(tmp_path):
    import os
    from damldg.preparation import check_records
    first = tmp_path / "first.wav"
    first.write_bytes(b"fixture")
    second = tmp_path / "second.wav"
    os.link(first, second)
    rows = [{"id": "a", "path": str(first), "label": 0, "split": "train"},
            {"id": "b", "path": str(second), "label": 1, "split": "test"}]
    with pytest.raises(ValueError, match="Duplicate|leakage"):
        check_records(rows, ["a", "b"], ["train", "test"])
