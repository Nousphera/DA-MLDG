import json
import zipfile
import numpy as np
import pytest
import soundfile as sf
from damldg.preparation import (
    parse_corpus,
    prepare_scenario,
    prepare_pacs,
    prepare_voxceleb,
    safe_extract,
    write_json,
)
from PIL import Image
from damldg.data import EMOTIONS, SCENARIOS, load_manifest


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(800), 16000)
    return path


@pytest.mark.parametrize(
    "dataset,name",
    [
        ("crema_d", "AudioWAV/1001_DFA_ANG_XX.wav"),
        ("savee", "DC/a01.wav"),
        ("ravdess", "Actor_01/03-01-05-01-01-01-01.wav"),
        ("tess", "OAF_angry/OAF_back_angry.wav"),
        ("emodb", "wav/03a01Wa.wav"),
        ("cafe", "Colère/Faible/01-C-1-1.wav"),
        ("emovo", "f1/rab-f1-b1.wav"),
        ("shemo", "female/F03A02.wav"),
        ("aesdd", "anger/a01 (1).wav"),
    ],
)
def test_native_provider_names(dataset, name, tmp_path):
    wav(tmp_path / name)
    rows = parse_corpus(dataset, tmp_path)
    assert len(rows) == 1
    assert rows[0]["label"] == EMOTIONS.index("anger")
    assert rows[0]["domain"] == dataset


def test_iemocap_all_sessions_and_consensus(tmp_path):
    for session in range(1, 6):
        uid = f"Ses0{session}F_script01_1_F000"
        wav(
            tmp_path
            / f"Session{session}/sentences/wav/{uid.rsplit('_', 1)[0]}/{uid}.wav"
        )
        annotation = tmp_path / f"Session{session}/dialog/EmoEvaluation/a.txt"
        annotation.parent.mkdir(parents=True)
        annotation.write_text(f"[0.0 - 1.0]\t{uid}\tang\t[1.0, 2.0, 3.0]\n")
    rows = parse_corpus("iemocap", tmp_path)
    assert len(rows) == 5
    assert {r["speaker"] for r in rows} == {f"Ses0{s}F" for s in range(1, 6)}
    annotation.write_text("[0.0 - 1.0]\tbad\tang\t[1.0, 2.0, 3.0]\n")
    with pytest.raises(ValueError, match="utterance|session"):
        parse_corpus("iemocap", tmp_path)


def test_reject_unknown_labels_and_escaped_audio(tmp_path):
    wav(tmp_path / "1001_DFA_BOG_XX.wav")
    with pytest.raises(ValueError, match="CREMA|crema|filename"):
        parse_corpus("crema_d", tmp_path)
    outside = wav(tmp_path / "outside.wav")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "1001_DFA_ANG_XX.wav").symlink_to(outside)
    with pytest.raises(ValueError, match="escape|outside"):
        parse_corpus("crema_d", root)


def test_scenario7_full_tree_no_disgust_in_shemo(tmp_path, monkeypatch):
    from damldg.preparation import RELEASE_COUNTS

    raw = tmp_path / "raw"
    for code in ["angry", "happy", "sad", "disgust", "fear"]:
        wav(raw / f"tess/OAF_back_{code}.wav")
    for code in ["W", "F", "T", "E", "A"]:
        wav(raw / f"emodb/03a01{code}a.wav")
    for code in ["A", "H", "S", "F"]:
        wav(raw / f"shemo/F03{code}02.wav")
    for name, count in [("tess", 5), ("emodb", 5), ("shemo", 4)]:
        monkeypatch.setitem(RELEASE_COUNTS, name, count)
    path = prepare_scenario("scn7", raw, tmp_path / "data", download=False)
    value = load_manifest(path)
    assert len(value["records"]) == 14
    assert value["sources"] == ["tess", "emodb"]
    assert value["target"] == "shemo"
    assert {r["split"] for r in value["records"] if r["domain"] == "shemo"} == {
        "target"
    }
    assert value["counts"]["shemo"]["disgust"] == 0
    assert value["inventory"]["tess"]["original_recordings"] == 5
    assert len(SCENARIOS) == 7
    value["inventory"]["tess"]["original_recordings"] = 4
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inventory"):
        load_manifest(path)


def test_partial_manual_corpora_never_publish_scenario(tmp_path):
    raw = tmp_path / "raw"
    wav(raw / "tess/OAF_back_angry.wav")
    wav(raw / "emodb/03a01Wa.wav")
    wav(raw / "shemo/F03A02.wav")
    with pytest.raises(ValueError, match="Incomplete tess release"):
        prepare_scenario("scn7", raw, tmp_path / "data", download=False)
    assert not (tmp_path / "data/scn7.json").exists()


def test_pacs_official_split_and_label_agreement(tmp_path, monkeypatch):
    from damldg.preparation import PACS_COUNTS
    from damldg.data import IMAGE_CLASSES

    raw = tmp_path / "raw"
    for domain in ["art_painting", "cartoon", "photo", "sketch"]:
        monkeypatch.setitem(PACS_COUNTS, domain, 14)
        lines = []
        for label, name in enumerate(IMAGE_CLASSES, 1):
            image = raw / "kfold" / domain / name / "a.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image)
            Image.new("RGB", (8, 8), "red").save(image.with_name("b.jpg"))
            lines.append(f"{domain}/{name}/a.jpg {label}\n")
        splits = raw / "splits"
        splits.mkdir(exist_ok=True)
        (splits / f"{domain}_train_kfold.txt").write_text("".join(lines))
        (splits / f"{domain}_crossval_kfold.txt").write_text("")
        (splits / f"{domain}_test_kfold.txt").write_text("".join(lines))
    path = prepare_pacs(raw, tmp_path / "data", "sketch")
    value = load_manifest(path)
    assert len(value["records"]) == 28
    assert value["inventory"]["photo"]["original_recordings"] == 14
    assert value["inventory"]["photo"]["selected_recordings"] == 7
    assert {r["domain"] for r in value["records"] if r["split"] == "target"} == {
        "sketch"
    }
    assert all(len(row["sha256"]) == 64 for row in value["records"])
    (splits / "photo_train_kfold.txt").write_text("photo/dog/a.jpg 2\n")
    with pytest.raises(ValueError, match="label|class"):
        prepare_pacs(raw, tmp_path / "other", "sketch")
    Image.new("RGB", (8, 8), "yellow").save(value["records"][0]["path"])
    with pytest.raises(ValueError, match="checksum changed"):
        load_manifest(path)


def test_voxceleb_speaker_labels(tmp_path, monkeypatch):
    from damldg.preparation import VOXCELEB_COUNTS

    monkeypatch.setitem(VOXCELEB_COUNTS, "speakers", 2)
    monkeypatch.setitem(VOXCELEB_COUNTS, "recordings", 2)
    wav(tmp_path / "raw/wav/id10001/video/00001.wav")
    wav(tmp_path / "raw/wav/id10002/video/00001.wav")
    path = prepare_voxceleb(tmp_path / "raw", tmp_path / "data")
    value = load_manifest(path)
    assert value["classes"] == ["id10001", "id10002"]
    assert value["representation"] == "spectrogram"
    assert {r["label"] for r in value["records"]} == {0, 1}


def test_safe_archive_and_atomic_manifest(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.wav", "bad")
    with pytest.raises(ValueError, match="archive|escape"):
        safe_extract(archive, tmp_path / "output")
    out = tmp_path / "manifest.json"
    write_json(out, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(out, {"a": 2})
    assert json.loads(out.read_text()) == {"a": 1}


def test_every_table1_source_target_scenario():
    assert SCENARIOS == {
        "scn1": (["crema_d", "savee", "ravdess"], "iemocap"),
        "scn2": (["crema_d", "savee", "tess"], "ravdess"),
        "scn3": (["crema_d", "savee", "ravdess"], "tess"),
        "scn4": (["savee", "ravdess", "emodb"], "cafe"),
        "scn5": (["emovo", "shemo", "emodb"], "aesdd"),
        "scn6": (["crema_d", "savee", "ravdess"], "shemo"),
        "scn7": (["tess", "emodb"], "shemo"),
    }


def test_download_resumes_range_and_validates_hash(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import hashlib
    import threading
    from damldg.preparation import _download

    payload = b"actual download bytes with partial prefix"
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ranges.append(self.headers.get("Range"))
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].rstrip("-"))
            self.send_response(206 if start else 200)
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "file.zip"
    destination.with_name("file.zip.part").write_bytes(payload[:7])
    try:
        _download(
            f"http://127.0.0.1:{server.server_port}/file.zip",
            destination,
            expected="sha256:" + hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert destination.read_bytes() == payload
    assert ranges == ["bytes=7-"]
    assert not destination.with_name("file.zip.part").exists()


def test_crema_uses_commit_not_tree_sha_for_lfs(tmp_path, monkeypatch):
    import io
    import damldg.preparation as preparation

    commit, tree = "commit-identity", "different-tree-identity"
    payload = {
        "sha": tree,
        "tree": [{"path": f"AudioWAV/{i:04d}_DFA_ANG_XX.wav"} for i in range(7442)],
    }

    def urlopen(url, **kwargs):
        value = {"object": {"sha": commit}} if "/git/ref/" in url else payload
        return io.BytesIO(json.dumps(value).encode())

    downloads = []

    def download(url, *args, **kwargs):
        downloads.append(url)
        raise OSError("stop before real download")

    monkeypatch.setattr(preparation.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(preparation, "_download", download)
    with pytest.raises(OSError, match="stop before"):
        preparation.download_corpus("crema_d", tmp_path)
    assert "/commit-identity/AudioWAV/" in downloads[0]
    assert tree not in downloads[0]


def test_complete_counts_still_require_selected_class_coverage(tmp_path, monkeypatch):
    from damldg.preparation import RELEASE_COUNTS, verify_corpus_inventory

    wav(tmp_path / "03a01Wa.wav")
    monkeypatch.setitem(RELEASE_COUNTS, "emodb", 1)
    with pytest.raises(ValueError, match="emotion coverage"):
        verify_corpus_inventory("emodb", tmp_path, parse_corpus("emodb", tmp_path))


def test_tess_known_provider_neutral_filename_typo_is_explicit(tmp_path):
    wav(tmp_path / "OAF_back_angry.wav")
    wav(tmp_path / "OA_bite_neutral.wav")
    assert len(parse_corpus("tess", tmp_path)) == 1


def test_invalid_image_never_publishes_manifest(tmp_path):
    from damldg.preparation import _publish

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a JPEG")
    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), "blue").save(good)
    value = {
        "representation": "image",
        "classes": ["a", "b"],
        "records": [
            {"id": "a", "path": str(bad), "domain": "a", "label": 0, "split": "source"},
            {
                "id": "b",
                "path": str(good),
                "domain": "b",
                "label": 1,
                "split": "target",
            },
        ],
    }
    destination = tmp_path / "manifest.json"
    with pytest.raises(OSError):
        _publish(destination, value)
    assert not destination.exists()
