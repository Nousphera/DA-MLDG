"""Import licensed corpora using explicit provider-derived emotion annotations."""
import csv
from pathlib import Path
import shutil
import tempfile
from .data import EMOTIONS, SCENARIOS, read_wave
from .preparation import checksum, write_json

CORPORA = sorted({d for sources, target in SCENARIOS.values() for d in sources + [target]})


def import_corpus(root, annotations, corpus, output):
    """CSV path,emotion; preserve relative speaker/session paths and source bytes."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if corpus not in CORPORA:
        raise ValueError("Unknown corpus")
    if output.exists():
        raise FileExistsError(output)
    entries, seen = [], set()
    with Path(annotations).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "emotion"}.issubset(reader.fieldnames or []):
            raise ValueError("Annotations require path,emotion columns")
        for entry in reader:
            relative = Path(entry["path"])
            source = (root / relative).resolve()
            emotion = entry["emotion"].strip()
            if emotion not in EMOTIONS:
                raise ValueError(f"Use an explicit shared emotion, received {emotion!r}")
            if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(root):
                raise ValueError("Annotation path escapes corpus root")
            if source in seen:
                raise ValueError("Duplicate annotation path")
            if source.suffix.lower() not in {".wav", ".flac"}:
                raise ValueError("Only WAV/FLAC inputs are supported")
            seen.add(source)
            read_wave(source)
            entries.append({"path": relative.as_posix(), "emotion": emotion, "sha256": checksum(source)})
    if not entries:
        raise ValueError("No annotated audio")
    entries.sort(key=lambda row: row["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-corpus-", dir=output.parent) as temporary:
        stage = Path(temporary) / "corpus"
        stage.mkdir()
        for entry in entries:
            destination = stage / entry["emotion"] / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / entry["path"], destination)
        write_json(stage / "preparation.json", {"schema": 1, "corpus": corpus, "source_root": str(root),
                   "annotations_sha256": checksum(annotations), "records": entries,
                   "label_policy": "explicit provider-derived CSV; no inferred emotion labels"})
        stage.rename(output)
    return entries
