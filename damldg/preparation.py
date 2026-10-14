"""Local manifest checks and atomic, non-overwriting JSON publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and fails if the destination already exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_records(records, classes, splits, root=None, decode=None, verify_hashes=False):
    if not records or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Expected nonempty records and distinct classes")
    ids, paths = set(), set()
    output = []
    for record in records:
        row = dict(record)
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Missing or duplicate example ID")
        ids.add(row["id"])
        label = row.get("label")
        if type(label) is not int or not 0 <= label < len(classes):
            raise ValueError("Invalid label index")
        if "class" in row and row["class"] != classes[label]:
            raise ValueError("Class name and label index disagree")
        if splits and row.get("split") not in splits:
            raise ValueError("Invalid split")
        original = Path(row["path"])
        path = (Path(root) / original).resolve() if root is not None else original.resolve()
        if root is not None and (original.is_absolute() or not path.is_relative_to(Path(root).resolve())):
            raise ValueError("Audio path escapes corpus root")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in paths:
            raise ValueError("Duplicate file or split leakage")
        paths.add(path)
        if decode is not None:
            decode(path)
            current = checksum(path)
            if verify_hashes and row.get("sha256") and row["sha256"] != current:
                raise ValueError(f"File checksum changed: {path}")
            row["sha256"] = current
            row["bytes"] = path.stat().st_size
        output.append(row)
    if splits and not set(splits).issubset({r["split"] for r in output}):
        raise ValueError("Every required split must be nonempty")
    return output
