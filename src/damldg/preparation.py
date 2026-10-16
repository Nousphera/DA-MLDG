"""Native-provider corpus preparation, resumable downloads and atomic manifests.

No corpus is replaced with synthetic data. Every recognized audio file is indexed;
only the paper's five emotions are selected. Original waveforms remain untouched.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

from .data import EMOTIONS, SCENARIOS, PACS, IMAGE_CLASSES, read_wave

PROVIDERS = {
    "crema_d": (
        "https://github.com/CheyneyComputerScience/CREMA-D",
        "AudioWAV/1001_DFA_ANG_XX.wav; use git-lfs for real audio, not pointer files",
    ),
    "savee": (
        "http://kahlan.eps.surrey.ac.uk/savee/",
        "DC/a01.wav, JE/a01.wav, JK/a01.wav, KL/a01.wav (AudioData archive)",
    ),
    "ravdess": (
        "https://zenodo.org/records/1188976",
        "Actor_01/03-01-05-01-01-01-01.wav (Audio_Speech_Actors_01-24.zip)",
    ),
    "tess": (
        "https://doi.org/10.5683/SP2/E8H2MF",
        "OAF_angry/OAF_back_angry.wav and YAF_*/*.wav, or flat original WAV files",
    ),
    "iemocap": (
        "https://sail.usc.edu/iemocap/",
        "Session1 through Session5, each with dialog/EmoEvaluation/*.txt and sentences/wav/*/*.wav",
    ),
    "emodb": (
        "http://emodb.bilderbar.info/download/",
        "wav/03a01Wa.wav (download.zip)",
    ),
    "cafe": (
        "https://zenodo.org/records/1478765",
        "CaFE_48k/Colère/Faible/01-C-1-1.wav; use one resolution only",
    ),
    "emovo": (
        "http://voice.fub.it/activities/corpora/emovo/",
        "f1/rab-f1-b1.wav through f3,m1,m2,m3; original EMOVO archive",
    ),
    "shemo": (
        "https://github.com/omidmnezami/Persian-Emotional-Speech-Database-ShEMO",
        "female/F03S02.wav and male/M01A01.wav (both original archives)",
    ),
    "aesdd": (
        "https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/",
        "AESDD/anger/a01 (1).wav; disgust/, fear/, happiness/, sadness/",
    ),
    "pacs": (
        "https://domaingeneralization.github.io/",
        "kfold/DOMAIN/CLASS/*.jpg with splits/DOMAIN_{train,crossval,test}_kfold.txt; or full flwrlabs/pacs parquet cache",
    ),
    "voxceleb": (
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html",
        "wav/id10001/VIDEO/00001.wav; combine original vox1_dev_wav and vox1_test_wav archives",
    ),
}
CORPORA = tuple(name for name in PROVIDERS if name not in {"pacs", "voxceleb"})
# Complete native releases, before five-emotion selection. IEMOCAP counts
# sentence WAVs only; RAVDESS counts audio-only speech only.
RELEASE_COUNTS = {
    "crema_d": 7442,
    "savee": 480,
    "ravdess": 1440,
    "tess": 2800,
    "iemocap": 10039,
    "emodb": 535,
    "cafe": 936,
    "emovo": 588,
    "shemo": 3000,
    "aesdd": 604,
}
PACS_COUNTS = {"art_painting": 2048, "cartoon": 2344, "photo": 1670, "sketch": 3929}
VOXCELEB_COUNTS = {"recordings": 153516, "speakers": 1251}
ALIASES = {
    "cremad": "crema_d",
    "crema-d": "crema_d",
    "emo-db": "emodb",
    "voxceleb1": "voxceleb",
}


def canonical_name(name):
    value = name.lower()
    return ALIASES.get(value, value)


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Publish atomically without replacing another manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_records(
    records, classes, splits=None, root=None, decode=None, verify_hashes=False
):
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or any(not isinstance(c, str) or not c for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Expected at least two distinct class names")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty records")
    ids, paths, file_ids, output = set(), set(), set(), []
    for record in records:
        row = dict(record)
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Missing or duplicate example ID")
        ids.add(row["id"])
        if type(row.get("label")) is not int or not 0 <= row["label"] < len(classes):
            raise ValueError("Invalid label index")
        if "class" in row and row["class"] != classes[row["label"]]:
            raise ValueError("Class name and label index disagree")
        if splits and row.get("split") not in splits:
            raise ValueError("Invalid split")
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("Record path must be a nonempty string")
        original = Path(row["path"]).expanduser()
        path = ((Path(root) / original) if root is not None else original).resolve()
        if (
            root is not None
            and not original.is_absolute()
            and (
                not path.is_relative_to(Path(root).resolve()) or ".." in original.parts
            )
        ):
            raise ValueError("Record path escapes corpus root")
        if not path.is_file():
            raise FileNotFoundError(path)
        identity = (path.stat().st_dev, path.stat().st_ino)
        if path in paths or identity in file_ids:
            raise ValueError("Duplicate file or source/target split leakage")
        paths.add(path)
        file_ids.add(identity)
        row["path"] = str(path)
        if decode is not None:
            decode(path)
        if decode is not None or verify_hashes:
            current = checksum(path)
            if verify_hashes and row.get("sha256") and row["sha256"] != current:
                raise ValueError(f"File checksum changed: {path}")
            row["sha256"] = current
            row["bytes"] = path.stat().st_size
        output.append(row)
    if splits and not set(splits).issubset({r["split"] for r in output}):
        raise ValueError("Every required split must be nonempty")
    return output


def _inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes corpus root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return resolved


def _audio_paths(root):
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}
        and not any(
            s.startswith(".") or s == "__MACOSX" for s in p.relative_to(root).parts
        )
    )
    for path in paths:
        _inside(root, path)
    return paths


def _row(root, path, corpus, emotion, speaker):
    return {
        "id": f"{corpus}/{path.relative_to(root).as_posix()}",
        "path": str(_inside(root, path)),
        "domain": corpus,
        "label": EMOTIONS.index(emotion),
        "class": emotion,
        "speaker": speaker,
    }


def _match(pattern, value, corpus, path):
    match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unrecognized {corpus} filename: {path}")
    return match


def parse_corpus(corpus, root):
    """Parse provider annotations or filenames, with explicit known exclusions.

    IEMOCAP uses all five sessions, scripted and improvised, the consensus labels
    ang/hap/sad/dis/fea, and excludes excited rather than silently merging it.
    ShEMO has no disgust class; it remains an explicit zero count.
    """
    corpus = canonical_name(corpus)
    if corpus not in CORPORA:
        raise ValueError(f"Unknown speech corpus: {corpus}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(_missing(corpus, root))
    if corpus == "iemocap":
        return _iemocap(root)
    rows = []
    for path in _audio_paths(root):
        stem = path.stem
        if corpus == "crema_d":
            m = _match(
                r"(\d{4})_([A-Z]{3})_(ANG|HAP|SAD|DIS|FEA|NEU)_(XX|LO|MD|HI)",
                stem,
                corpus,
                path,
            )
            speaker, code = m[1], m[3].upper()
            emotion = {
                "ANG": "anger",
                "HAP": "happiness",
                "SAD": "sadness",
                "DIS": "disgust",
                "FEA": "fear",
                "NEU": None,
            }[code]
        elif corpus == "savee":
            name = (
                stem
                if re.match(r"^(DC|JE|JK|KL)", stem, re.I)
                else path.parent.name + stem
            )
            m = _match(r"(DC|JE|JK|KL)_?(a|d|f|h|n|sa|su)(\d{2})", name, corpus, path)
            speaker = m[1].upper()
            emotion = {
                "a": "anger",
                "d": "disgust",
                "f": "fear",
                "h": "happiness",
                "sa": "sadness",
                "n": None,
                "su": None,
            }[m[2].lower()]
        elif corpus == "ravdess":
            m = _match(
                r"(0[123])-(0[12])-(0[1-8])-(0[12])-(0[12])-(0[12])-(\d{2})",
                stem,
                corpus,
                path,
            )
            if not 1 <= int(m[7]) <= 24:
                raise ValueError(f"Invalid RAVDESS actor: {path}")
            if m[1] != "03" or m[2] != "01":
                continue  # Explicit audio-only speech selection; never include song.
            speaker = m[7]
            emotion = {
                "01": None,
                "02": None,
                "03": "happiness",
                "04": "sadness",
                "05": "anger",
                "06": "fear",
                "07": "disgust",
                "08": None,
            }[m[3]]
        elif corpus == "tess":
            # One Borealis individual-file metadata entry omits F; the complete
            # ZIP uses OAF_bite_neutral.wav with the identical provider MD5.
            if stem.lower() == "oa_bite_neutral":
                stem = "OAF_bite_neutral"
            m = _match(
                r"(OAF|YAF)_(.+)_(angry|disgust|fear|happy|sad|neutral|ps)",
                stem,
                corpus,
                path,
            )
            speaker = m[1].upper()
            emotion = {
                "angry": "anger",
                "disgust": "disgust",
                "fear": "fear",
                "happy": "happiness",
                "sad": "sadness",
                "neutral": None,
                "ps": None,
            }[m[3].lower()]
        elif corpus == "emodb":
            m = _match(r"(\d{2})[ab]\d{2}([WLEAFTN])[a-z]?", stem, corpus, path)
            speaker = m[1]
            emotion = {
                "W": "anger",
                "F": "happiness",
                "T": "sadness",
                "E": "disgust",
                "A": "fear",
                "L": None,
                "N": None,
            }[m[2].upper()]
        elif corpus == "cafe":
            m = _match(r"(\d{2})-([CDJNPST])-([12])-([1-6])", stem, corpus, path)
            speaker = m[1]
            emotion = {
                "C": "anger",
                "D": "disgust",
                "J": "happiness",
                "P": "fear",
                "T": "sadness",
                "N": None,
                "S": None,
            }[m[2].upper()]
        elif corpus == "emovo":
            m = _match(
                r"(dis|gio|neu|pau|rab|sor|tri)-([fm][1-3])-([blnd]\d)",
                stem,
                corpus,
                path,
            )
            speaker = m[2].lower()
            emotion = {
                "dis": "disgust",
                "gio": "happiness",
                "pau": "fear",
                "rab": "anger",
                "tri": "sadness",
                "neu": None,
                "sor": None,
            }[m[1].lower()]
        elif corpus == "shemo":
            m = _match(r"([FM]\d{2})([SAHWFN])(\d{2,})", stem, corpus, path)
            speaker = m[1].upper()
            emotion = {
                "S": "sadness",
                "A": "anger",
                "H": "happiness",
                "F": "fear",
                "W": None,
                "N": None,
            }[m[2].upper()]
        else:  # AESDD filenames encode emotion and actor; folders cross-check it.
            m = _match(r"([adhfs])\d{2}\s*\((\d+)\)(?:\s*\d*)?", stem, corpus, path)
            speaker = m[2]
            emotion = {
                "a": "anger",
                "d": "disgust",
                "h": "happiness",
                "f": "fear",
                "s": "sadness",
            }[m[1].lower()]
            folders = {p.lower() for p in path.relative_to(root).parts[:-1]}
            selected_folders = folders.intersection(EMOTIONS)
            if selected_folders and selected_folders != {emotion}:
                raise ValueError(f"AESDD filename and emotion folder disagree: {path}")
        if emotion is not None:
            rows.append(_row(root, path, corpus, emotion, speaker))
    if not rows:
        raise ValueError(
            f"No selected emotion recordings at {root}; {_missing(corpus, root)}"
        )
    # Multiple resolutions, copies, symlink aliases and duplicate utterances are errors.
    identities = [(r["speaker"], Path(r["path"]).stem.lower()) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate {corpus} utterance; keep one original resolution")
    return rows


def _iemocap(root):
    # Accept the common enclosing release directory while requiring all sessions.
    candidates = [root] + sorted(
        p.parent for p in root.glob("*/Session1") if p.is_dir()
    )
    corpus_root = next((p for p in candidates if (p / "Session1").is_dir()), root)
    summary = re.compile(
        r"^\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s+(\S+)\s+(\S+)\s+\[([^]]+)\]\s*$"
    )
    mapping = {
        "ang": "anger",
        "hap": "happiness",
        "sad": "sadness",
        "dis": "disgust",
        "fea": "fear",
        "neu": None,
        "exc": None,
        "fru": None,
        "sur": None,
        "oth": None,
        "xxx": None,
    }
    rows, seen = [], set()
    for session in range(1, 6):
        files = sorted(
            (corpus_root / f"Session{session}/dialog/EmoEvaluation").glob("*.txt")
        )
        if not files:
            raise FileNotFoundError(
                f"IEMOCAP Session{session} consensus annotations missing. {_missing('iemocap', root)}"
            )
        for annotation in files:
            _inside(root, annotation)
            for number, line in enumerate(
                annotation.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                line = line.strip()
                if not line.startswith("[") or line.startswith("[START_TIME"):
                    continue
                match = summary.fullmatch(line)
                if not match:
                    raise ValueError(
                        f"Malformed IEMOCAP annotation {annotation}:{number}"
                    )
                start, stop, uid, code, _ = match.groups()
                identity = re.fullmatch(
                    r"Ses0([1-5])([FM])_(?:impro|script)\w+_[FM]\d{3}", uid
                )
                if not identity or int(identity[1]) != session:
                    raise ValueError(f"Invalid IEMOCAP utterance/session {uid}")
                if (
                    not math.isfinite(float(start))
                    or not math.isfinite(float(stop))
                    or float(stop) <= float(start)
                ):
                    raise ValueError(f"Invalid IEMOCAP timestamps: {uid}")
                if uid in seen:
                    raise ValueError(f"Duplicate IEMOCAP annotation: {uid}")
                seen.add(uid)
                if code not in mapping:
                    raise ValueError(f"Unknown IEMOCAP consensus emotion: {code}")
                if mapping[code] is None:
                    continue
                path = (
                    corpus_root
                    / f"Session{session}/sentences/wav/{uid.rsplit('_', 1)[0]}/{uid}.wav"
                )
                speaker = f"Ses0{session}{uid.rsplit('_', 1)[1][0]}"
                rows.append(_row(root, path, "iemocap", mapping[code], speaker))
    for path in corpus_root.glob("Session*/sentences/wav/*/*.wav"):
        if path.stem not in seen:
            raise ValueError(f"Missing IEMOCAP consensus annotation: {path}")
    if not rows:
        raise ValueError("No selected IEMOCAP utterances")
    return sorted(rows, key=lambda r: r["id"])


def _missing(name, root):
    provider, layout = PROVIDERS[name]
    return f"{name}: original corpus unavailable at {root}. Obtain it from {provider}; expected layout: {layout}."


def verify_corpus_inventory(corpus, root, rows):
    """Fail closed on incomplete manual copies or interrupted acquisitions."""
    paths = _audio_paths(Path(root))
    if corpus == "iemocap":
        paths = [p for p in paths if "/sentences/wav/" in p.as_posix()]
    elif corpus == "ravdess":
        paths = [p for p in paths if p.name.startswith("03-01-")]
    actual, expected = len(paths), RELEASE_COUNTS[corpus]
    if actual != expected:
        raise ValueError(
            f"Incomplete {corpus} release: found {actual} original recordings; expected {expected}. {_missing(corpus, root)}"
        )
    labels = {row["label"] for row in rows}
    required = set(range(len(EMOTIONS))) - (
        {EMOTIONS.index("disgust")} if corpus == "shemo" else set()
    )
    if labels != required:
        raise ValueError(
            f"Incomplete {corpus} selected emotion coverage: expected {[EMOTIONS[i] for i in sorted(required)]}"
        )
    return {
        "original_recordings": actual,
        "expected_original_recordings": expected,
        "selected_recordings": len(rows),
        "class_counts": dict(Counter(EMOTIONS[r["label"]] for r in rows)),
        "provider": PROVIDERS[corpus][0],
    }


def _download(url, destination, *, expected=None, size=None):
    """Resume only after a 206 response; verify length/checksum before publication."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected:
            algorithm, digest = expected.split(":", 1)
            if checksum(destination, algorithm) != digest:
                raise ValueError(f"Cached download checksum mismatch: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DA-MLDG-research-preparation/0.3"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60
        )
    except urllib.error.HTTPError as error:
        if error.code != 416 or size is None or offset != size:
            raise
        response = None
    if response is not None:
        with response:
            resumed = response.status == 206 and response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {offset}-")
            if response.status == 206 and not resumed:
                raise ValueError("Download server returned an unexpected byte range")
            if not resumed:
                offset = 0
            remaining = int(response.headers.get("Content-Length", 0))
            if size is None and remaining:
                size = offset + remaining
            if (
                remaining
                and shutil.disk_usage(destination.parent).free
                < remaining + 64 * 1024**2
            ):
                raise OSError(f"Insufficient free space to download {url}")
            with partial.open("ab" if resumed else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
    if size is not None and partial.stat().st_size != size:
        raise ValueError(f"Incomplete download retained for resume: {partial}")
    if expected:
        algorithm, digest = expected.split(":", 1)
        if checksum(partial, algorithm) != digest:
            partial.unlink()
            raise ValueError(f"Download checksum mismatch: {url}")
    partial.rename(destination)
    return destination


def safe_extract(archive, destination):
    """Reject traversal, symlinks, hardlinks and special files before extracting."""
    archive, destination = Path(archive), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def target(name):
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not (destination / relative).resolve().is_relative_to(destination)
        ):
            raise ValueError(f"Unsafe archive path: {name}")
        return destination / relative

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            for member in members:
                target(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (
                    stat.S_IFMT(mode)
                    and not stat.S_ISREG(mode)
                    and not stat.S_ISDIR(mode)
                ):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            required = sum(member.file_size for member in members)
            if required > shutil.disk_usage(destination).free - 64 * 1024**2:
                raise OSError("Insufficient free space to extract complete archive")
            for member in members:
                path = target(member.filename)
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as inp, path.open("xb") as out:
                        shutil.copyfileobj(inp, out)
    else:
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for member in members:
                target(member.name)
                if not member.isfile() and not member.isdir():
                    raise ValueError(f"Unsafe archive member: {member.name}")
            if (
                sum(m.size for m in members)
                > shutil.disk_usage(destination).free - 64 * 1024**2
            ):
                raise OSError("Insufficient free space to extract complete archive")
            for member in members:
                path = target(member.name)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, path.open("xb") as out:
                        shutil.copyfileobj(inp, out)
    return destination


def download_corpus(name, raw_root):
    """Download complete original open releases; never bypass restricted access."""
    name = canonical_name(name)
    raw_root = Path(raw_root).expanduser().resolve()
    root = raw_root / name
    if root.exists() and _audio_paths(root):
        return root
    cache = raw_root / ".downloads"
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    tess_files = None
    if name in {"ravdess", "cafe"}:
        record = "1188976" if name == "ravdess" else "1478765"
        filename = (
            "Audio_Speech_Actors_01-24.zip" if name == "ravdess" else "CaFE_48k.zip"
        )
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{record}", timeout=30
        ) as response:
            metadata = json.load(response)
        entry = next(f for f in metadata["files"] if f["key"] == filename)
        items = [(entry["links"]["self"], filename, entry["checksum"], entry["size"])]
    elif name == "tess":
        with urllib.request.urlopen(
            "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
            timeout=30,
        ) as response:
            metadata = json.load(response)
        tess_files = metadata["data"]["latestVersion"]["files"]
        if len(tess_files) != 2800 or any(f.get("restricted") for f in tess_files):
            raise ValueError(
                "Unexpected or restricted TESS release; obtain the full original release manually"
            )
        items = [
            (
                "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
                "tess.zip",
                None,
                None,
            )
        ]
    elif name == "emodb":
        items = [
            (
                "http://emodb.bilderbar.info/download/download.zip",
                "emodb.zip",
                None,
                None,
            )
        ]
    elif name == "shemo":
        # Original authors' LFS payloads; their README's pariajm zip links now 404.
        original = "https://media.githubusercontent.com/media/omidmnezami/Persian-Emotional-Speech-Database-ShEMO/master/"
        items = [
            (
                original + "female.zip",
                "shemo-female.zip",
                "sha256:663d8d7dceb44adbbf87d84be3a080fb691d169fbd972f6941e6302bc39f71e6",
                374368830,
            ),
            (
                original + "male.zip",
                "shemo-male.zip",
                "sha256:747670d0d918e59d5e5020cf66610242890b093d42968c467c4f02eb5e6a6b1a",
                496606401,
            ),
        ]
    elif name == "crema_d":
        # The provider repository uses LFS. A recursive Git tree is complete;
        # GitHub's contents API silently caps this directory at 1000 entries.
        with urllib.request.urlopen(
            "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/ref/heads/master",
            timeout=30,
        ) as response:
            revision = json.load(response)["object"]["sha"]
        with urllib.request.urlopen(
            f"https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/{revision}?recursive=1",
            timeout=30,
        ) as response:
            tree = json.load(response)
        if tree.get("truncated"):
            raise ValueError(
                "CREMA-D provider tree truncated; use the git-lfs original release"
            )
        paths = sorted(
            item["path"]
            for item in tree["tree"]
            if item["path"].startswith("AudioWAV/") and item["path"].endswith(".wav")
        )
        if len(paths) != 7442:
            raise ValueError(
                f"Unexpected CREMA-D release size: {len(paths)} instead of 7442"
            )
        stage = raw_root / ".crema_d.partial"
        stage.mkdir(parents=True, exist_ok=True)
        for relative in paths:
            _download(
                f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{revision}/{relative}",
                stage / relative,
            )
        for path in _audio_paths(stage):
            read_wave(path)  # Reject Git LFS pointer files and interrupted payloads.
        write_json(
            stage / "acquisition.json",
            {
                "provider": PROVIDERS[name][0],
                "revision": revision,
                "records": len(paths),
            },
        )
        if root.exists():
            raise FileExistsError(f"Unrecognized existing CREMA-D directory: {root}")
        stage.rename(root)
        return root
    else:
        raise FileNotFoundError(_missing(name, root))
    # Stage all parts together. Interrupted extraction never appears as a corpus.
    archives = [
        _download(url, cache / filename, expected=digest, size=size)
        for url, filename, digest, size in items
    ]
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=raw_root) as directory:
        stage = Path(directory) / name
        stage.mkdir()
        for archive in archives:
            safe_extract(archive, stage)
        if not _audio_paths(stage):
            raise ValueError(f"Download contained no audio for {name}")
        if tess_files is not None:
            found = {p.name: p for p in _audio_paths(stage)}
            if len(found) != 2800:
                raise ValueError(
                    f"Incomplete TESS archive: {len(found)} of 2800 recordings"
                )
            for entry in tess_files:
                info = entry["dataFile"]
                filename = {"OA_bite_neutral.wav": "OAF_bite_neutral.wav"}.get(
                    info["filename"], info["filename"]
                )
                if (
                    filename not in found
                    or checksum(found[filename], "md5") != info["md5"]
                ):
                    raise ValueError(
                        f"TESS provider checksum mismatch: {info['filename']}"
                    )
        verify_corpus_inventory(name, stage, parse_corpus(name, stage))
        if root.exists():
            raise FileExistsError(
                f"Incomplete or unrecognized corpus exists: {root}; preserve it and select a fresh raw root"
            )
        stage.rename(root)
    return root


def _corpus_root(raw_root, name, download):
    raw_root = Path(raw_root).expanduser().resolve()
    root = raw_root / name
    if root.is_dir() and _audio_paths(root):
        return root
    aliases = (
        [p for p in raw_root.iterdir() if p.is_dir() and canonical_name(p.name) == name]
        if raw_root.is_dir()
        else []
    )
    if len(aliases) == 1 and _audio_paths(aliases[0]):
        return aliases[0]
    if download:
        try:
            return download_corpus(name, raw_root)
        except (OSError, ValueError, StopIteration) as error:
            raise ValueError(
                f"Automatic acquisition failed: {error}. {_missing(name, root)}"
            ) from error
    raise FileNotFoundError(_missing(name, root))


def _counts(rows, classes):
    counts = Counter((r["domain"], r["label"]) for r in rows)
    return {
        domain: {label: counts[domain, i] for i, label in enumerate(classes)}
        for domain in sorted({r["domain"] for r in rows})
    }


def _publish(path, value, decode=None):
    from .data import load_manifest, validate_manifest

    value["records"] = check_records(
        value["records"], value["classes"], None, decode=decode
    )
    value["counts"] = _counts(value["records"], value["classes"])
    path = Path(path)
    value = validate_manifest(value, path.parent)
    if path.exists():
        previous = load_manifest(path)
        if previous != value:
            raise FileExistsError(
                f"Manifest already exists with different contents: {path}; choose another --data-root"
            )
        return path
    write_json(path, value)
    load_manifest(path)
    return path


def prepare_scenario(scenario, raw_root="raw", data_root="data", *, download=True):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    sources, target = SCENARIOS[scenario]
    rows, errors, inventory = [], [], {}
    for corpus in sources + [target]:
        try:
            root = _corpus_root(raw_root, corpus, download)
            parsed = parse_corpus(corpus, root)
            inventory[corpus] = verify_corpus_inventory(corpus, root, parsed)
            for row in parsed:
                row["split"] = "target" if corpus == target else "source"
            rows.extend(parsed)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    if errors:
        raise ValueError(f"{scenario} could not be prepared:\n" + "\n".join(errors))
    return _publish(
        Path(data_root) / f"{scenario}.json",
        {
            "schema": 1,
            "dataset": "ser",
            "scenario": scenario,
            "representation": "audio",
            "classes": list(EMOTIONS),
            "sources": sources,
            "target": target,
            "selection": "all original utterances in five selected emotions; IEMOCAP excited excluded",
            "inventory": inventory,
            "records": rows,
        },
        decode=read_wave,
    )


def prepare_corpus(name, raw_root="raw", data_root="data", *, download=True):
    root = _corpus_root(raw_root, name, download)
    rows = parse_corpus(name, root)
    inventory = verify_corpus_inventory(name, root, rows)
    for row in rows:
        row["split"] = "source"
    # Individual indexes are acquisition artifacts, not training DG manifests.
    value = {
        "schema": 1,
        "dataset": name,
        "classes": list(EMOTIONS),
        "representation": "audio",
        "provider": PROVIDERS[name][0],
        "inventory": inventory,
        "records": check_records(rows, EMOTIONS, None, decode=read_wave),
    }
    value["counts"] = _counts(value["records"], EMOTIONS)
    path = Path(data_root) / f"{name}.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"Different manifest already exists: {path}")
    else:
        write_json(path, value)
    return path


def prepare_pacs(root, data_root="data", target="sketch"):
    """Use official per-domain train/crossval source lists and target test list.

    A complete extracted domain/class tree is also accepted (all source images),
    including the full 9991-image Flower parquet conversion. Official list labels
    are one based and must agree with the class directory.
    """
    if target not in PACS:
        raise ValueError(f"Invalid PACS target: {target}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(_missing("pacs", root))
    rows = []
    split_paths = list(root.rglob("*_train_kfold.txt"))
    image_roots = [root / "kfold", root / "images", root]
    image_roots.extend(
        sorted({p.parent for p in root.glob("*/art_painting") if p.is_dir()})
    )
    image_root = next(
        (p for p in image_roots if all((p / d).is_dir() for d in PACS)), None
    )
    if image_root is None:
        raise FileNotFoundError(_missing("pacs", root))
    if split_paths:
        for domain in PACS:
            suffixes = ["test"] if domain == target else ["train", "crossval"]
            for suffix in suffixes:
                candidates = list(root.rglob(f"{domain}_{suffix}_kfold.txt"))
                if suffix == "crossval" and not candidates:
                    candidates = list(root.rglob(f"{domain}_val_kfold.txt"))
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected exactly one official PACS {domain}/{suffix} split file"
                    )
                for number, line in enumerate(
                    candidates[0].read_text().splitlines(), 1
                ):
                    fields = line.rsplit(None, 1)
                    if len(fields) != 2:
                        raise ValueError(
                            f"Malformed PACS split at {candidates[0]}:{number}"
                        )
                    relative, raw_label = fields
                    parts = Path(relative).parts
                    if parts and parts[0] == "kfold":
                        parts = parts[1:]
                    if (
                        len(parts) != 3
                        or parts[0] != domain
                        or parts[1] not in IMAGE_CLASSES
                        or ".." in parts
                    ):
                        raise ValueError(f"Invalid PACS split path: {relative}")
                    label = IMAGE_CLASSES.index(parts[1])
                    if raw_label != str(label + 1):
                        raise ValueError(
                            f"PACS split label disagrees with class directory: {relative}"
                        )
                    path = _inside(root, image_root.joinpath(*parts))
                    rows.append(
                        {
                            "id": "/".join(parts),
                            "path": str(path),
                            "domain": domain,
                            "label": label,
                            "split": "target" if domain == target else "source",
                            "official_split": suffix,
                        }
                    )
    else:
        for domain in PACS:
            for path in sorted((image_root / domain).rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                relative = path.relative_to(image_root)
                if len(relative.parts) != 3 or relative.parts[1] not in IMAGE_CLASSES:
                    raise ValueError(f"Invalid PACS domain/class path: {path}")
                rows.append(
                    {
                        "id": relative.as_posix(),
                        "path": str(_inside(root, path)),
                        "domain": domain,
                        "label": IMAGE_CLASSES.index(relative.parts[1]),
                        "split": "target" if domain == target else "source",
                    }
                )
    inventory = {}
    for domain in PACS:
        domain_rows = [r for r in rows if r["domain"] == domain]
        original = [
            p
            for p in (image_root / domain).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(original) != PACS_COUNTS[domain] or {
            r["label"] for r in domain_rows
        } != set(range(len(IMAGE_CLASSES))):
            raise ValueError(
                f"Incomplete PACS domain {domain}: expected {PACS_COUNTS[domain]} original images with seven selected classes; received {len(original)} original / {len(domain_rows)} selected"
            )
        inventory[domain] = {
            "original_recordings": len(original),
            "selected_recordings": len(domain_rows),
            "expected_original_recordings": PACS_COUNTS[domain],
        }
    return _publish(
        Path(data_root) / f"pacs-{target}.json",
        {
            "schema": 1,
            "dataset": "pacs",
            "representation": "image",
            "classes": IMAGE_CLASSES,
            "sources": [d for d in PACS if d != target],
            "target": target,
            "selection": "official source train+crossval / target test"
            if split_paths
            else "all domain/class images",
            "inventory": inventory,
            "records": rows,
        },
    )


def download_pacs(raw_root):
    """Acquire the complete 9991-image PACS conversion, never an implicit subset."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "PACS automatic acquisition requires pyarrow; install the data extra or supply the official extracted layout"
        ) from error
    root = Path(raw_root).expanduser().resolve() / "pacs"
    if root.is_dir() and all((root / d).is_dir() for d in PACS):
        return root
    api = "https://huggingface.co/api/datasets/flwrlabs/pacs/tree/main/data"
    with urllib.request.urlopen(api, timeout=30) as response:
        files = [f for f in json.load(response) if f["path"].endswith(".parquet")]
    if not files:
        raise ValueError("No PACS provider parquet files")
    cache = root.parent / ".downloads"
    archives = [
        _download(
            "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/" + f["path"],
            cache / Path(f["path"]).name,
            size=f["size"],
        )
        for f in files
    ]
    with tempfile.TemporaryDirectory(prefix=".pacs-", dir=root.parent) as temporary:
        stage = Path(temporary) / "pacs"
        count = 0
        for archive in archives:
            for batch in pq.ParquetFile(archive).iter_batches(batch_size=32):
                for entry in batch.to_pylist():
                    domain, label = entry["domain"], entry["label"]
                    if (
                        domain not in PACS
                        or type(label) is not int
                        or not 0 <= label < len(IMAGE_CLASSES)
                    ):
                        raise ValueError("Invalid PACS parquet domain or label")
                    image = entry["image"]["bytes"]
                    if not image:
                        raise ValueError("PACS parquet lacks embedded image bytes")
                    destination = (
                        stage / domain / IMAGE_CLASSES[label] / f"{count:06d}.jpg"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(image)
                    count += 1
        if count != 9991:
            raise ValueError(
                f"Incomplete PACS release: expected 9991 images; received {count}"
            )
        write_json(
            stage / "acquisition.json",
            {
                "provider": "https://huggingface.co/datasets/flwrlabs/pacs",
                "records": count,
                "label_order": IMAGE_CLASSES,
                "selection": "full released dataset; no row limit",
            },
        )
        stage.rename(root)
    return root


def prepare_voxceleb(root, data_root="data"):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(_missing("voxceleb", root))
    paths = _audio_paths(root)
    identities = []
    for path in paths:
        candidates = [
            p for p in path.relative_to(root).parts if re.fullmatch(r"id\d{5}", p)
        ]
        if len(candidates) != 1:
            raise ValueError(f"Invalid VoxCeleb speaker path: {path}")
        identities.append(candidates[0])
    classes = sorted(set(identities))
    if (
        len(classes) != VOXCELEB_COUNTS["speakers"]
        or len(paths) != VOXCELEB_COUNTS["recordings"]
    ):
        raise ValueError(
            f"Incomplete VoxCeleb1 release: expected {VOXCELEB_COUNTS['recordings']} recordings / {VOXCELEB_COUNTS['speakers']} speakers, found {len(paths)} / {len(classes)}. {_missing('voxceleb', root)}"
        )
    labels = {speaker: i for i, speaker in enumerate(classes)}
    rows = [
        {
            "id": path.relative_to(root).as_posix(),
            "path": str(path.resolve()),
            "speaker": speaker,
            "domain": "voxceleb",
            "label": labels[speaker],
            "split": "source",
        }
        for path, speaker in zip(paths, identities)
    ]
    return _publish(
        Path(data_root) / "voxceleb.json",
        {
            "schema": 1,
            "dataset": "voxceleb",
            "representation": "spectrogram",
            "classes": classes,
            "sources": ["voxceleb"],
            "selection": "all provided speaker recordings",
            "records": rows,
        },
        decode=read_wave,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare full native SER corpora, all available Table I scenarios, PACS, and VoxCeleb. Missing datasets are reported together."
    )
    parser.add_argument(
        "--dataset",
        help="Corpus name, pacs, or voxceleb; omitted attempts every dataset",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Prepare exactly the selected Table I source/target scenario",
    )
    parser.add_argument("--raw-root", default="raw", type=Path)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--target", choices=PACS, help="PACS held-out domain; omitted prepares all four"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Index local corpora only, without network access",
    )
    args = parser.parse_args(argv)
    if args.dataset and args.scenario:
        parser.error("--dataset and --scenario are mutually exclusive")
    if args.scenario:
        tasks = [("scenario", args.scenario)]
    elif args.dataset:
        name = canonical_name(args.dataset)
        if name not in PROVIDERS:
            parser.error("Unknown dataset: " + args.dataset)
        tasks = [("dataset", name)]
    else:
        tasks = [("dataset", name) for name in PROVIDERS] + [
            ("scenario", name) for name in SCENARIOS
        ]
    errors = []
    for kind, name in tasks:
        try:
            if kind == "scenario":
                paths = [
                    prepare_scenario(
                        name,
                        args.raw_root,
                        args.data_root,
                        download=bool(args.scenario) and not args.no_download,
                    )
                ]
            elif name == "pacs":
                root = args.raw_root / "pacs"
                if not root.is_dir() and not args.no_download:
                    root = download_pacs(args.raw_root)
                paths = [
                    prepare_pacs(root, args.data_root, target)
                    for target in ([args.target] if args.target else PACS)
                ]
            elif name == "voxceleb":
                paths = [prepare_voxceleb(args.raw_root / "voxceleb", args.data_root)]
            else:
                paths = [
                    prepare_corpus(
                        name,
                        args.raw_root,
                        args.data_root,
                        download=not args.no_download,
                    )
                ]
            for path in paths:
                print(f"{name}: ready: {path.resolve()}")
        except (OSError, ValueError, RuntimeError, StopIteration) as error:
            errors.append((name, str(error)))
            print(f"{name}: unavailable")
    if errors:
        print("\nPreparation incomplete; no failed scenario manifest was published:")
        for name, message in errors:
            print(f"- {name}: {message}")
        return 1
    return 0
