"""Explicit local data preparation and training; no import-time downloads."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DA-MLDG source-domain training")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--kind", choices=["audio", "image", "features", "speaker"], default="audio"
    )
    prep.add_argument("--scenario", default="scn1")
    prep.add_argument("--target", default="art_painting")
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/smoke.json")
    train.add_argument("--output")
    train.add_argument("--seed", type=int)
    train.add_argument("--method", choices=["deepall", "rsc", "mldg", "damldg"])
    args = parser.parse_args()
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if args.command == "prepare":
        from .data import (
            read_audio_manifest,
            index_folders,
            EMOTIONS,
            IMAGE_CLASSES,
            SCENARIOS,
            PACS,
        )

        representation = args.kind
        classes = EMOTIONS
        if args.kind == "audio":
            rows = read_audio_manifest(args.root, args.scenario)
        elif args.kind == "image":
            if args.target not in PACS:
                raise ValueError("Unknown PACS target domain")
            classes = IMAGE_CLASSES
            rows = index_folders(
                args.root,
                [d for d in PACS if d != args.target],
                args.target,
                classes,
                (".jpg", ".jpeg", ".png"),
            )
        elif args.kind == "features":
            sources, target = SCENARIOS[args.scenario]
            rows = index_folders(args.root, sources, target, classes, (".npy",))
        else:
            representation = "spectrogram"
            classes = sorted(p.name for p in args.root.iterdir() if p.is_dir())
            rows = []
            for label, speaker in enumerate(classes):
                for path in sorted((args.root / speaker).rglob("*.wav")):
                    rows.append(
                        {
                            "id": str(path.relative_to(args.root)),
                            "path": str(path.resolve()),
                            "domain": "voxceleb",
                            "label": label,
                            "split": "source",
                        }
                    )
            if len(classes) < 2 or not rows:
                raise ValueError("Need at least two speaker directories with WAV files")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            json.dump(
                {
                    "schema": 1,
                    "representation": representation,
                    "scenario": (
                        args.scenario if args.kind in {"audio", "features"} else None
                    ),
                    "classes": classes,
                    "records": rows,
                },
                file,
                indent=2,
            )
        print(f"Indexed {len(rows)} examples in {len(classes)} classes")
    else:
        from .train import load_config, run

        run(
            load_config(
                args.config, output=args.output, seed=args.seed, method=args.method
            )
        )


if __name__ == "__main__":
    main()
