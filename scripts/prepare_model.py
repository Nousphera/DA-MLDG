"""Prepare VoxCeleb/ImageNet ResNet-18 encoders or TRILLsson embeddings."""

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from damldg.config import Config, resolve_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=("voxceleb", "imagenet", "trillsson"), default="voxceleb"
    )
    p.add_argument("--model-root", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--feature-root", type=Path, default=Path("features"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--scenario", choices=[f"scn{i}" for i in range(1, 8)])
    p.add_argument(
        "--encoder", type=Path, help="Import an existing DA-MLDG PyTorch encoder"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Pretraining gradient cap; 0 disables clipping",
    )
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-bn-calibration", action="store_true")
    args = p.parse_args(argv)
    try:
        import torch

        torch.set_num_threads(args.threads)
        device = resolve_device(args.device)
        args.model_root.mkdir(parents=True, exist_ok=True)
        if args.model == "imagenet":
            from damldg.weights import prepare_imagenet

            output = prepare_imagenet(args.model_root)
        elif args.model == "trillsson":
            from damldg.weights import load_trillsson, prepare_embeddings

            manifest = args.manifest or (
                args.data_root / f"{args.scenario}.json" if args.scenario else None
            )
            if manifest:
                output = prepare_embeddings(
                    manifest, args.model_root, args.feature_root, device
                )
            else:
                load_trillsson(args.model_root, device)
                output = args.model_root / "trillsson3.pt"
        else:
            from damldg.model import build_model, load_encoder

            output = args.model_root / "voxceleb-resnet18.pt"
            model = build_model()
            if args.encoder or output.exists():
                source = args.encoder or output
                load_encoder(model, source)
                if source.resolve() != output.resolve():
                    if output.exists():
                        raise FileExistsError(output)
                    shutil.copyfile(source, output)
            else:
                from damldg.training import run_trial, write_json, digest

                manifest = args.manifest or args.data_root / "voxceleb.json"
                from damldg.data import load_manifest

                if load_manifest(manifest)["representation"] != "spectrogram":
                    raise ValueError(
                        "VoxCeleb pretraining requires the speaker spectrogram manifest"
                    )
                cfg = Config(
                    manifest=str(manifest.resolve()),
                    method="deepall",
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    max_grad_norm=args.max_grad_norm,
                    calibrate_bn=not args.no_bn_calibration,
                    seed=args.seed,
                    initialization="random",
                    encoder="",
                    trials=1,
                    augment=False,
                    device=device,
                    output=str(
                        (
                            args.model_root / f"voxceleb-pretraining-seed{args.seed}"
                        ).resolve()
                    ),
                    backend="sequential",
                    threads=args.threads,
                )
                metrics = run_trial(cfg, 0)
                shutil.copyfile(
                    Path(cfg.output) / f"seed-{cfg.seed}" / "encoder.pt", output
                )
                write_json(
                    output.with_suffix(".json"),
                    {
                        "training": metrics,
                        "manifest_sha256": digest(manifest),
                        "sha256": digest(output),
                    },
                )
        print(f"Ready: {output}")
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, f"Model preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
