"""Run DA-MLDG and its baselines as reproducible Flower trials."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from .config import Config, METHODS, resolve_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario", choices=[f"scn{i}" for i in range(1, 8)], default="scn7"
    )
    p.add_argument("--manifest", type=Path)
    p.add_argument("--method", choices=METHODS, default="damldg")
    p.add_argument(
        "--representation", choices=("audio", "trillsson", "image"), default="audio"
    )
    p.add_argument(
        "--pacs-target", choices=("art_painting", "cartoon", "photo", "sketch")
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--encoder", type=Path)
    p.add_argument(
        "--initialization", choices=("pretrained", "random"), default="pretrained"
    )
    p.add_argument("--backend", choices=("flower", "sequential"), default="flower")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-bn-calibration", action="store_true")
    defaults = Config()
    for key in ("epochs", "batch_size", "seed", "trials", "threads"):
        p.add_argument(
            "--" + key.replace("_", "-"), type=int, default=getattr(defaults, key)
        )
    for key in (
        "learning_rate",
        "inner_lr",
        "momentum",
        "max_grad_norm",
        "registration_timeout",
        "reply_timeout",
    ):
        p.add_argument(
            "--" + key.replace("_", "-"), type=float, default=getattr(defaults, key)
        )
    p.add_argument("--parallel-trials", type=int, default=1)
    p.add_argument("--worker-cpus", type=int, default=2)
    p.add_argument("--worker-gpus", type=float)
    p.add_argument("--dry-run", action="store_true")
    return p


def build_config(args):
    name = f"pacs-{args.pacs_target}" if args.pacs_target else args.scenario
    if args.representation == "image" and not args.pacs_target and not args.manifest:
        raise ValueError("Select --pacs-target or --manifest for images")
    if args.pacs_target and args.representation == "trillsson":
        raise ValueError("PACS uses images, not TRILLsson embeddings")
    if args.representation == "trillsson":
        name += "-trillsson"
    manifest = args.manifest or args.data_root / f"{name}.json"
    encoder = args.encoder or Path("checkpoints") / (
        "imagenet-resnet18.pt"
        if args.pacs_target or args.representation == "image"
        else "voxceleb-resnet18.pt"
    )
    if args.parallel_trials < 1 or args.worker_cpus < 1:
        raise ValueError("Worker counts must be positive")
    if args.worker_gpus is not None and (
        not __import__("math").isfinite(args.worker_gpus) or args.worker_gpus < 0
    ):
        raise ValueError("worker-gpus must be finite and nonnegative")
    values = {
        key: getattr(args, key)
        for key in (
            "method",
            "epochs",
            "batch_size",
            "learning_rate",
            "inner_lr",
            "momentum",
            "max_grad_norm",
            "seed",
            "trials",
            "threads",
            "backend",
            "initialization",
            "registration_timeout",
            "reply_timeout",
        )
    }
    return Config(
        **values,
        manifest=str(manifest.expanduser().resolve()),
        encoder=str(encoder.expanduser().resolve()),
        output=str(
            (args.output or Path("results") / f"{args.method}-{name}")
            .expanduser()
            .resolve()
        ),
        device=resolve_device(args.device),
        augment=not args.no_augment,
        calibrate_bn=not args.no_bn_calibration,
    )


@contextmanager
def flower_app(cfg):
    import tomli_w

    descriptor = {
        "project": {
            "name": "damldg",
            "version": "2.0.0",
            "description": __doc__,
            "license": "LicenseRef-NotSpecified",
            "requires-python": ">=3.11,<3.14",
            "dependencies": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "nousphera",
                    "components": {
                        "serverapp": "damldg.server_app:app",
                        "clientapp": "damldg.client_app:app",
                    },
                    "config": cfg,
                }
            }
        },
    }
    with TemporaryDirectory(prefix="damldg-flower-") as folder:
        path = Path(folder)
        shutil.copytree(
            Path(__file__).parent,
            path / "damldg",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (path / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield path


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        if args.dry_run:
            print(json.dumps(cfg.as_dict(), indent=2))
            return
        from .training import prepare_run

        prepare_run(cfg)
        output = Path(cfg.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}; choose --output")
        output.mkdir(parents=True, exist_ok=True)
        if cfg.backend == "sequential":
            from .server_app import run_experiment

            run_experiment(cfg)
        else:
            import tomli_w

            gpus = (
                args.worker_gpus
                if args.worker_gpus is not None
                else float(cfg.device.startswith("cuda"))
            )
            if cfg.device.startswith("cuda") and gpus <= 0:
                raise ValueError("CUDA workers need positive --worker-gpus")
            env = os.environ.copy()
            env["PATH"] = (
                str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            )
            env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            federation = f"num-supernodes={cfg.trials} client-resources-num-cpus={args.worker_cpus} client-resources-num-gpus={gpus} init-args-num-cpus={args.worker_cpus * min(args.parallel_trials, cfg.trials)}"
            with flower_app(cfg.as_dict()) as app:
                config_file = app / "run.toml"
                config_file.write_text(tomli_w.dumps(cfg.as_dict()))
                command = [
                    str(Path(sys.executable).with_name("flwr")),
                    "run",
                    str(app),
                    "local",
                    "--run-config",
                    str(config_file),
                    "--federation-config",
                    federation,
                    "--stream",
                ]
                print(shlex.join(command), flush=True)
                from .runtime import local_service

                with local_service(env, output / "flower-service.log") as runtime_env:
                    subprocess.run(command, env=runtime_env, check=True)
        result = json.loads((output / "results.json").read_text())
        if len(result["trials"]) != cfg.trials:
            raise RuntimeError("Not all trials completed")
        print(f"Saved {output}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
