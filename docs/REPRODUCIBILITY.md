# Experiment guide

## Environment

Use Python 3.11 or 3.12 and the [setup script](../scripts/setup_env.sh).
`requirements-lock-py311.txt` records a Python 3.11.15/macOS arm64 CPU environment
with TensorFlow 2.20.0, Keras 3.15.1, NumPy 1.26.4, ml-dtypes 0.5.4,
librosa 0.11.0, SoundFile 0.13.1 and SciPy 1.17.1.

```bash
python -m pip check
python -m pytest -q
damldg train --config configs/smoke.json --output runs/smoke
```

Tests cover analytical losses/energies, finite-difference meta-gradients,
second-order contributions, batch-normalization state isolation, full ResNet-18
updates/serialization, source/target separation, deterministic augmentation,
prepared WAV training, encoder transfer and RSC input/label alignment.

## Experiment controls

Prepare the selected scenario's corpus manifest using the [data guide](DATA.md).
Run `damldg validate --manifest PATH` before training. The seven scenario configs
use random initialization unless `pretrained_encoder` is set. For pretrained
experiments, supply a compatible speaker encoder for SER or ImageNet encoder
for PACS and record its hash.

The [method guide](METHOD.md) defines the second-order objective, BN state and
sampling rules. Balanced source batches allocate `batch_size // domains`
examples per domain and distribute the remainder by sorted domain order.
Smaller domains cycle with reshuffling until the largest is covered.
`steps_per_epoch` can bound an experiment explicitly.

The seed controls Python/NumPy/TensorFlow, domain batching, augmentation and
meta-test-domain choice. TensorFlow deterministic operations are enabled.
Use ten runs with recorded seeds for the paper's averaging protocol; example
seeds are 2021–2030. Report means and variation. Hardware and library changes
can affect trajectories.

Target-domain data is excluded from optimization. Training runs for the
configured epochs, then evaluates the final model on the target. There is no
early stopping or target-driven model selection.

## Outputs

Each run creates a fresh output directory:

| File | Contents |
|---|---|
| `config.json` | Resolved settings, initialization and augmentation choices |
| `partitions.json` | Class order and source/target records; real paths are local absolute paths |
| `environment.json` | Packages, interpreter/platform/devices, manifest and encoder hashes |
| `metrics.json` | Dataset/representation, method, initialization, duration, losses, gamma, meta-test domains and final target accuracy/loss/count |
| `model.keras` | Final classifier |
| `encoder.keras` | ResNet spatial feature extractor for transfer |

Speaker pretraining has no target split and writes `target: null`. Load model
artifacts with `tf.keras.models.load_model`. Preserve data checksums and encoder
identifiers with the run; share local paths according to the datasets' terms.
Prepared feature experiments additionally need the extraction model/version,
preprocessing and tensor mapping recorded alongside their manifest.
