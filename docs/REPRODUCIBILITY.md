# Reproducibility record

## Tested implementation

The maintained package is `damldg/`. Historical `src/` is retained as source evidence and is not imported by the maintained commands. Base public checkout: `e78dff95a4884bb0e15155af585299e53b13cc7a`; original public source snapshot: `8a95bc8003777f84a0cb2ea18efd9a53e929507d`. See [provenance](../PROVENANCE.md) and [method mapping](METHOD.md).

The CPU environment uses Python 3.11.15, TensorFlow 2.20.0, Keras 3.15.1, NumPy 1.26.4, ml-dtypes 0.5.4, librosa 0.11.0, SoundFile 0.13.1 and SciPy 1.17.1 on macOS arm64. `requirements-lock-py311.txt` records the resolved direct/transitive runtime environment; it is not a platform-independent wheel lock. Python 3.12/Linux CI are supported targets, not locally measured platforms. No GPU run or full corpus benchmark was performed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-py311.txt
python -m pip install -e '.[test]'
python -m pip check
python -m pytest -q
damldg train --config configs/smoke.json
```

The tests establish: analytical domain-loss/energy definitions; central finite-difference agreement for MLDG and DA-MLDG; actual meta-test-label influence on parameter updates; a measurable second-order contribution; absence of stateless parameter/BN side effects; full ResNet-18 update and serialization; correct scenario 5/source-target separation; deterministic augmentation; generated WAV→manifest→full training; strict classifier-independent encoder transfer; RSC sample/label alignment. Synthetic smokes also execute all four maintained methods.

## Experiment controls

Use the seven scenario configs with prepared local data and a compatible pretrained encoder for paper-oriented experiments. The defaults follow the stated optimizer and crop values; source discrepancies and additional implementation choices are explicit in METHOD.md. There is no early stopping or target-driven model selection. Target performance is evaluated only after the configured epochs and the final model is saved.

Seed controls Python/NumPy/TensorFlow, domain batches, stochastic augmentation and meta-test-domain choice. TensorFlow deterministic operations are enabled. Use ten full runs with separately recorded seeds to match the paper's averaging protocol; exact original seeds are unknown. Ten seeds such as 2021–2030 are a new reproducible choice, not recovered historical evidence. Repeated tiny CPU runs test deterministic execution; hardware/library changes can alter trajectories.

Balanced source batches allocate `batch_size // domains` examples per domain, distributing the remainder by sorted domain order. Smaller domains cycle with reshuffling. An epoch covers the largest source domain; set `steps_per_epoch` only for explicitly bounded experiments. These semantics differ from the historical pooled-shuffle sampler and are documented rather than hidden.

## Artifacts

Each run refuses to overwrite an existing output directory and writes:

- `config.json`: resolved config, including initialization and augmentation choices.
- `partitions.json`: class order and source/target records, without generated feature arrays. Real paths are local absolute paths.
- `environment.json`: package versions, interpreter/platform, devices, manifest SHA256 and pretrained encoder SHA256 where applicable.
- `metrics.json`: method, initialization, explicit `benchmark_reproduction: false`, training duration, epoch/step losses and gamma, selected meta-test domains, and final target accuracy/loss/count. Pretraining with no target records writes `target: null`.
- `model.keras`: final classifier. `encoder.keras`: ResNet spatial feature extractor for reuse with another classifier.

Load artifacts with `tf.keras.models.load_model`. No corpus, model checkpoint, environment, or generated result is committed to the repository. Record prepared audio/feature checksums separately before comparing benchmark results.

## Scientific boundaries

The published results are not reproduced by this maintenance release. Missing artifacts include original pretrained VoxCeleb/ImageNet weights, raw-corpus normalization/annotation converters, exact split/seed ledger, TRILLsson extraction/tensor mapping, and the code used to produce every reported table. The repaired gradient is mathematically validated; it does not prove what historical implementation generated the paper's numbers.

The paper's own Table II printed averages contain arithmetic inconsistencies; project pages retain direct scenario rows instead of relying on those aggregate claims. Main comparisons report ten-run means without dispersion. This release makes no stronger efficacy claim. It adds no federated/distributed system or formal privacy mechanism.
