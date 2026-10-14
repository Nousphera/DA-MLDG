# DA-MLDG

**On efficacy of Meta-Learning for Domain Generalization in Speech Emotion Recognition**

Raeshak King Gandhi · Vasileios Tsouvalas · Nirvana Meratnia · PerFail @ IEEE PerCom 2023

[Paper](https://ieeexplore.ieee.org/document/10150265/) · [Project](https://nousphera.github.io/DA_MLDG/) · [Poster](https://nousphera.github.io/DA_MLDG/poster.pdf)

## Abstract

Speech Emotion Recognition (SER) refers to the recognition of human emotions from natural speech, vital for building human-centered context-aware intelligent systems. Here, domain shift, where models’ trained on one domain exhibit performance degradation when exposed to an unseen domain with different statistics, is a major limiting factor in SER applicability, as models have a strong dependence on speakers and languages characteristics used during training. Meta-Learning for Domain Generalization (MLDG) has shown great success in improving models’ generalization capacity and alleviate the domain shift problem in the vision domain; yet, its’ efficacy on SER remains largely unexplored. In this work, we propose a “domain-shift aware” MLDG approach to learn generalizable models across multiple domains in SER. Based on our extensive evaluation, we identify a number of pitfalls that contribute to poor models’ DG ability, and demonstrate that log-mel spectrograms representations lack distinct features required for MLDG in SER. We further explore the use of appropriate features to achieve DG in SER as to provide insides to future research directions for DG in SER.

## Setup

Use Python 3.11 (tested on macOS arm64 CPU); Python 3.12 is supported but untested.
A CPU is sufficient for the generated-data smoke.

```bash
git clone https://github.com/Nousphera/DA_MLDG.git
cd DA_MLDG
bash scripts/setup_env.sh "$HOME/.venvs/damldg"
source "$HOME/.venvs/damldg/bin/activate"
python -m pytest -q
```

The script creates a new virtual environment, installs `.[test]` using the tested
dependency constraints, and runs `pip check`. It refuses to modify an existing
environment. Training uses the maintained `damldg/` package.

## Datasets

Acquire the selected speech corpora from their providers under their access terms.
The [data guide](docs/DATA.md) lists all ten corpora, seven scenarios, label
mapping requirements, and import commands. For every raw corpus, supply a CSV
with provider-derived `path,emotion` labels; the same importer validates, hashes
and copies the recordings into the model's normalized layout:

```bash
python scripts/prepare_data.py import-corpus --corpus tess \
  --root /path/to/TESS --annotations /path/to/tess-labels.csv \
  --output datasets/ser/tess
```

Repeat for EmoDB and ShEMO to run scenario 7, then prepare and validate its manifest:

```bash
python scripts/prepare_data.py prepare --root datasets/ser --scenario scn7 --output data/scn7.json
python scripts/prepare_data.py validate --manifest data/scn7.json
```

If the original normalized layout already exists, start with `prepare`.
The five class IDs are anger, happiness, sadness, disgust and fear. The importer
preserves speaker/session paths and audio bytes; it never guesses labels or
fabricates missing classes. PACS images, speaker pretraining and feature-array
inputs have [separate complete recipes](docs/DATA.md).

## Run

A bounded smoke trains the full ResNet-18 using a second-order DA-MLDG update
on generated source-domain features, then evaluates a held-out domain:

```bash
damldg train --config configs/smoke.json --output runs/smoke
```

For prepared scenario 7 audio:

```bash
damldg train --config configs/scn7.json --seed 2021 --output runs/scn7-seed2021
```

Use `configs/scn1.json` through `configs/scn7.json` for the other scenarios after
preparing their manifests. Select a baseline with `--method mldg`, `--method deepall`
or `--method rsc`, and a fresh `--output` directory. This is centralized domain
generalization; it does not require Flower.

## Outputs and reproducibility

Each run saves `config.json`, `partitions.json`, `environment.json`, `metrics.json`,
`model.keras`, and `encoder.keras`. Targets stay outside parameter updates and
are evaluated after training. The configs use random initialization until a
compatible `pretrained_encoder` is explicitly supplied; they are not the paper's
pretrained setup without that artifact.

The [method mapping](docs/METHOD.md) explains the repaired higher-order gradient,
energy weighting and source-domain batching. [Reproducibility](docs/REPRODUCIBILITY.md)
and [provenance](PROVENANCE.md) record source differences and validation.
Full published benchmarks, original pretrained checkpoints and the exact TRILLsson
tensor preparation remain unavailable or unverified. The paper's limited and
uneven generalization results are not replaced by smoke-test accuracy.

## Citation

```bibtex
@inproceedings{gandhi2023damldg,
  author = {Gandhi, Raeshak King and Tsouvalas, Vasileios and Meratnia, Nirvana},
  title = {On efficacy of Meta-Learning for Domain Generalization in Speech Emotion Recognition},
  booktitle = {2023 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)},
  year = {2023},
  pages = {421--426},
  note = {PerFail 2023: Second International Workshop on Negative Results in Pervasive Computing},
  doi = {10.1109/PerComWorkshops56833.2023.10150265},
  url = {https://doi.org/10.1109/PerComWorkshops56833.2023.10150265}
}
```

## Source terms

No software license was present in the recovered repository. Attribution is preserved; reuse terms remain unspecified. Dataset and external-model terms remain with their providers.
