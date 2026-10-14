# Data and pretraining

## Speech emotion corpora

Obtain each dataset under its access terms and prepare WAV/FLAC files in:

```text
datasets/ser/<corpus>/<emotion>/<utterance>.wav
```

Corpus IDs are `crema_d`, `savee`, `ravdess`, `iemocap`, `tess`, `emodb`, `cafe`, `emovo`, `shemo`, and `aesdd`. The common class order, retained from source, is `anger,happiness,sadness,disgust,fear`. Other emotion directories are excluded. A corpus may legitimately miss a common class (ShEMO has no disgust samples); no samples are fabricated.

### Acquire and import raw corpora

Use the provider releases cited by the [paper](https://doi.org/10.1109/PerComWorkshops56833.2023.10150265).
Access terms and annotation formats differ; the importer accepts an explicit
CSV rather than guessing emotion labels from arbitrary filenames.

| Corpus ID | Provider/resource | Input annotation requirement |
|---|---|---|
| `crema_d` | [CREMA-D authors](https://github.com/CheyneyComputerScience/CREMA-D) | Use released audio and actor/utterance emotion annotations |
| `savee` | [SAVEE registration](https://cvssp.org/savee/Register.html) | Obtain licensed audio and emotion labels |
| `ravdess` | [RAVDESS authors](https://smartlaboratory.org/resources/speech-song-database-ravdess/) | Select audio-only speech, not song; follow release emotion codes |
| `iemocap` | [IEMOCAP access](https://sail.usc.edu/iemocap/) | Use utterance annotations; explicitly record happy/excited policy |
| `tess` | [TESS authors' dataset](https://doi.org/10.5683/SP2/E8H2MF) | Use provided emotion labels and retain both speakers |
| `emodb` | [TU Berlin EmoDB](https://www.tu.berlin/en/kw/research/projects/emotional-speech) | Obtain recordings and annotations from the same release |
| `cafe` | [CaFE dataset](https://doi.org/10.5281/zenodo.1219621) | Use released Canadian French emotion labels |
| `emovo` | [EMOVO publication](https://aclanthology.org/L14-1478/) | Obtain author corpus and documented Italian emotion labels |
| `shemo` | [ShEMO authors' paper](https://arxiv.org/abs/1906.01155) | Obtain author corpus/annotations; disgust is absent |
| `aesdd` | [AESDD authors' paper](https://secure.aes.org/forum/pubs/journal/?elib=19585) | Obtain author corpus and five emotion labels |

For each corpus, create a UTF-8 CSV from its provider labels with the columns
below. Paths are relative to the extracted raw corpus. `emotion` must be one of
the five canonical names; omit neutral, surprise and other excluded categories.
Preserve and document the provider-to-canonical mapping alongside the CSV.
This annotation conversion is a required user input because the historical
prepared files and harmonization choices were not recovered.

```csv
path,emotion
speaker01/session01/utterance1.wav,anger
speaker01/session01/utterance2.wav,happiness
```

For any corpus ID in the table:

```bash
damldg import-corpus --corpus tess --root /path/to/TESS \
  --annotations /path/to/tess-labels.csv --output datasets/ser/tess
```

Repeat for the other source/target corpora in the selected scenario. The importer
decodes and hashes every audio file, rejects invalid labels, duplicate paths and
paths outside the raw root, preserves relative speaker/session paths under each
emotion, and atomically publishes the copied directory plus `preparation.json`.
It refuses an existing destination and never edits source files. Already
normalized folders can be indexed directly by `damldg prepare`.

The original source expects normalized emotion directories. This importer makes
that boundary executable without claiming to recover its missing original label
harmonization or benchmark partitions.

| Scenario | Source corpora | Target |
|---|---|---|
| scn1 | CREMA-D, SAVEE, RAVDESS | IEMOCAP |
| scn2 | CREMA-D, SAVEE, TESS | RAVDESS |
| scn3 | CREMA-D, SAVEE, RAVDESS | TESS |
| scn4 | SAVEE, RAVDESS, EmoDB | CaFE |
| scn5 | EMOVO, ShEMO, EmoDB | AESDD |
| scn6 | CREMA-D, SAVEE, RAVDESS | ShEMO |
| scn7 | TESS, EmoDB | ShEMO |

```bash
damldg prepare --root datasets/ser --scenario scn5 --output data/scn5.json
damldg train --config configs/scn5.json
```

Preparation decodes each input, records SHA-256 and byte counts, and fails if a selected domain has no recognized audio, a file escapes the root or a resolved path appears twice. `damldg validate --manifest data/scn5.json` repeats decoding and hash checks without rewriting the manifest. Manifests record source/target assignments explicitly; training validates disjoint domain identities. All source corpora are available during training, while the target is used only for final evaluation. This is not the separate speaker-independent 80/20 within-corpus experiment in Table I; that split recipe is not reconstructed here.

SoundFile reads audio, channels are averaged, and SciPy polyphase filtering resamples to 16 kHz. Training augmentation/crops are seeded. Hold-out target audio gets no augmentation, and labels never enter parameter updates. File paths in manifests are absolute and local; keep manifests and outputs private if their metadata is sensitive.

## Speaker pretraining and encoder transfer

Use a separately obtained [VoxCeleb corpus](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) arranged as `datasets/voxceleb/<speaker>/<session>/<clip>.wav`:

```bash
damldg prepare --kind speaker --root datasets/voxceleb --output data/voxceleb.json
damldg validate --manifest data/voxceleb.json
damldg train --config configs/pretrain.json
```

The reconstruction trains ResNet-18 to classify speakers for 50 epochs, SGD0.01/momentum 0.9, batch 64, two-second crops, 25 ms Hamming windows and 10 ms hops. It uses log-magnitude STFT with FFT1024; that FFT and logarithm choice are explicit because the original pretraining implementation is missing. `encoder.keras` excludes the speaker head and transfers into downstream SER via `pretrained_encoder`. The directory labels determine the speaker count; the paper describes 1,251 speakers.

A downstream config with `pretrained_encoder: null` uses random initialization and is not the paper's pretrained setup. Encoder weight loading is strict; incompatible input channels/layers raise an error rather than silently skipping weights. No pretrained checkpoint is automatically downloaded or bundled. Record encoder provenance and the input dataset version.

## PACS

Obtain PACS through the [authors' project](https://domaingeneralization.github.io/) and arrange the images as `<root>/<domain>/<class>/*.{jpg,jpeg,png}`. Domains: `art_painting,cartoon,photo,sketch`; classes: `dog,elephant,giraffe,guitar,horse,house,person`.

```bash
damldg prepare --kind image --root datasets/pacs --target art_painting --output data/pacs.json
damldg validate --manifest data/pacs.json
damldg train --config configs/pacs.json
```

Images are resized to 256×256 then randomly cropped to 224×224 and flipped during training; evaluation uses a center crop. RGB means 123.68/116.779/103.939 are subtracted, as in source. A compatible ImageNet-pretrained encoder is required for the paper's setup; the missing original checkpoint is not replaced with random weights without recording the difference. Repeat with each held-out domain and a distinct manifest/output to perform leave-one-domain-out evaluation.

## Prepared feature inputs / TRILLsson boundary

Use `--kind features` to index `.npy` files with the same corpus/emotion layout. Files must contain finite arrays shaped `[H,W,C]` (or `[H,W]`, adding one channel). Set `representation: features` and point the config at that manifest. Loading uses `allow_pickle=False`. Arrays in a training batch must share a shape.

This adapter permits experiments with prepared representations, but it is not a recovered TRILLsson-v3 pipeline. The paper names EfficientNetv2-B3 TRILLsson embeddings without specifying the reshape/projection needed by this ResNet-18; the original source has no extractor. Record the external model/version, extraction settings and tensor mapping in the manifest if using that experiment. Do not label arbitrary feature arrays as a reproduction of Table IV.


To index prepared representations for scenario 7:

```bash
damldg prepare --kind features --root /path/to/features --scenario scn7 --output data/scn7-features.json
damldg validate --manifest data/scn7-features.json
```

Copy `configs/scn7.json`, set `representation` to `features`, set `manifest` to
this file, and choose a new output directory. This validates finite array content
without pickle loading. The extraction model and tensor mapping are external,
explicit inputs; no fabricated TRILLsson extractor is included.

Paths in shipped files under `configs/` resolve against the checkout. Paths in
configs saved elsewhere resolve against the config's own directory. Keep each
manifest's input files available at the recorded absolute paths. Import,
preparation and validation do not execute training or download restricted data.
