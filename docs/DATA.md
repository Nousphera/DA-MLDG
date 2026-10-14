# Data and pretraining

## Speech emotion corpora

Obtain each dataset under its access terms and prepare WAV/FLAC files in:

```text
datasets/ser/<corpus>/<emotion>/<utterance>.wav
```

Corpus IDs are `crema_d`, `savee`, `ravdess`, `iemocap`, `tess`, `emodb`, `cafe`, `emovo`, `shemo`, and `aesdd`. The common class order, retained from source, is `anger,happiness,sadness,disgust,fear`. Other emotion directories are excluded. A corpus may legitimately miss a common class (ShEMO has no disgust samples); no samples are fabricated.

The original repository assumes already normalized emotion directories; raw dataset annotation converters and original prepared split manifests are absent. Create these directories from the providers' annotations, preserve utterance/speaker IDs, record the mapping and checksums, and document decisions such as IEMOCAP happy/excited merging. Do not infer labels from an arbitrary filename rule. This release validates normalized directories but cannot reconstruct undisclosed annotation decisions.

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

Preparation fails if a selected domain has no recognized audio or a resolved file path appears twice. Manifests record source/target assignments explicitly; training validates disjoint domain identities. All source corpora are available during training, while the target is used only for final evaluation. This is not the separate speaker-independent 80/20 within-corpus experiment in Table I; that split recipe is not reconstructed here.

SoundFile reads audio, channels are averaged, and SciPy polyphase filtering resamples to 16 kHz. Training augmentation/crops are seeded. Hold-out target audio gets no augmentation, and labels never enter parameter updates. File paths in manifests are absolute and local; keep manifests and outputs private if their metadata is sensitive.

## Speaker pretraining and encoder transfer

Use a separately obtained VoxCeleb corpus arranged as `datasets/voxceleb/<speaker>/<session>/<clip>.wav`:

```bash
damldg prepare --kind speaker --root datasets/voxceleb --output data/voxceleb.json
damldg train --config configs/pretrain.json
```

The reconstruction trains ResNet-18 to classify speakers for 50 epochs, SGD0.01/momentum 0.9, batch 64, two-second crops, 25 ms Hamming windows and 10 ms hops. It uses log-magnitude STFT with FFT1024; that FFT and logarithm choice are explicit because the original pretraining implementation is missing. `encoder.keras` excludes the speaker head and transfers into downstream SER via `pretrained_encoder`. The directory labels determine the speaker count; the paper describes 1,251 speakers.

A downstream config with `pretrained_encoder: null` uses random initialization and is not the paper's pretrained setup. Encoder weight loading is strict; incompatible input channels/layers raise an error rather than silently skipping weights. No pretrained checkpoint is automatically downloaded or bundled. Record encoder provenance and the input dataset version.

## PACS

Arrange the locally obtained PACS images as `<root>/<domain>/<class>/*.{jpg,jpeg,png}`. Domains: `art_painting,cartoon,photo,sketch`; classes: `dog,elephant,giraffe,guitar,horse,house,person`.

```bash
damldg prepare --kind image --root datasets/pacs --target art_painting --output data/pacs.json
damldg train --config configs/pacs.json
```

Images are resized to 256×256 then randomly cropped to 224×224 and flipped during training; evaluation uses a center crop. RGB means 123.68/116.779/103.939 are subtracted, as in source. A compatible ImageNet-pretrained encoder is required for the paper's setup; the missing original checkpoint is not replaced with random weights without recording the difference. Repeat with each held-out domain and a distinct manifest/output to perform leave-one-domain-out evaluation.

## Prepared feature inputs / TRILLsson boundary

Use `--kind features` to index `.npy` files with the same corpus/emotion layout. Files must contain finite arrays shaped `[H,W,C]` (or `[H,W]`, adding one channel). Set `representation: features` and point the config at that manifest. Loading uses `allow_pickle=False`. Arrays in a training batch must share a shape.

This adapter permits experiments with prepared representations, but it is not a recovered TRILLsson-v3 pipeline. The paper names EfficientNetv2-B3 TRILLsson embeddings without specifying the reshape/projection needed by this ResNet-18; the original source has no extractor. Record the external model/version, extraction settings and tensor mapping in the manifest if using that experiment. Do not label arbitrary feature arrays as a reproduction of Table IV.
