# Method and configuration

DA-MLDG implements domain-shift aware meta-learning for speech emotion
recognition. See [King Gandhi, Tsouvalas and Meratnia, PerCom Workshops 2023](https://doi.org/10.1109/PerComWorkshops56833.2023.10150265).

## Model and objective

The ResNet-18 backbone uses a 7×7 stride-2 stem, max pooling, four two-block
residual stages with 64/128/256/512 channels, batch normalization, global average
pooling and a classifier. Encoder-only transfer loads weights strictly and
supports a new downstream classifier.

Each training step holds out one complete source domain as the meta-test
partition. Other source domains form the meta-training partition. The final
target domain is excluded from both.

```text
L_train = mean of per-domain mean cross-entropy
θ′ = θ - inner_lr * gradient(L_train)
gamma = abs(mean(logsumexp(meta_train_logits(θ′)))
            - mean(logsumexp(meta_test_logits(θ′))))
L_outer = L_train + gamma * L_test(θ′)
```

The energy gap uses temperature 1, with no added offset or clipping. Both the
adaptation Hessian and gamma's parameter dependence are differentiated.
`mldg` fixes gamma to 1; `damldg` uses the energy gap. CLI training computes full
second-order gradients with Keras stateless tensor updates. The test-only
`first_order` option isolates the second-order contribution.

## Batch normalization

[Keras stateless calls](https://keras.io/keras_3/) pass trainable and non-trainable
tensors without mutating the base model. Meta-training uses batch statistics.
Its updated moving statistics are detached and passed to the adapted model;
energy evaluation uses inference mode and meta-test CE uses its own batch
statistics. After the outer update, only meta-training moving statistics are
committed. Temporary meta-test state is discarded, and final-target statistics
never affect training.

This implementation differentiates gamma and treats BN state as state rather
than a meta-learned parameter. These choices specify details beyond the paper's
written parameter objective. Finite-difference tests check the parameter
objective separately from BN state-isolation tests.

## Batching and optimization

Source batches are domain-balanced. Smaller domains cycle with reshuffling
until the largest domain is covered. Meta-test domains are selected uniformly
per step. All four methods share this sampler.

Defaults are outer SGD learning rate 0.05, momentum 0.9, inner rate 0.01 and batch
size 32. Scenario configs start at 50 epochs. `deepall` pools source examples.
`rsc` uses two-stage ordinary/challenged updates, percentile-66 feature
suppression, 33% challenged examples and an additional 512-unit head.

## Inputs

Speech is mono 16 kHz with random two-second training crops. Features use a
400-sample Hamming window, hop 160, FFT 1024, 64 mel bins from 60–7800 Hz and
`log(mel+1e-6)`. Evaluation processes the complete utterance, padded to two
seconds when shorter. Augmentation applies pitch-preserving time stretch at a
uniform rate in `[0.8,1.25]`, then a zero-filled shift within ±10% of two seconds.
The interpolation, operation order and zero-fill policy are explicit settings.

The seven scenario configs specify source and target corpora; scenario 5 uses
EMOVO/ShEMO/EmoDB as sources and AESDD as target. Speaker pretraining uses
ResNet-18 with speaker labels and exports `encoder.keras`. Record the selected
VoxCeleb release: the paper describes 1,251 speakers while citing VoxCeleb2.

Prepared feature arrays can be supplied to the same trainer. Embedding
extraction, model version and tensor mapping are external inputs, including for
TRILLsson-based experiments. See the [data guide](DATA.md) for the required
layouts and the [experiment guide](REPRODUCIBILITY.md) for initialization,
seed control and outputs.
