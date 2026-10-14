# Paper-to-code mapping

Reference: King Gandhi, Tsouvalas, Meratnia, *On efficacy of Meta-Learning for Domain Generalization in Speech Emotion Recognition*, PerFail / IEEE PerCom Workshops 2023, pp.421–426, [DOI](https://doi.org/10.1109/PerComWorkshops56833.2023.10150265). The source is the final six-page article plus repository cover. The paper itself emphasizes inadequate practical generalization and negative results.

| Component | Maintained code | Exact behavior |
|---|---|---|
| ResNet-18 | `backbone.py`, `model.py` | Original architecture: 7×7 stride2 stem, max pool, four two-block residual stages with 64/128/256/512 channels, batch normalization, global average pooling and classifier. Strict encoder-only transfer; no silent weight mismatch skipping. |
| Eq. 2 inner loss | `method.py:domain_loss` | Mean CE within each source domain, then equal mean across domains. Original source instead pooled examples regardless of domain size. |
| Algorithm 1 inner update | `method.py:meta_gradients` | `theta_prime = theta - inner_lr * grad(L_train)`. Adapted tensors stay in the outer tape and are passed to `model.stateless_call`. No assignments or disconnected temporary network. |
| Eq. 3 domain-shift margin | `method.py:energy_gap` | Absolute difference of mean `logsumexp(logits)` for meta-train and meta-test samples using adapted parameters; T=1. No extra minus sign, temperature factor, clipping, or added offset. |
| Eq. 2 outer objective | `method.py:meta_gradients,train_step` | `L_train + gamma * L_test(theta_prime)`. Both the adaptation Hessian and gamma's parameter dependence are differentiated. MLDG sets gamma=1. The diagnostic `first_order` argument exists only for tests; CLI training uses full second order. |
| Source-domain split | `data.py:make_batches`, `train.py:run` | Domain-balanced batches; uniformly choose one complete source domain as meta-test each step. Other source domains are meta-train. Final target domain never enters either subset. |
| Paper scenarios | `data.py:SCENARIOS` | All seven Table I scenarios, including scenario 5 target AESDD and source EMOVO/ShEMO/EmoDB. Source `src/utils.py` instead targets EmoDB in scenario 5. |
| Speech inputs | `data.py:features` | 16kHz mono, two-second random training crop, STFT400/hop160/FFT1024 with Hamming window, 64 mel bins 60–7800Hz, `log(mel+1e-6)`. Evaluation uses the whole utterance, as source does, padding to two seconds if shorter. |
| Augmentation | `data.py:features` | Pitch-preserving librosa time-stretch with uniform rate[0.8,1.25], then zero-filled time shift uniform±10% of two seconds. Parameters come from paper; interpolation, operation order, zero fill, and applying both each training example are explicit choices because the source omitted augmentation. |
| Optimization | `train.py` | SGD outer learning rate 0.05, momentum 0.9; inner rate 0.01. No source exponential decay (not specified by paper). Batch 32; configs with 50 epochs are an explicit experiment starting point, following historical launcher duration. |
| Deep-All / RSC | `method.py:train_step,rsc_step` | Deep-All pools source examples. RSC ports the source's two-stage normal/challenged update, with percentile 66 feature suppression and 33% challenged examples; corrected feature/label alignment. Its source-specific extra 512-unit head is retained and disclosed. |

## Batch normalization and differentiability

[Keras stateless calls](https://keras.io/keras_3/) take trainable and non-trainable tensors and return new non-trainable state without mutating the model. The meta-train call runs with training-mode batch statistics. Its updated moving statistics are passed, detached, to the adapted model; energy uses inference mode, and meta-test CE uses its own training-mode batch statistics. After the outer optimizer step, only the meta-train moving statistics are committed to the base model. Temporary meta-test state is discarded. No final-target batch statistics are used for training.

The paper does not specify these state details or whether gamma is detached. The maintained default differentiates gamma, matching its placement in the written objective and original outer tape. BN state is treated as state, not a meta-learned parameter. The finite-difference tests use a model without BN to isolate the stated parameter objective; a separate BN test verifies state isolation and explicit commit behavior.

## Source defects and limits

The original `copy_model` and inner update set `variable.assign = tensor`, replacing a method rather than assigning a value. Even replacing this with `assign(...)` would disconnect the second-order parameter graph. The maintained implementation uses functional tensors instead. Tests verify numerical gradients for both methods, a nonzero Hessian contribution, meta-test-label influence on actual weights, and full ResNet execution.

Other repaired differences: original scenario 5 domain error; sample-size-biased meta-test selection; 3.5-second CLI crops versus paper 2 seconds; batch 128 versus 32; momentum 0.85 versus 0.9; missing augmentation/resampling; import-time required environment variable; absent output directories; legacy optimizer APIs. The previous launcher deleted shared `logs`/`checkpoint` folders; `run.py` now invokes only the maintained CLI and never cleans other runs. The source RSC loop overwrote `x` with its ordinary-training subset before selecting challenged inputs, pairing those inputs with another subset's labels; current RSC keeps original inputs and labels aligned.

Domain-balanced batching cycles smaller domains until the largest has been covered. It guarantees a valid meta split and implements equal domain emphasis, but changes the historical pooled-shuffle sampling trajectory. All baselines share the maintained sampler. The exact original seeds, data membership, BN choices and update behavior that produced the paper tables were not recovered, so these repairs cannot establish historical numerical equivalence.

The VoxCeleb pretraining path is a documented reconstruction of the paper's recipe, not original recovered pretraining code. The paper states 150,000 samples / 1,251 speakers (consistent with VoxCeleb1) while citing a VoxCeleb2 reference; the original weights filename says `voxceleb1`. The release does not resolve that historical ambiguity or ship weights. TRILLsson embedding extraction and its mapping into ResNet-18 are absent in source; arbitrary precomputed arrays are supported, without claiming the Table IV pipeline has been recovered.

This is centralized domain-generalization training, not federated learning. A Flower wrapper would change the task without implementing an additional method, so none is provided.
