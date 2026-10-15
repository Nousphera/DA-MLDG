# Pretrained encoders

Use the [speaker pretraining recipe](../../../docs/DATA.md#speaker-pretraining-and-encoder-transfer)
to train on an acquired [VoxCeleb corpus](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
The trainer writes `encoder.keras` in the run output directory. Set
`pretrained_encoder` in a downstream configuration to that file.
The [paper](../../../report.pdf) describes the experimental setup.
