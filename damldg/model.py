"""Original ResNet-18 backbone with explicit, strict encoder transfer."""

import tensorflow as tf
from .backbone import ResNet18


def build_model(
    input_shape=(None, None, 1), num_classes=5, encoder_path=None, rsc=False
):
    encoder = ResNet18.model(
        input_shape=input_shape, num_classes=num_classes, use_rsc=True
    )
    # A stable nested model boundary enables classifier-independent pretraining.
    encoder.name = "encoder"
    if encoder_path:
        pretrained = tf.keras.models.load_model(encoder_path)
        encoder.set_weights(pretrained.get_weights())
    layers = [tf.keras.layers.GlobalAveragePooling2D()]
    if rsc:
        layers.append(tf.keras.layers.Dense(512))
    layers.append(tf.keras.layers.Dense(num_classes))
    head = tf.keras.Sequential(layers, name="head")
    inputs = tf.keras.Input(input_shape, name="features")
    return tf.keras.Model(inputs, head(encoder(inputs)), name="damldg_resnet18")
