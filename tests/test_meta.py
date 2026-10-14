import numpy as np
import pytest
import tensorflow as tf
from damldg.method import domain_loss, energy_gap, meta_gradients, train_step
from damldg.model import build_model
from damldg.data import SCENARIOS, read_audio_manifest, make_batches, features


def toy():
    model = tf.keras.Sequential(
        [tf.keras.Input((2,)), tf.keras.layers.Dense(2, use_bias=False)]
    )
    model.set_weights([np.array([[0.3, -0.2], [0.7, 0.1]], np.float32)])
    return model


def batch():
    return (
        tf.constant([[0.2, 0.7], [1.0, -0.5], [1.3, 0.1], [-0.3, 0.8]], tf.float32),
        tf.constant([0, 1, 1, 0]),
        tf.constant([0, 0, 1, 1]),
    )


def test_domain_mean_and_energy_definition():
    logits = tf.constant([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    labels = tf.constant([0, 0, 0])
    domains = tf.constant([0, 0, 1])
    expected = (np.log(2) + np.log1p(np.exp(-2))) / 2
    assert float(domain_loss(logits, labels, domains)) == pytest.approx(expected)
    assert float(energy_gap([[0.0, 0.0]], [[2.0, 2.0]])) == pytest.approx(2.0)


@pytest.mark.parametrize("method", ["mldg", "damldg"])
def test_full_meta_gradient_matches_finite_difference(method):
    model = toy()
    x, y, d = batch()
    base = model.get_weights()[0].copy()
    g, _, _ = meta_gradients(
        model, x, y, d, 1, inner_lr=0.3, method=method, training=False
    )
    numeric = np.zeros_like(base)
    epsilon = 0.002
    for index in np.ndindex(base.shape):
        plus = base.copy()
        plus[index] += epsilon
        model.set_weights([plus])
        _, _, a = meta_gradients(
            model, x, y, d, 1, inner_lr=0.3, method=method, training=False
        )
        minus = base.copy()
        minus[index] -= epsilon
        model.set_weights([minus])
        _, _, b = meta_gradients(
            model, x, y, d, 1, inner_lr=0.3, method=method, training=False
        )
        numeric[index] = (float(a["loss"]) - float(b["loss"])) / (2 * epsilon)
    model.set_weights([base])
    np.testing.assert_allclose(g[0], numeric, rtol=0.006, atol=3e-4)


def test_meta_test_labels_and_second_order_term_change_update():
    model = toy()
    x, y, d = batch()
    full, _, _ = meta_gradients(
        model, x, y, d, 1, inner_lr=0.3, method="mldg", training=False
    )
    changed, _, _ = meta_gradients(
        model,
        x,
        tf.constant([0, 1, 0, 1]),
        d,
        1,
        inner_lr=0.3,
        method="mldg",
        training=False,
    )
    first, _, _ = meta_gradients(
        model, x, y, d, 1, inner_lr=0.3, method="mldg", training=False, first_order=True
    )
    assert np.max(np.abs(full[0] - changed[0])) > 0.01
    assert np.max(np.abs(full[0] - first[0])) > 0.001
    assert callable(model.trainable_variables[0].assign)
    before = model.get_weights()[0].copy()
    train_step(model, tf.keras.optimizers.SGD(0.05), x, y, d, 1, 0.3, "mldg")
    np.testing.assert_allclose(
        model.get_weights()[0], before - 0.05 * full[0], atol=1e-6
    )


def test_meta_calls_do_not_mutate_parameters_or_batchnorm_state():
    model = tf.keras.Sequential(
        [
            tf.keras.Input((2,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2),
        ]
    )
    x, y, d = batch()
    before = [v.numpy().copy() for v in model.variables]
    gradients, state, details = meta_gradients(model, x, y, d, 1, 0.01, "damldg")
    for a, b in zip(before, model.variables):
        np.testing.assert_array_equal(a, b)
    assert all(g is not None and np.isfinite(g).all() for g in gradients)
    assert any(
        not np.array_equal(a, b) for a, b in zip(state, model.non_trainable_variables)
    )


def test_reject_one_domain():
    model = toy()
    x, y, d = batch()
    with pytest.raises(ValueError):
        meta_gradients(model, x, y, tf.zeros_like(d), 0, 0.01, "damldg")


def test_paper_scenario5_and_target_separation(tmp_path):
    assert SCENARIOS["scn5"] == (["emovo", "shemo", "emodb"], "aesdd")
    for domain in ["emovo", "shemo", "emodb", "aesdd"]:
        p = tmp_path / domain / "anger" / "a.wav"
        p.parent.mkdir(parents=True)
        tf.io.write_file(str(p), tf.audio.encode_wav(tf.zeros((32000, 1)), 16000))
    rows = read_audio_manifest(tmp_path, "scn5")
    sources = [r for r in rows if r["split"] == "source"]
    targets = [r for r in rows if r["split"] == "target"]
    assert {r["domain"] for r in sources} == {"emovo", "shemo", "emodb"}
    assert {r["domain"] for r in targets} == {"aesdd"}
    assert not ({r["path"] for r in sources} & {r["path"] for r in targets})
    batches = make_batches(sources, 6, np.random.default_rng(1))
    assert all({r["domain"] for r in b} == {"emovo", "shemo", "emodb"} for b in batches)
    assert all(r["split"] == "source" for b in batches for r in b)


def test_audio_feature_shape_and_seeded_augmentation():
    wave = np.sin(np.arange(48000) * 0.03).astype(np.float32)
    a = features(wave, True, np.random.default_rng(4))
    b = features(wave, True, np.random.default_rng(4))
    assert a.shape == (198, 64, 1)
    np.testing.assert_array_equal(a, b)


def test_full_resnet_meta_step_and_serialization(tmp_path):
    tf.keras.utils.set_random_seed(4)
    model = build_model((32, 32, 1))
    x = tf.random.normal((4, 32, 32, 1))
    y = tf.constant([0, 1, 2, 3])
    d = tf.constant([0, 0, 1, 1])
    before = model.get_weights()[0].copy()
    result = train_step(
        model, tf.keras.optimizers.SGD(0.001), x, y, d, 1, 0.01, "damldg"
    )
    assert all(np.isfinite(float(v)) for v in result.values())
    assert not np.array_equal(model.get_weights()[0], before)
    model.save(tmp_path / "model.keras")
    loaded = tf.keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(
        model(x, training=False), loaded(x, training=False), atol=1e-6
    )


def test_real_wav_pipeline_and_strict_encoder_transfer(tmp_path):
    from dataclasses import replace
    from damldg.train import Config, run
    import json

    for domain in ["tess", "emodb", "shemo"]:
        for label in ["anger", "happiness"]:
            p = tmp_path / "audio" / domain / label / "a.wav"
            p.parent.mkdir(parents=True)
            wave = tf.sin(
                tf.range(16000, dtype=tf.float32) * (0.05 if label == "anger" else 0.08)
            )[:, None]
            tf.io.write_file(str(p), tf.audio.encode_wav(wave, 16000))
    rows = read_audio_manifest(tmp_path / "audio", "scn7")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "classes": ["anger", "happiness", "sadness", "disgust", "fear"],
                "representation": "audio",
                "records": rows,
            }
        )
    )
    config = Config(
        dataset="manifest",
        manifest=str(manifest),
        method="damldg",
        epochs=1,
        batch_size=4,
        steps_per_epoch=1,
        learning_rate=0.001,
        augment=False,
        output=str(tmp_path / "run"),
    )
    result = run(config)
    assert result["target"]["num_examples"] == 2 and np.isfinite(
        result["target"]["loss"]
    )
    loaded = build_model(num_classes=9, encoder_path=tmp_path / "run/encoder.keras")
    encoder = tf.keras.models.load_model(tmp_path / "run/encoder.keras")
    for a, b in zip(loaded.get_layer("encoder").get_weights(), encoder.get_weights()):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        build_model((None, None, 3), encoder_path=tmp_path / "run/encoder.keras")


def test_rsc_baseline_keeps_challenged_inputs_and_labels_aligned():
    from damldg.method import rsc_step

    inputs = tf.keras.Input((2, 2, 1))
    encoder = tf.keras.Model(
        inputs, tf.keras.layers.Conv2D(2, 1)(inputs), name="encoder"
    )
    head = tf.keras.Sequential(
        [tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(2)],
        name="head",
    )
    model = tf.keras.Model(inputs, head(encoder(inputs)))
    observed = []
    original = encoder.call

    def capture(x, *args, **kwargs):
        if tf.executing_eagerly():
            observed.append(np.asarray(x))
        return original(x, *args, **kwargs)

    encoder.call = capture
    x = tf.reshape(tf.range(24, dtype=tf.float32), (6, 2, 2, 1))
    y = tf.constant([1, 0, 1, 0, 1, 0])
    metrics = rsc_step(
        model,
        tf.keras.optimizers.SGD(0.001),
        x,
        y,
        np.random.default_rng(1),
        batch_fraction=0.5,
    )
    np.testing.assert_array_equal(observed[-1], x[:3])
    assert all(np.isfinite(v) for v in metrics.values())
