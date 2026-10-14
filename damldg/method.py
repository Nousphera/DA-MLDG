"""Full second-order MLDG and energy-weighted DA-MLDG with stateless Keras."""

import numpy as np
import tensorflow as tf


def domain_loss(logits, labels, domains):
    """Equation 2: equal domain weight, mean cross entropy within a domain."""
    labels = tf.cast(labels, tf.int32)
    domains = tf.convert_to_tensor(domains)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    _, indices = tf.unique(domains)
    means = tf.math.unsorted_segment_mean(losses, indices, tf.reduce_max(indices) + 1)
    return tf.reduce_mean(means)


def energy_gap(train_logits, test_logits):
    """Equation 3, T=1; per-example means and no extra negative/T factor."""
    a = tf.reduce_mean(tf.reduce_logsumexp(train_logits, axis=-1))
    b = tf.reduce_mean(tf.reduce_logsumexp(test_logits, axis=-1))
    return tf.abs(a - b)


def meta_gradients(
    model,
    x,
    y,
    domains,
    meta_test_domain,
    inner_lr=0.01,
    method="damldg",
    training=True,
    first_order=False,
):
    """Return outer gradients and source-train BN state without mutating a model.

    The adapted tensors remain in the outer tape, so the gradient includes the
    inner Hessian term. Gamma is differentiated as written in the objective.
    BN moving statistics are explicit state, not meta-learned parameters.
    """
    if method not in {"mldg", "damldg"}:
        raise ValueError("Unknown meta method")
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    domains = tf.convert_to_tensor(domains)
    test_mask = domains == meta_test_domain
    if not bool(tf.reduce_any(test_mask)) or not bool(tf.reduce_any(~test_mask)):
        raise ValueError(
            "Meta-train and meta-test must contain disjoint nonempty source domains"
        )
    xt, yt, dt = (tf.boolean_mask(a, ~test_mask) for a in (x, y, domains))
    xv, yv, dv = (tf.boolean_mask(a, test_mask) for a in (x, y, domains))
    weights = [tf.convert_to_tensor(v) for v in model.trainable_variables]
    state = [tf.convert_to_tensor(v) for v in model.non_trainable_variables]
    with tf.GradientTape() as outer:
        outer.watch(weights)
        with tf.GradientTape() as inner:
            inner.watch(weights)
            logits, train_state = model.stateless_call(
                weights, state, xt, training=training
            )
            train_loss = domain_loss(logits, yt, dt)
        gradients = inner.gradient(train_loss, weights)
        if any(g is None for g in gradients):
            raise ValueError("Disconnected inner variable")
        if first_order:
            gradients = [tf.stop_gradient(g) for g in gradients]
        adapted = [w - inner_lr * g for w, g in zip(weights, gradients)]
        local_state = [tf.stop_gradient(v) for v in train_state]
        if method == "damldg":
            train_energy, _ = model.stateless_call(
                adapted, local_state, xt, training=False
            )
            test_energy, _ = model.stateless_call(
                adapted, local_state, xv, training=False
            )
            gamma = energy_gap(train_energy, test_energy)
        else:
            gamma = tf.constant(1.0, tf.float32)
        test_logits, _ = model.stateless_call(
            adapted, local_state, xv, training=training
        )
        test_loss = domain_loss(test_logits, yv, dv)
        loss = train_loss + gamma * test_loss
    outer_gradients = outer.gradient(loss, weights)
    if any(g is None for g in outer_gradients):
        raise ValueError("Disconnected outer variable")
    return (
        outer_gradients,
        train_state,
        {
            "loss": loss,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "gamma": gamma,
        },
    )


def train_step(
    model,
    optimizer,
    x,
    y,
    domains,
    meta_test_domain=None,
    inner_lr=0.01,
    method="damldg",
):
    if method in {"mldg", "damldg"}:
        gradients, state, details = meta_gradients(
            model, x, y, domains, meta_test_domain, inner_lr, method
        )
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite meta gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Persist source-train state only; target and temporary meta-test state are never committed.
        for variable, value in zip(model.non_trainable_variables, state):
            variable.assign(value)
    elif method == "deepall":
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        details = {
            "loss": loss,
            "train_loss": loss,
            "test_loss": tf.constant(0.0),
            "gamma": tf.constant(0.0),
        }
    else:
        raise ValueError("Unknown method")
    for value in details.values():
        tf.debugging.assert_all_finite(value, "Nonfinite objective")
    return {k: float(v) for k, v in details.items()}


def rsc_step(model, optimizer, x, y, rng, percentile=66.0, batch_fraction=0.33):
    """Port the source's two-stage RSC baseline, fixing its sample/label mismatch."""
    n = max(1, int(len(x) * batch_fraction))
    if len(x) < 2:
        raise ValueError("RSC needs >=2 examples")
    with tf.GradientTape() as tape:
        logits = model(x[n:], training=True)
        plain = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[n:], logits=logits)
        )
    optimizer.apply_gradients(
        zip(tape.gradient(plain, model.trainable_variables), model.trainable_variables)
    )
    encoder = model.get_layer("encoder")
    head = model.get_layer("head")
    with tf.GradientTape() as tape:
        z = encoder(x[:n], training=False)
        tape.watch(z)
        score = tf.reduce_sum(head(z) * tf.one_hot(y[:n], head.layers[-1].units))
    sensitivity = tape.gradient(score, z)
    if int(rng.integers(9)) <= 4:
        s = tf.reduce_mean(sensitivity, axis=-1).numpy()
        mask = (s < np.percentile(s, percentile, axis=(1, 2), keepdims=True))[..., None]
    else:
        s = tf.reduce_mean(sensitivity, axis=(1, 2)).numpy()
        mask = (s < np.percentile(s, percentile, axis=1, keepdims=True))[
            :, None, None, :
        ]
    with tf.GradientTape() as tape:
        logits = head(encoder(x[:n], training=True) * tf.cast(mask, tf.float32))
        challenged = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:n], logits=logits)
        )
    optimizer.apply_gradients(
        zip(
            tape.gradient(challenged, model.trainable_variables),
            model.trainable_variables,
        )
    )
    return {
        "loss": float((plain + challenged) / 2),
        "train_loss": float(plain),
        "test_loss": float(challenged),
        "gamma": 0.0,
    }
