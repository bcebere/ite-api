# third party
import numpy as np
import tensorflow as tf


def PEHE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Precision in Estimation of Heterogeneous Effect(Tensorflow version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return tf.reduce_mean(
        tf.math.squared_difference((y[:, 1] - y[:, 0]), (hat_y[:, 1] - hat_y[:, 0]))
    )


def ATE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return tf.abs(
        tf.reduce_mean(y[:, 1] - y[:, 0]) - tf.reduce_mean(hat_y[:, 1] - hat_y[:, 0])
    )


def RPol(t: tf.Variable, y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Policy risk(RPol).
    RPol is the average loss in value when treating according to the policy implied by an ITE estimator.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    Output:

    """
    hat_t = np.sign(hat_y[:, 1] - hat_y[:, 0])
    hat_t = 0.5 * (hat_t + 1)
    new_hat_t = np.abs(1 - hat_t)

    # Intersection
    idx1 = hat_t * t
    idx0 = new_hat_t * (1 - t)

    # risk policy computation
    RPol1 = (np.sum(idx1 * y) / (np.sum(idx1) + 1e-8)) * np.mean(hat_t)
    RPol0 = (np.sum(idx0 * y) / (np.sum(idx0) + 1e-8)) * np.mean(new_hat_t)

    return 1 - (RPol1 + RPol0)


def ATT(t: tf.Variable, y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Average Treatment Effect on the Treated(ATT).
    ATT measures what is the expected causal effect of the treatment for individuals in the treatment group.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    """
    # Original ATT
    ATT_value = np.sum(t * y) / (np.sum(t) + 1e-8) - np.sum((1 - t) * y) / (
        np.sum(1 - t) + 1e-8
    )
    # Estimated ATT
    ATT_estimate = np.sum(t * (hat_y[:, 1] - hat_y[:, 0])) / (np.sum(t) + 1e-8)
    return np.abs(ATT_value - ATT_estimate)


def xavier_init(size: tf.Variable) -> tf.Variable:
    """
    Xavier Weight initialization strategy.
    Xavier Initialization initializes the weights in the network by drawing them from a distribution
    with zero mean and a specific variance.
    Args:
        size: Shape of the tensor.
    """
    in_dim = tf.cast(size[0], tf.float32)
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random.normal(shape=size, stddev=xavier_stddev)
