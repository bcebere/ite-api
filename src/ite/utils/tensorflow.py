# third party
import tensorflow.compat.v1 as tf


def sqrt_PEHE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Precision in Estimation of Heterogeneous Effect(Tensorflow version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    y_cast = tf.cast(y, tf.float64)
    hat_y_cast = tf.cast(hat_y, tf.float64)
    return tf.sqrt(
        tf.reduce_mean(
            tf.math.squared_difference(
                (y_cast[:, 1] - y_cast[:, 0]), (hat_y_cast[:, 1] - hat_y_cast[:, 0])
            )
        )
    )


def ATE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    y_cast = tf.cast(y, tf.float64)
    hat_y_cast = tf.cast(hat_y, tf.float64)
    return tf.abs(
        tf.reduce_mean(y_cast[:, 1] - y_cast[:, 0])
        - tf.reduce_mean(hat_y_cast[:, 1] - hat_y_cast[:, 0])
    )


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
