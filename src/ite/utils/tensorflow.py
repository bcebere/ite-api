# stdlib
from typing import List

# third party
import numpy as np
import tensorflow as tf


def PEHE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    e_PEHE = tf.reduce_mean(
        tf.squared_difference((y[:, 1] - y[:, 0]), (hat_y[:, 1] - hat_y[:, 0]))
    )
    return e_PEHE


def ATE(y: tf.Variable, hat_y: tf.Variable) -> tf.Variable:
    e_PEHE = tf.abs(
        tf.reduce_mean(y[:, 1] - y[:, 0]) - tf.reduce_mean(hat_y[:, 1] - hat_y[:, 0])
    )
    return e_PEHE


def xavier_init(size: tf.Variable) -> tf.Variable:
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Performance Metrics
def Perf_RPol_ATT(
    Test_T: tf.Variable, Test_Y: tf.Variable, Output_Y: tf.Variable
) -> List[tf.Variable]:
    # RPol
    # Decision of Output_Y
    hat_t = np.sign(Output_Y[:, 1] - Output_Y[:, 0])
    hat_t = 0.5 * (hat_t + 1)
    new_hat_t = np.abs(1 - hat_t)

    # Intersection
    idx1 = hat_t * Test_T
    idx0 = new_hat_t * (1 - Test_T)

    # RPol Computation
    RPol1 = (np.sum(idx1 * Test_Y) / (np.sum(idx1) + 1e-8)) * np.mean(hat_t)
    RPol0 = (np.sum(idx0 * Test_Y) / (np.sum(idx0) + 1e-8)) * np.mean(new_hat_t)
    RPol = 1 - (RPol1 + RPol0)

    # ATT
    # Original ATT
    ATT_value = np.sum(Test_T * Test_Y) / (np.sum(Test_T) + 1e-8) - np.sum(
        (1 - Test_T) * Test_Y
    ) / (np.sum(1 - Test_T) + 1e-8)
    # Estimated ATT
    ATT_estimate = np.sum(Test_T * (Output_Y[:, 1] - Output_Y[:, 0])) / (
        np.sum(Test_T) + 1e-8
    )
    # Final ATT
    ATT = np.abs(ATT_value - ATT_estimate)
    print(
        "pol0:{} pol1:{} pol:{} mean hat:{} mean new hat:{} ATT:{}".format(
            RPol0, RPol1, RPol, np.mean(hat_t), np.mean(new_hat_t), ATT
        )
    )
    return [RPol, ATT]
