"""
GANITE:
Jinsung Yoon 10/11/2017
"""
# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm

# ite absolute
from ite.utils.metrics import HistoricMetrics
from ite.utils.metrics import Metrics
import ite.utils.tensorflow as tf_utils

tf.disable_v2_behavior()


class CounterfactualGenerator:
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        # Generator Layer
        self.G_W1 = tf.Variable(
            tf_utils.xavier_init([(Dim + 2), DimHidden])
        )  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector (Z)
        self.G_b1 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.G_W2 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.G_W31 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.G_b31 = tf.Variable(
            tf.zeros(shape=[DimHidden])
        )  # Output: Estimated Potential Outcomes

        self.G_W32 = tf.Variable(tf_utils.xavier_init([DimHidden, 1]))
        self.G_b32 = tf.Variable(
            tf.zeros(shape=[1])
        )  # Output: Estimated Potential Outcomes

        self.G_W41 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.G_b41 = tf.Variable(
            tf.zeros(shape=[DimHidden])
        )  # Output: Estimated Potential Outcomes

        self.G_W42 = tf.Variable(tf_utils.xavier_init([DimHidden, 1]))
        self.G_b42 = tf.Variable(
            tf.zeros(shape=[1])
        )  # Output: Estimated Potential Outcomes

        self.theta_G = [
            self.G_W1,
            self.G_W2,
            self.G_W31,
            self.G_W32,
            self.G_W41,
            self.G_W42,
            self.G_b1,
            self.G_b2,
            self.G_b31,
            self.G_b32,
            self.G_b41,
            self.G_b42,
        ]

    def forward(self, x: tf.Variable, t: tf.Variable, y: tf.Variable) -> tf.Variable:
        inputs = tf.concat(axis=1, values=[x, t, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)

        G_h31 = tf.nn.relu(tf.matmul(G_h2, self.G_W31) + self.G_b31)
        G_prob1 = tf.matmul(G_h31, self.G_W32) + self.G_b32

        G_h41 = tf.nn.relu(tf.matmul(G_h2, self.G_W41) + self.G_b41)
        G_prob2 = tf.matmul(G_h41, self.G_W42) + self.G_b42

        G_prob = tf.nn.sigmoid(tf.concat(axis=1, values=[G_prob1, G_prob2]))

        return G_prob


class CounterfactualDiscriminator:
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        self.D_W1 = tf.Variable(
            tf_utils.xavier_init([(Dim + 2), DimHidden])
        )  # Inputs: X + Factual Outcomes + Estimated Counterfactual Outcomes
        self.D_b1 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.D_W2 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.D_W3 = tf.Variable(tf_utils.xavier_init([DimHidden, 1]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_b1,
            self.D_b2,
            self.D_b3,
        ]

    def forward(
        self, x: tf.Variable, t: tf.Variable, y: tf.Variable, hat_y: tf.Variable
    ) -> tf.Variable:
        # Factual & Counterfactual outcomes concatenate
        inp0 = (1.0 - t) * y + t * tf.reshape(hat_y[:, 0], [-1, 1])
        inp1 = t * y + (1.0 - t) * tf.reshape(hat_y[:, 1], [-1, 1])

        inputs = tf.concat(axis=1, values=[x, inp0, inp1])

        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3

        return D_logit


class InferenceNets:
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        self.I_W1 = tf.Variable(tf_utils.xavier_init([(Dim), DimHidden]))
        self.I_b1 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.I_W2 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.I_b2 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.I_W31 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.I_b31 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.I_W32 = tf.Variable(tf_utils.xavier_init([DimHidden, 1]))
        self.I_b32 = tf.Variable(tf.zeros(shape=[1]))

        self.I_W41 = tf.Variable(tf_utils.xavier_init([DimHidden, DimHidden]))
        self.I_b41 = tf.Variable(tf.zeros(shape=[DimHidden]))

        self.I_W42 = tf.Variable(tf_utils.xavier_init([DimHidden, 1]))
        self.I_b42 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_I = [
            self.I_W1,
            self.I_W2,
            self.I_W31,
            self.I_W32,
            self.I_W41,
            self.I_W42,
            self.I_b1,
            self.I_b2,
            self.I_b31,
            self.I_b32,
            self.I_b41,
            self.I_b42,
        ]

    def forward(self, x: tf.Variable) -> tf.Variable:
        I_h1 = tf.nn.relu(tf.matmul(x, self.I_W1) + self.I_b1)
        I_h2 = tf.nn.relu(tf.matmul(I_h1, self.I_W2) + self.I_b2)

        I_h31 = tf.nn.relu(tf.matmul(I_h2, self.I_W31) + self.I_b31)
        I_prob1 = tf.matmul(I_h31, self.I_W32) + self.I_b32

        I_h41 = tf.nn.relu(tf.matmul(I_h2, self.I_W41) + self.I_b41)
        I_prob2 = tf.matmul(I_h41, self.I_W42) + self.I_b42

        I_prob = tf.nn.sigmoid(tf.concat(axis=1, values=[I_prob1, I_prob2]))

        return I_prob


class Ganite:
    """
    The GANITE framework generates potential outcomes for a given feature vector x.
    It consists of 2 components:
     - The Counterfactual Generator block(generator + discriminator).
     - The ITE block(InferenceNets).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        dim_hidden: int = 8,
        alpha: float = 2,
        beta: float = 2,
        minibatch_size: int = 128,
        depth: int = 1,
        num_iterations: int = 10000,
        test_step: int = 200,
        num_discr_iterations: int = 10,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.num_iterations = num_iterations
        self.test_step = test_step
        self.num_discr_iterations = num_discr_iterations

        tf.reset_default_graph()

        # 1. Input
        # 1.1. Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, dim])
        # 1.2. Treatment (T)
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        # 1.3. Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        # 1.6. Test Outcome (Y_T) - Potential outcome
        self.Y_T = tf.placeholder(tf.float32, shape=[None, dim_out])

        # 2. layer construction
        # 2.1 Generator Layer
        self.counterfactual_generator = CounterfactualGenerator(dim, dim_hidden, depth)

        # 2.2 Discriminator
        self.counterfactual_discriminator = CounterfactualDiscriminator(
            dim, dim_hidden, depth
        )

        # 2.3 Inference Layer
        self.inference_nets = InferenceNets(dim, dim_hidden, depth)

        # Structure
        # 1. Generator
        self.Tilde = self.counterfactual_generator.forward(self.X, self.T, self.Y)
        # 2. Discriminator
        self.D_logit = self.counterfactual_discriminator.forward(
            self.X, self.T, self.Y, self.Tilde
        )
        # 3. Inference function
        self.Hat = self.inference_nets.forward(self.X)

        # Loss
        # 1. Discriminator loss
        self.D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.T, logits=self.D_logit)
        )

        # 2. Generator loss
        self.G_loss_GAN = -self.D_loss

        self.G_loss_R = tf.reduce_mean(
            tf.losses.mean_squared_error(
                self.Y,
                (
                    self.T * tf.reshape(self.Tilde[:, 1], [-1, 1])
                    + (1.0 - self.T) * tf.reshape(self.Tilde[:, 0], [-1, 1])
                ),
            )
        )

        self.G_loss = self.G_loss_R + self.alpha * self.G_loss_GAN

        # 4. Inference loss

        self.I_loss1 = tf.reduce_mean(
            tf.losses.mean_squared_error(
                (self.T) * self.Y
                + (1 - self.T) * tf.reshape(self.Tilde[:, 1], [-1, 1]),
                tf.reshape(self.Hat[:, 1], [-1, 1]),
            )
        )
        self.I_loss2 = tf.reduce_mean(
            tf.losses.mean_squared_error(
                (1 - self.T) * self.Y
                + (self.T) * tf.reshape(self.Tilde[:, 0], [-1, 1]),
                tf.reshape(self.Hat[:, 0], [-1, 1]),
            )
        )

        self.I_loss = self.I_loss1 + self.beta * self.I_loss2

        # Solver
        self.G_solver = tf.train.AdamOptimizer().minimize(
            self.G_loss, var_list=self.counterfactual_generator.theta_G
        )
        self.D_solver = tf.train.AdamOptimizer().minimize(
            self.D_loss, var_list=self.counterfactual_discriminator.theta_D
        )
        self.I_solver = tf.train.AdamOptimizer().minimize(
            self.I_loss, var_list=self.inference_nets.theta_I
        )

        # Sessions
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Metrics
        self.train_perf_metrics = HistoricMetrics()

    def sample_minibatch(
        self, X: tf.Variable, T: tf.Variable, Y: tf.Variable
    ) -> Tuple[tf.Variable, tf.Variable, tf.Variable]:
        idx_mb = np.random.randint(0, X.shape[0], self.minibatch_size)

        X_mb = X[idx_mb, :]
        T_mb = np.reshape(T[idx_mb], [self.minibatch_size, 1])
        Y_mb = np.reshape(Y[idx_mb], [self.minibatch_size, 1])

        return X_mb, T_mb, Y_mb

    def train(
        self,
        Train_X: pd.DataFrame,
        Train_T: pd.DataFrame,
        Train_Y: pd.DataFrame,
        Opt_Train_Y: pd.DataFrame,
        Test_X: pd.DataFrame,
        Test_Y: pd.DataFrame,
    ) -> HistoricMetrics:
        # Iterations
        # Train G and D first
        for it in tqdm(range(self.num_iterations)):
            for kk in range(self.num_discr_iterations):
                X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

                _, D_loss_curr = self.sess.run(
                    [self.D_solver, self.D_loss],
                    feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
                )

            X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

            _, G_loss_curr, Tilde_curr = self.sess.run(
                [self.G_solver, self.G_loss, self.Tilde],
                feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
            )

            # Testing
            if it % self.test_step == 0:
                metric_block = "Counterfactual Block"
                self.train_perf_metrics.add(
                    "Discriminator loss", D_loss_curr, metric_block
                )
                self.train_perf_metrics.add("Generator loss", G_loss_curr, metric_block)

        # Train I and ID
        for it in tqdm(range(self.num_iterations)):
            X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

            _, I_loss_curr = self.sess.run(
                [self.I_solver, self.I_loss],
                feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
            )

            # Testing
            if it % self.test_step == 0:
                metric_block = "ITE Block"
                self.train_perf_metrics.add("Loss", I_loss_curr, metric_block)

                metric_block = "ITE Block in-sample metrics"
                metrics_for_step = self.test(Train_X, Opt_Train_Y)

                self.train_perf_metrics.add(
                    "sqrt_PEHE", metrics_for_step.sqrt_PEHE(), metric_block
                )
                self.train_perf_metrics.add("ATE", metrics_for_step.ATE(), metric_block)

                metric_block = "ITE Block out-sample metrics"
                metrics_for_step = self.test(Test_X, Test_Y)

                self.train_perf_metrics.add(
                    "sqrt_PEHE", metrics_for_step.sqrt_PEHE(), metric_block
                )
                self.train_perf_metrics.add("ATE", metrics_for_step.ATE(), metric_block)

        return self.train_perf_metrics

    def train_metrics(self) -> HistoricMetrics:
        return self.train_perf_metrics

    def predict(self, Test_X: pd.DataFrame) -> pd.DataFrame:
        Hat_curr = self.sess.run([self.Hat], feed_dict={self.X: Test_X})[0]
        return pd.DataFrame(Hat_curr, columns=["y_hat_0", "y_hat_1"])

    def test(self, Test_X: pd.DataFrame, Test_Y: pd.DataFrame) -> Metrics:
        y_hat = self.predict(Test_X)

        return Metrics(y_hat.to_numpy(), Test_Y)
