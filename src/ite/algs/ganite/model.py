"""
GANITE:
Jinsung Yoon 10/11/2017
"""
# third party
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm

# ite absolute
import ite.utils.tensorflow as tf_utils

tf.disable_v2_behavior()

# TODO: use depth for GANs
# TODO: fix Test metrics


class CounterfactualGenerator:
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


def sample_X(X: tf.Variable, size: int) -> int:
    start_idx = np.random.randint(0, X.shape[0], size)
    return start_idx


class Ganite:
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        dim_out: int,
        alpha: float = 1,
        beta: float = 1,
        minibatch_size: int = 256,
        depth: int = 1,
        num_iterations: int = 10000,
        test_step: int = 50,
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

    def fit(
        self,
        Train_X: pd.DataFrame,
        Train_T: pd.DataFrame,
        Train_Y: pd.DataFrame,
        Test_X: pd.DataFrame,
        Test_Y: pd.DataFrame,
    ) -> dict:
        metrics: dict = {
            "gen_block": {
                "D_loss": [],
                "G_loss": [],
            },
            "ite_block": {
                "I_loss": [],
                "Loss_sqrt_PEHE": [],
                "Loss_ATE": [],
            },
        }

        # Iterations
        # Train G and D first
        for it in tqdm(range(self.num_iterations)):
            for kk in range(self.num_discr_iterations):
                idx_mb = sample_X(Train_X, self.minibatch_size)
                X_mb = Train_X[idx_mb, :]
                T_mb = np.reshape(Train_T[idx_mb], [self.minibatch_size, 1])
                Y_mb = np.reshape(Train_Y[idx_mb], [self.minibatch_size, 1])

                _, D_loss_curr = self.sess.run(
                    [self.D_solver, self.D_loss],
                    feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
                )

            idx_mb = sample_X(Train_X, self.minibatch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.minibatch_size, 1])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.minibatch_size, 1])

            _, G_loss_curr, Tilde_curr = self.sess.run(
                [self.G_solver, self.G_loss, self.Tilde],
                feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
            )

            # Testing
            if it % self.test_step == 0:
                metrics["gen_block"]["D_loss"].append(D_loss_curr)
                metrics["gen_block"]["G_loss"].append(G_loss_curr)

                print(f"Iter: {it}")
                print(f"D_loss: {D_loss_curr:.4}")
                print(f"G_loss: {G_loss_curr:.4}")
                print()

        # Train I and ID
        for it in tqdm(range(self.num_iterations)):

            idx_mb = sample_X(Train_X, self.minibatch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.minibatch_size, 1])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.minibatch_size, 1])

            _, I_loss_curr = self.sess.run(
                [self.I_solver, self.I_loss],
                feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb},
            )

            # Testing
            if it % self.test_step == 0:
                metrics_for_step = self.test(Test_X, Test_Y)

                metrics["ite_block"]["I_loss"].append(I_loss_curr)
                metrics["ite_block"]["Loss_sqrt_PEHE"].append(
                    metrics_for_step["sqrt_PEHE"]
                )
                metrics["ite_block"]["Loss_ATE"].append(metrics_for_step["ATE"])

                Loss_sqrt_PEHE = metrics_for_step["sqrt_PEHE"]
                Loss_ATE = metrics_for_step["ATE"]

                print(f"Iter: {it}")
                print(f"I_loss: {I_loss_curr:.4}")
                print(f"Loss_sqrt_PEHE_Out: {Loss_sqrt_PEHE:.4}")
                print(f"Loss_ATE_Out: {Loss_ATE:.4}")
                print("")

        return metrics

    def predict(self, Test_X: pd.DataFrame) -> pd.DataFrame:
        Hat_curr = self.sess.run([self.Hat], feed_dict={self.X: Test_X})[0]
        return pd.DataFrame(Hat_curr, columns=["A", "B"])

    def test(self, Test_X: pd.DataFrame, Test_Y: pd.DataFrame) -> pd.DataFrame:
        Loss_sqrt_PEHE = tf_utils.sqrt_PEHE(self.Y_T, self.Hat)
        Loss_ATE = tf_utils.ATE(self.Y_T, self.Hat)

        sqrt_PEHE, ATE = self.sess.run(
            [Loss_sqrt_PEHE, Loss_ATE],
            feed_dict={self.X: Test_X, self.Y_T: Test_Y},
        )

        return {
            "sqrt_PEHE": float(sqrt_PEHE),
            "ATE": float(ATE),
        }
