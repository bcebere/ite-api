# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# ite absolute
from ite.utils.metrics import HistoricMetrics
from ite.utils.metrics import Metrics
import ite.utils.torch as torch_utils


class CounterfactualGenerator(nn.Module):
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        super(CounterfactualGenerator, self).__init__()
        # Generator Layer
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.ReLU(),
        ] * depth
        self.common = nn.Sequential(
            nn.Linear(
                Dim + 2, DimHidden
            ),  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector      (Z)
            nn.ReLU(),
            *hidden,
        )

        self.out1 = nn.Sequential(
            nn.Linear(DimHidden, DimHidden), nn.ReLU(), nn.Linear(DimHidden, 1)
        )

        self.out2 = nn.Sequential(
            nn.Linear(DimHidden, DimHidden), nn.ReLU(), nn.Linear(DimHidden, 1)
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([x, t, y], dim=1)

        G_h2 = self.common(inputs)

        G_prob1 = self.out1(G_h2)
        G_prob2 = self.out2(G_h2)

        G_prob = torch.sigmoid(torch.cat([G_prob1, G_prob2], dim=1))

        return G_prob


class CounterfactualDiscriminator(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        super(CounterfactualDiscriminator, self).__init__()
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.ReLU(),
        ] * depth
        self.model = nn.Sequential(
            nn.Linear(Dim + 2, DimHidden),
            nn.ReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, hat_y: torch.Tensor
    ) -> torch.Tensor:
        # Factual & Counterfactual outcomes concatenate
        inp0 = (1.0 - t) * y + t * hat_y[:, 0].reshape([-1, 1])
        inp1 = t * y + (1.0 - t) * hat_y[:, 1].reshape([-1, 1])

        inputs = torch.cat([x, inp0, inp1], dim=1)
        return self.model(inputs)


class InferenceNets(nn.Module):
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(self, Dim: int, DimHidden: int, depth: int) -> None:
        super(InferenceNets, self).__init__()
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.ReLU(),
        ] * depth
        self.common = nn.Sequential(
            nn.Linear(Dim, DimHidden),
            nn.ReLU(),
            *hidden,
        )
        self.out1 = nn.Sequential(
            nn.Linear(DimHidden, DimHidden),
            nn.ReLU(),
            nn.Linear(DimHidden, 1),
        )

        self.out2 = nn.Sequential(
            nn.Linear(DimHidden, DimHidden),
            nn.ReLU(),
            nn.Linear(DimHidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I_h = self.common(x)

        I_prob1 = self.out1(I_h)
        I_prob2 = self.out2(I_h)

        return torch.sigmoid(torch.cat([I_prob1, I_prob2], dim=1))


class GaniteTorch:
    """
    The GANITE framework generates potential outcomes for a given feature vector x.
    It consists of 2 components:
     - The Counterfactual Generator block(generator + discriminator)
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
        num_iterations: int = 5000,
        test_step: int = 200,
        num_discr_iterations: int = 5,
    ) -> None:
        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.num_iterations = num_iterations
        self.test_step = test_step
        self.num_discr_iterations = num_discr_iterations

        # Layers
        self.counterfactual_generator = CounterfactualGenerator(dim, dim_hidden, depth)
        self.counterfactual_discriminator = CounterfactualDiscriminator(
            dim, dim_hidden, depth
        )
        self.inference_nets = InferenceNets(dim, dim_hidden, depth)

        # Solvers
        self.G_solver = torch.optim.Adam(self.counterfactual_generator.parameters())
        self.D_solver = torch.optim.Adam(self.counterfactual_discriminator.parameters())
        self.I_solver = torch.optim.Adam(self.inference_nets.parameters())

        # Metrics
        self.train_perf_metrics = HistoricMetrics()

    def sample_minibatch(
        self, X: torch.Tensor, T: torch.tensor, Y: torch.Tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        idx_mb = np.random.randint(0, X.shape[0], self.minibatch_size)

        X_mb = X[idx_mb, :]
        T_mb = T[idx_mb].reshape([self.minibatch_size, 1])
        Y_mb = Y[idx_mb].reshape([self.minibatch_size, 1])

        return X_mb, T_mb, Y_mb

    def train(
        self,
        df_Train_X: pd.DataFrame,
        df_Train_T: pd.DataFrame,
        df_Train_Y: pd.DataFrame,
        df_Opt_Train_Y: pd.DataFrame,
        df_Test_X: pd.DataFrame,
        df_Test_Y: pd.DataFrame,
    ) -> HistoricMetrics:
        Train_X = torch.from_numpy(df_Train_X).float()
        Train_T = torch.from_numpy(df_Train_T).float()
        Train_Y = torch.from_numpy(df_Train_Y).float()
        Opt_Train_Y = torch.from_numpy(df_Opt_Train_Y).float()
        Test_X = torch.from_numpy(df_Test_X).float()
        Test_Y = torch.from_numpy(df_Test_Y).float()

        # Iterations
        # Train G and D first
        self.counterfactual_generator.train()
        self.counterfactual_discriminator.train()
        self.inference_nets.train()

        for it in tqdm(range(self.num_iterations)):
            self.G_solver.zero_grad()

            for kk in range(self.num_discr_iterations):
                self.D_solver.zero_grad()

                X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

                Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
                D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)

                D_loss = torch_utils.sigmoid_cross_entropy_with_logits(T_mb, D_out)

                D_loss.backward(retain_graph=True)
                self.D_solver.step()

            X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
            D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)
            D_loss = torch_utils.sigmoid_cross_entropy_with_logits(T_mb, D_out)

            G_loss_GAN = -D_loss

            G_loss_R = torch.mean(
                nn.MSELoss()(
                    Y_mb,
                    T_mb * Tilde[:, 1].reshape([-1, 1])
                    + (1.0 - T_mb) * Tilde[:, 0].reshape([-1, 1]),
                )
            )
            G_loss = G_loss_R + self.alpha * G_loss_GAN

            G_loss.backward()
            self.G_solver.step()

            # Testing
            if it % self.test_step == 0:
                metric_block = "Counterfactual Block"
                self.train_perf_metrics.add(
                    "Discriminator loss", D_loss.detach().numpy(), metric_block
                )
                self.train_perf_metrics.add(
                    "Generator loss", G_loss.detach().numpy(), metric_block
                )

        # Train I and ID
        for it in tqdm(range(self.num_iterations)):
            self.I_solver.zero_grad()

            X_mb, T_mb, Y_mb = self.sample_minibatch(Train_X, Train_T, Train_Y)

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)

            hat = self.inference_nets(X_mb)
            I_loss1 = torch.mean(
                nn.MSELoss()(
                    T_mb * Y_mb + (1 - T_mb) * Tilde[:, 1].reshape([-1, 1]),
                    hat[:, 1].reshape([-1, 1]),
                )
            )
            I_loss2 = torch.mean(
                nn.MSELoss()(
                    (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0].reshape([-1, 1]),
                    hat[:, 0].reshape([-1, 1]),
                )
            )
            I_loss = I_loss1 + self.beta * I_loss2

            I_loss.backward()
            self.I_solver.step()

            # Testing
            if it % self.test_step == 0:
                metric_block = "ITE Block"
                self.train_perf_metrics.add(
                    "ITE loss", I_loss.detach().numpy(), metric_block
                )

                metric_block = "ITE Block in-sample metrics"
                metrics_for_step = self.test(Train_X.numpy(), Opt_Train_Y.numpy())

                self.train_perf_metrics.add(
                    "sqrt_PEHE", metrics_for_step.sqrt_PEHE(), metric_block
                )
                self.train_perf_metrics.add("ATE", metrics_for_step.ATE(), metric_block)
                self.train_perf_metrics.add("MSE", metrics_for_step.MSE(), metric_block)

                metric_block = "ITE Block out-sample metrics"
                metrics_for_step = self.test(Test_X.numpy(), Test_Y.numpy())

                self.train_perf_metrics.add(
                    "sqrt_PEHE", metrics_for_step.sqrt_PEHE(), metric_block
                )
                self.train_perf_metrics.add("ATE", metrics_for_step.ATE(), metric_block)
                self.train_perf_metrics.add("MSE", metrics_for_step.MSE(), metric_block)

        return self.train_perf_metrics

    def train_metrics(self) -> HistoricMetrics:
        return self.train_perf_metrics

    def predict(self, df_Test_X: pd.DataFrame) -> pd.DataFrame:
        with torch.no_grad():
            Test_X = torch.from_numpy(df_Test_X).float()
            y_hat = self.inference_nets(Test_X).detach().numpy()

        return pd.DataFrame(y_hat, columns=["y_hat_0", "y_hat_1"])

    def test(self, Test_X: pd.DataFrame, Test_Y: pd.DataFrame) -> Metrics:
        hat = self.predict(Test_X)

        return Metrics(hat.to_numpy(), Test_Y)
