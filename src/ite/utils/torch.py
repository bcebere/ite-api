# third party
import torch


def squared_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute (x - y)(x - y) element-wise.
    """
    return (x - y) ** 2


def sqrt_PEHE(y: torch.Tensor, hat_y: torch.Tensor) -> torch.Tensor:
    """
    Precision in Estimation of Heterogeneous Effect(PyTorch version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return torch.sqrt(
        torch.mean(squared_difference((y[:, 1] - y[:, 0]), (hat_y[:, 1] - hat_y[:, 0])))
    )


def ATE(y: torch.Tensor, hat_y: torch.Tensor) -> torch.Tensor:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return torch.abs(
        torch.mean(y[:, 1] - y[:, 0]) - torch.mean(hat_y[:, 1] - hat_y[:, 0])
    )


def sigmoid_cross_entropy_with_logits(
    labels: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """
    Equivalent of TensorFlow sigmoid_cross_entropy_with_logits.
    Measures the probability error in discrete classification tasks
    in which each class is independent and not mutually exclusive.
    """

    logits[logits < 0] = 0
    return torch.mean(
        logits + -logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))
    )
