"""
Loss functions used by DIR methods.

Functions
---------
l1_loss               standard L1
weighted_l1_loss      sample-weighted L1     (LDS, SQInv, ConR regression term)
focal_l1_loss         focal regression loss  (Focal L1)
uvote_nll_loss        Laplace NLL loss       (UVote per-expert loss)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard mean-L1 loss."""
    return F.l1_loss(pred, target)


def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample weighted L1 loss; weights shape (B,), broadcast over residuals."""
    loss = F.l1_loss(pred, target, reduction="none")
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()


def focal_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.2,
    gamma: float = 1.0,
    activate: str = "sigmoid",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal L1 regression loss.

        loss_i = |r_i| * (2*sigmoid(beta*|r_i|) - 1)^gamma   if activate='sigmoid'
        loss_i = |r_i| * tanh(beta*|r_i|)^gamma              if activate='tanh'

    where r_i = pred - target.
    """
    loss = F.l1_loss(pred, target, reduction="none")
    residual = torch.abs(pred - target)
    if activate == "tanh":
        loss = loss * (torch.tanh(beta * residual)) ** gamma
    else:
        loss = loss * (2 * torch.sigmoid(beta * residual) - 1) ** gamma
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()


def uvote_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weighted Laplace NLL loss for one UVote expert head.

    Parameters
    ----------
    mu, log_var, target, weights : shape (B, 1).  log_var is clamped to [-5, 5].
    """
    log_sigma = torch.clamp(log_var, min=-5, max=5)
    sigma = log_sigma.exp() + 1e-10
    lap_dist = torch.distributions.Laplace(loc=mu, scale=sigma)
    loss = -lap_dist.log_prob(target)
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()
