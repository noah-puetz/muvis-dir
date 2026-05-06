"""
ConR (Contrastive Regularizer) loss for imbalanced regression.

For a batch of features ``f_i`` and labels ``y_i``:
    Positive pair (i, j): ``|y_i - y_j| <= w``
    Negative pair (i, j): ``|y_i - y_j| > w`` AND ``|yhat_i - yhat_j| <= w``  (hard)

ConR pulls positives together and pushes hard negatives apart, with a
label-distance pushing weight that up-weights negatives whose labels are
farther from the anchor.

Shape contract
--------------
features : (N, D)    N = 2B (doubled batch)
targets  : (N, 1)
preds    : (N, 1)    (detached)
weights  : (N, 1) or scalar
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def conr_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    w: float = 1.0,
    weights: torch.Tensor | float = 1.0,
    e: float = 0.01,
    temperature: float = 0.2,
) -> torch.Tensor:
    """ConR contrastive regularisation loss."""
    t = temperature

    q = F.normalize(features, dim=1)
    k = F.normalize(features, dim=1)

    l_k = targets.flatten()[None, :]
    l_q = targets
    p_k = preds.flatten()[None, :]
    p_q = preds

    l_dist = torch.abs(l_q - l_k)
    p_dist = torch.abs(p_q - p_k)

    pos_i = l_dist.le(w)
    neg_i = (~l_dist.le(w)) & (p_dist.le(w))

    for i in range(pos_i.shape[0]):
        pos_i[i, i] = False

    prod = torch.einsum("nc,kc->nk", q, k) / t
    pos = prod * pos_i.float()
    neg = prod * neg_i.float()

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * torch.exp(neg) * neg_i.float()).sum(1)

    no_neg_flag = neg_i.sum(1).bool()

    denom = pos_i.float().sum(1)
    denom = torch.clamp(denom, min=1.0)

    sum_pos_exp = torch.exp(pos).sum(1)
    denom_full = (sum_pos_exp + neg_exp_dot).unsqueeze(-1)
    log_probs = torch.log(torch.div(torch.exp(pos), denom_full) + 1e-8)
    per_anchor = (-log_probs * pos_i.float()).sum(1) / denom

    per_anchor = (per_anchor * no_neg_flag.float()).unsqueeze(-1)
    return (weights * per_anchor).mean()
