"""
Rank-N-Contrast (RnC) loss for imbalanced regression (Zha et al. 2023).

Two-stage training: stage 1 contrastively pre-trains the encoder using
``RnCLoss``; stage 2 trains a fresh linear head on the frozen encoder with
plain L1.  The training driver lives in ``muvis_dir.train``; this module
implements the loss only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LabelDifference(nn.Module):
    """Pairwise L1 distance between scalar labels."""

    def __init__(self, distance_type: str = "l1") -> None:
        super().__init__()
        assert distance_type == "l1", distance_type
        self.distance_type = distance_type

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)


class FeatureSimilarity(nn.Module):
    """Pairwise negative L2 distance between feature vectors (acts as similarity)."""

    def __init__(self, similarity_type: str = "l2") -> None:
        super().__init__()
        assert similarity_type == "l2", similarity_type
        self.similarity_type = similarity_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1)


class RnCLoss(nn.Module):
    """Rank-N-Contrast loss.  Inputs: features (B, 2, D), labels (B, 1)."""

    def __init__(
        self,
        temperature: float = 2.0,
        label_diff: str = "l1",
        feature_sim: str = "l2",
    ) -> None:
        super().__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)
        labels = labels.repeat(2, 1)

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]
        eye_mask = (1 - torch.eye(n, device=logits.device)).bool()
        logits = logits[eye_mask].view(n, n - 1)
        exp_logits = exp_logits[eye_mask].view(n, n - 1)
        label_diffs = label_diffs[eye_mask].view(n, n - 1)

        loss = 0.0
        for k in range(n - 1):
            pos_logits = logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()
            pos_log_probs = pos_logits - torch.log(
                (neg_mask * exp_logits).sum(dim=-1) + 1e-8
            )
            loss += -(pos_log_probs / (n * (n - 1))).sum()

        return loss
