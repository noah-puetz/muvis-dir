"""
UVote (Uncertainty-aware Voting Ensemble) multi-head model.

Replaces the regression head with ``num_branch`` expert heads, each
predicting a Laplace location (mu) and log-scale (log_var).  Each expert
trains under a different per-sample reweighting; ``dynamic_loss=True``
phases later experts in gradually with ``alpha = 1 - (epoch/E)^2``.
At inference, predictions are precision-weighted (inverse-variance)
averages of expert means.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..models.backbone import ResNet1DBackbone


class UVOTEModel(nn.Module):
    """UVote wrapper around ``ResNet1DBackbone``."""

    def __init__(self, backbone: ResNet1DBackbone, num_branch: int = 2) -> None:
        super().__init__()
        assert num_branch >= 1, num_branch

        self.encoder = backbone.get_encoder()
        feat_dim = backbone.feature_dim
        self.num_branch = num_branch

        self.mu_linears = nn.ModuleList(
            [nn.Linear(feat_dim, 1) for _ in range(num_branch)]
        )
        self.logvar_linears = nn.ModuleList(
            [nn.Linear(feat_dim, 1) for _ in range(num_branch)]
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list:
        features = self.encoder(x)
        outputs = []
        for i in range(self.num_branch):
            mu = self.mu_linears[i](features)
            log_var = self.logvar_linears[i](features)
            outputs.append((mu, log_var))
        return outputs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Precision-weighted average of expert means.  Output shape: (B,)."""
        outputs = self.forward(x)
        mus = torch.cat([mu for mu, _ in outputs], dim=1)
        logvars = torch.cat([lv for _, lv in outputs], dim=1)
        logvars = torch.clamp(logvars, -5, 5)
        w = torch.exp(-logvars)
        return (w * mus).sum(dim=1) / w.sum(dim=1)

    def get_parameter_groups(self, lr: float) -> list:
        """Two parameter groups: main params at ``lr``, log-var heads at ``lr * 0.1``."""
        logvar_params = list(self.logvar_linears.parameters())
        logvar_ids = {id(p) for p in logvar_params}
        main_params = [p for p in self.parameters() if id(p) not in logvar_ids]
        return [
            {"params": main_params, "lr": lr},
            {"params": logvar_params, "lr": lr * 0.1},
        ]
