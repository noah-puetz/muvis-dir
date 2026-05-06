"""ResNet1D architecture (from MuViS).

Multivariate-time-series regression backbone:
    Conv1d(C -> num_filters, k=1) → num_res_blocks × ResidualBlock1D(k=3, padding='same')
    → AdaptiveAvgPool1d(1) → Linear(num_filters -> fc_units) → Linear(fc_units -> 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Two 3x conv layers with BatchNorm + ReLU + skip connection (1x conv if C changes)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ResNet1D(nn.Module):
    """1-D ResNet for multivariate time-series regression."""

    def __init__(
        self,
        feat_dim: int,
        output_size: int = 1,
        num_filters: int = 64,
        fc_units: int = 64,
        num_res_blocks: int = 2,
        dropout1: float = 0.2,
        dropout2: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout_res = nn.Dropout(dropout1)
        self.dropout_fc = nn.Dropout(dropout2)

        self.input_conv = nn.Conv1d(feat_dim, num_filters, kernel_size=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(num_filters, num_filters) for _ in range(num_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, fc_units)
        self.out = nn.Linear(fc_units, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feat_dim) — MuViS convention
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.dropout_res(x)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc(x))
        x = self.dropout_fc(x)
        out = self.out(x)
        return out[:, -1]
