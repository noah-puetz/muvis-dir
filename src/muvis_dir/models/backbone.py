"""Encoder + linear-head split of ResNet1D, exposing the interface DIR methods need.

The encoder runs everything up to and including the FC bottleneck and dropout;
the head is the final ``Linear(fc_units -> output_size)``.  This split lets RnC
freeze the encoder (stage 2 linear probing) and lets UVote replace the head
with multiple expert (mu, log_var) heads.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .resnet1d import ResNet1D


class _Encoder(nn.Module):
    """Encoder portion of ResNet1D — output shape: (batch, fc_units)."""

    def __init__(self, resnet: ResNet1D) -> None:
        super().__init__()
        self.input_conv = resnet.input_conv
        self.res_blocks = resnet.res_blocks
        self.dropout_res = resnet.dropout_res
        self.pool = resnet.pool
        self.fc = resnet.fc
        self.relu = resnet.relu
        self.dropout_fc = resnet.dropout_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.dropout_res(x)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc(x))
        x = self.dropout_fc(x)
        return x


class ResNet1DBackbone(nn.Module):
    """Wrapper around ``ResNet1D`` exposing encoder + replaceable head."""

    def __init__(
        self,
        dataset_name: str,
        config_dir: str,
        feat_dim: int,
    ) -> None:
        """
        Parameters
        ----------
        dataset_name : MuViS dataset id (e.g. ``"BeijingPM25Quality"`` or
            ``"REVS/2013_Monterey_Motorsports_Reunion"``).  Used to locate the
            architecture config file at ``<config_dir>/<dataset_name>.yaml``
            (slashes in the dataset id are replaced with ``__`` for the lookup
            so all configs sit in one flat directory).
        config_dir   : path to the dataset architecture config directory.
        feat_dim     : number of input channels (``X.shape[2]``); discovered
            once at data-load time and passed in.
        """
        super().__init__()

        cfg_name = dataset_name.replace("/", "__") + ".yaml"
        config_path = Path(config_dir) / cfg_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"ResNet1D config not found at {config_path}. "
                f"dataset_name={dataset_name!r}, config_dir={config_dir!r}."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        params = dict(config["architecture"])
        params["feat_dim"] = feat_dim

        resnet = ResNet1D(**params)

        self.encoder: _Encoder = _Encoder(resnet)
        self.head: nn.Module = resnet.out

        self._feature_dim = int(params["fc_units"])
        self._dataset_name = dataset_name

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder-only forward; output shape: (batch, feature_dim)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.head(features)
        return out[:, -1]

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_head(self) -> nn.Module:
        return self.head

    def replace_head(self, new_head: nn.Module) -> None:
        self.head = new_head
