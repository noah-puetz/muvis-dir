# Exposing Blind Spots in Deep Imbalanced Regression Evaluation

## Overview

This repository reproduces the per-(method, dataset) results table from
*Exposing Blind Spots in Deep Imbalanced Regression Evaluation* (NeurIPS 2026
Datasets & Benchmarks track, anonymous submission).  It applies six published
DIR methods — LDS, SQInv, Focal-L<sub>1</sub>, ConR, RnC, and UVote — plus a
vanilla L<sub>1</sub> baseline to the nine multivariate time-series datasets
of MuViS, using a per-dataset ResNet1D backbone with a single shared training
protocol per method.  The point of the paper is that standard MAE / R²
disguise systematic failures in the tail; this repo emits per-seed
predictions whose balanced metrics expose those failures.

## Hardware Requirements

Tested on a single NVIDIA H100 (80 GB) with CUDA 13.2.  Minimum 24 GB GPU is
sufficient for every method; RnC stage 1 (400 epochs) is the binding
constraint.  The full sweep is 6 methods × 9 datasets × 10 seeds = 540 runs.

## Installation

Requires Python 3.13.7. No system packages beyond a working
CUDA-enabled PyTorch are needed.

```bash
git clone <repo-url> muvis-dir && cd muvis-dir
uv venv && source .venv/bin/activate
uv pip install -e .
```

Addittionally you neede the MuViS package and data. For that follow the instructions under [MuViS](https://github.com/noah-puetz/MuViS), download the corresponding data and execute the prepocessing.

## Quick Start

Train Vanilla on the smallest dataset (Vehicle Dynamics, 1.4k train) for 5
epochs as a sanity check (< 10 minutes on the recommended GPU). DATA_DIR should correspond to the `MuViS/data/processed/` directory.

```bash
make smoke DATA_DIR=$DATA_DIR
```

Equivalent direct invocation:

```bash
python -m muvis_dir.train \
    --dataset VehicleDynamicsDataset \
    --method  vanilla \
    --seed    42 \
    --data-dir $DATA_DIR \
    --output-dir predictions/seed_42 \
    --epochs 5
```

Output: ``predictions/seed_42/VehicleDynamicsDataset_vanilla_seed42_predictions.npz``
with arrays ``y_test, yhat_test, y_train``.

## Reproducing the Paper Table

``make reproduce`` walks the full grid of 6 methods × 9 datasets × 10 seeds
= 540 training runs and writes one prediction file per run under
``predictions/seed_<seed>/<dataset_safe>_<method>_seed<seed>_predictions.npz``.

```bash
make reproduce DATA_DIR=$DATA_DIR
```

## Repository Layout

```
muvis-dir/
├── README.md                          this file
├── Makefile                           smoke / reproduce
├── LICENSE                            MIT (code only)
├── pyproject.toml                     dependencies + entry points
├── configs/
│   ├── shared.yaml                    common training defaults
│   ├── method/<method>.yaml           per-method static hyperparameters
│   ├── dataset/<dataset>.yaml         per-dataset ResNet1D architecture
│   └── best_per_pair.yaml             paper-selected per-pair overrides
└── src/muvis_dir/
    ├── train.py                       training driver + CLI
    ├── data/loader.py                 .ts loading + z-scoring + 90/10 split
    ├── methods/
    │   ├── losses.py                  L1 / weighted L1 / focal / Laplace NLL
    │   ├── label_distribution.py      LDS / SQInv / UVote weight computation
    │   ├── conr.py                    ConR contrastive regularizer
    │   ├── rnc.py                     RnC ranking contrastive loss
    │   └── uvote.py                   UVote multi-head model
    ├── models/
    │   ├── resnet1d.py                ResNet1D backbone (ResidualBlock1D)
    │   └── backbone.py                encoder/head split for DIR methods
    └── eval/
        └── metrics.py                 bMAE / bMASE / Freedman--Diaconis bins
```

## Configuration

Configs compose left-to-right; each later layer overrides earlier ones:

1. ``configs/shared.yaml`` — defaults common to every method.
2. ``configs/method/<method>.yaml`` — method-specific values (LR, batch
   size, schedule, method-introduced hyperparameters).
3. ``configs/best_per_pair.yaml`` — the per-(dataset, method) overrides
   selected by the paper sweep on test-set bMAE at seed 42.
4. CLI flags — explicit overrides; pass ``--lr 0.01`` etc.  All
   hyperparameters in the YAML files are exposed on the ``muvis_dir.train``
   CLI.

The ResNet1D architecture is *separate* and read from
``configs/dataset/<dataset>.yaml`` — its values are upstream MuViS HPO
output and are not modified by the DIR experiments.

## Citation

```bibtex
@inproceedings{anonymous2026exposing,
  title  = {Exposing Blind Spots in Deep Imbalanced Regression Evaluation},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems 39: Datasets and Benchmarks Track},
  year   = {2026}
}
```

## License

Code in this repository is released under the MIT license (see ``LICENSE``).
The MuViS-DIR datasets it consumes are released by the upstream MuViS
benchmark under the licenses documented there; this repository does not
re-license the data.
