"""
train.py — unified training driver for all DIR methods on MuViS datasets.

Methods: vanilla  lds  sqinv  focal_l1  conr  rnc  uvote

Usage
-----
    python -m muvis_dir.train \
        --dataset BeijingPM25Quality \
        --method  lds \
        --seed    42 \
        --data-dir   /path/to/data \
        --config-dir configs

Outputs
-------
    <output-dir>/<dataset_safe>_<method>_seed<seed>_predictions.npz
        keys: y_test, yhat_test, y_train

Hyperparameter resolution order (later overrides earlier):
    1. configs/shared.yaml                                    — defaults common to all methods
    2. configs/method/<method>.yaml                           — method-specific values
    3. configs/best_per_pair.yaml entry for (dataset, method) — paper-selected overrides
    4. CLI flags                                              — explicit overrides
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data import TimeSeriesDataset, load_muvis_dataset
from .methods.conr import conr_loss
from .methods.label_distribution import (
    compute_lds_weights,
    compute_sqinv_weights,
    compute_uvote_weights,
)
from .methods.losses import focal_l1_loss, l1_loss, uvote_nll_loss, weighted_l1_loss
from .methods.rnc import RnCLoss
from .methods.uvote import UVOTEModel
from .models import ResNet1DBackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


METHODS = ("vanilla", "lds", "sqinv", "focal_l1", "conr", "rnc", "uvote")

# ─── Reproducibility ─────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─── LR schedules ────────────────────────────────────────────────────────────


def _stepwise_lr(optimizer, epoch: int, lr: float, schedule: list[int]) -> None:
    """Decay lr by ×0.1 at each milestone epoch."""
    new_lr = lr
    for milestone in schedule:
        if epoch >= milestone:
            new_lr *= 0.1
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


def _cosine_lr(optimizer, epoch: int, lr: float, epochs: int, decay_rate: float) -> None:
    """Cosine annealing — RnC convention."""
    eta_min = lr * (decay_rate**3)
    new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


# ─── Config resolution ───────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_config(args: argparse.Namespace, config_dir: Path) -> dict:
    """Compose shared + method + per-pair config; CLI flags override last."""
    shared = _load_yaml(config_dir / "shared.yaml")
    method = _load_yaml(config_dir / "method" / f"{args.method}.yaml")
    best_per_pair = _load_yaml(config_dir / "best_per_pair.yaml")

    pair_overrides: dict = {}
    pair_entry = best_per_pair.get(_pair_key(args.dataset), {}).get(args.method)
    if pair_entry:
        pair_overrides = pair_entry.get("overrides") or {}

    cfg = {**shared, **method, **pair_overrides}

    # CLI overrides for the few flags exposed on the command line
    cli_keys = (
        "lr",
        "batch_size",
        "epochs",
        "schedule",
        "num_workers",
        "num_bins",
        "lds_kernel",
        "lds_ks",
        "lds_sigma",
        "lds_reweight",
        "focal_beta",
        "focal_gamma",
        "conr_w",
        "conr_beta",
        "conr_e",
        "conr_temp",
        "rnc_temp",
        "rnc_lr_s1",
        "rnc_epochs_s1",
        "rnc_decay_s1",
        "rnc_lr_s2",
        "rnc_epochs_s2",
        "rnc_decay_s2",
        "rnc_batch_size",
        "uvote_num_branch",
        "uvote_dynamic_loss",
    )
    for k in cli_keys:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def _pair_key(dataset_name: str) -> str:
    """Manifest keys flatten ``REVS/2013_…`` to ``REVS_2013_…``."""
    return dataset_name.replace("/", "_")


# ─── DataLoaders ─────────────────────────────────────────────────────────────


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, g) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )


# ─── Eval / prediction helpers ───────────────────────────────────────────────


@torch.no_grad()
def _validate(model, loader, device, is_uvote: bool = False) -> float:
    model.eval()
    total_l1, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model.predict(x) if is_uvote else model(x)
        total_l1 += torch.abs(pred - y).sum().item()
        n += len(y)
    return total_l1 / n if n > 0 else 0.0


@torch.no_grad()
def _predict_all(model, loader, device, is_uvote: bool = False) -> np.ndarray:
    model.eval()
    chunks = []
    for x, _, _ in loader:
        x = x.to(device)
        pred = model.predict(x) if is_uvote else model(x)
        chunks.append(pred.cpu().numpy())
    return np.concatenate(chunks)


# ─── Training loops ──────────────────────────────────────────────────────────


def _train_epoch_standard(
    model,
    loader,
    optimizer,
    device,
    method: str,
    cfg: dict,
) -> float:
    """vanilla / lds / sqinv / focal_l1 / conr — one epoch."""
    model.train()
    total_loss, n = 0.0, 0
    for x, y, w in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(w, torch.Tensor):
            w = w.to(device)

        if method in ("vanilla", "lds", "sqinv"):
            pred = model(x)
            loss = l1_loss(pred, y) if method == "vanilla" else weighted_l1_loss(pred, y, w)
        elif method == "focal_l1":
            pred = model(x)
            loss = focal_l1_loss(
                pred, y,
                beta=cfg.get("focal_beta", 0.2),
                gamma=cfg.get("focal_gamma", 1.0),
            )
        elif method == "conr":
            features = model.extract_features(x)
            pred = model.head(features)[:, -1]
            loss_reg = weighted_l1_loss(pred, y, w)

            x_c = torch.cat([x, x], dim=0)
            y_col = y.unsqueeze(1)
            w_col = w.unsqueeze(1) if w.ndim == 1 else w
            tgt_c = y_col.repeat(2, 1)
            wgt_c = w_col.repeat(2, 1)
            feat_c = model.extract_features(x_c)
            out_c = model.head(feat_c).detach()
            loss_conr = conr_loss(
                feat_c, tgt_c, out_c,
                w=cfg.get("conr_w", 1.0),
                weights=wgt_c,
                e=cfg.get("conr_e", 0.01),
                temperature=cfg.get("conr_temp", 0.2),
            )
            loss = loss_reg + cfg.get("conr_beta", 4.0) * loss_conr
        else:
            raise ValueError(method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return total_loss / n if n > 0 else 0.0


def _train_epoch_uvote(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    total_epochs: int,
    dynamic_loss: bool,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    alpha = 1.0 - (epoch / total_epochs) ** 2

    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        branch_w = [wi.to(device) for wi in w]
        y_col = y.unsqueeze(1)

        outputs = model(x)
        loss = None
        for i, (mu, lv) in enumerate(outputs):
            lv = torch.clamp(lv, -5, 5)
            wi = branch_w[i].unsqueeze(1)
            nll = uvote_nll_loss(mu, lv, y_col, weights=wi)
            if loss is None:
                loss = nll
            else:
                loss = loss + ((1.0 - alpha) * nll if dynamic_loss else nll)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return total_loss / n if n > 0 else 0.0


def _train_rnc_stage1(backbone, train_loader, val_loader, device, cfg: dict) -> None:
    backbone.train()
    criterion = RnCLoss(temperature=cfg["rnc_temp"])
    lr = cfg["rnc_lr_s1"]
    epochs = cfg["rnc_epochs_s1"]
    decay_rate = cfg["rnc_decay_s1"]
    optimizer = torch.optim.SGD(
        backbone.get_encoder().parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    log.info("RnC Stage 1: contrastive pre-training for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        _cosine_lr(optimizer, epoch, lr, epochs, decay_rate)
        backbone.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            bsz = x.shape[0]
            x_2 = torch.cat([x, x], dim=0)
            f = backbone.extract_features(x_2)
            f1, f2 = f[:bsz], f[bsz:]
            f_pair = torch.stack([f1, f2], dim=1)
            labels = y.unsqueeze(1)
            loss = criterion(f_pair, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0 or epoch == epochs:
            log.info("RnC S1 epoch %4d/%d  val L1 %.4f", epoch, epochs,
                     _validate(backbone, val_loader, device))


def _train_rnc_stage2(backbone, train_loader, val_loader, device, cfg: dict) -> None:
    for param in backbone.get_encoder().parameters():
        param.requires_grad = False

    new_head = nn.Linear(backbone.feature_dim, 1)
    nn.init.normal_(new_head.weight, 0, 0.01)
    nn.init.constant_(new_head.bias, 0)
    new_head = new_head.to(device)
    backbone.replace_head(new_head)

    lr = cfg["rnc_lr_s2"]
    epochs = cfg["rnc_epochs_s2"]
    decay_rate = cfg["rnc_decay_s2"]
    optimizer = torch.optim.SGD(new_head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.L1Loss()

    log.info("RnC Stage 2: linear probing for %d epochs", epochs)
    best_val_l1, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        _cosine_lr(optimizer, epoch, lr, epochs, decay_rate)
        backbone.train()
        new_head.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = backbone.extract_features(x)
            pred = new_head(features)[:, -1]
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_l1 = _validate(backbone, val_loader, device)
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_state = {k: v.clone() for k, v in new_head.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs:
            log.info("RnC S2 epoch %3d/%d  val L1 %.4f (best %.4f)",
                     epoch, epochs, val_l1, best_val_l1)

    if best_state is not None:
        new_head.load_state_dict(best_state)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="muvis_dir.train",
        description="Train one DIR method on one MuViS dataset and write predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   help="MuViS dataset id, e.g. 'BeijingPM25Quality' or "
                        "'REVS/2013_Monterey_Motorsports_Reunion'")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", required=True,
                   help="Root containing per-dataset .ts files.")
    p.add_argument("--config-dir", default="configs",
                   help="Directory containing shared.yaml, method/, dataset/, "
                        "best_per_pair.yaml.")
    p.add_argument("--output-dir", default="predictions",
                   help="Output directory for prediction .npz files.")
    p.add_argument("--device", default=None,
                   help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")

    # CLI overrides for hyperparameters (None → resolve from configs)
    for flag, kind in (
        ("--lr", float), ("--batch_size", int), ("--epochs", int),
        ("--num_workers", int), ("--num_bins", int),
        ("--lds_ks", int), ("--lds_sigma", float),
        ("--focal_beta", float), ("--focal_gamma", float),
        ("--conr_w", float), ("--conr_beta", float),
        ("--conr_e", float), ("--conr_temp", float),
        ("--rnc_temp", float), ("--rnc_lr_s1", float),
        ("--rnc_epochs_s1", int), ("--rnc_decay_s1", float),
        ("--rnc_lr_s2", float), ("--rnc_epochs_s2", int),
        ("--rnc_decay_s2", float), ("--rnc_batch_size", int),
        ("--uvote_num_branch", int), ("--uvote_dynamic_loss", int),
    ):
        p.add_argument(flag, type=kind, default=None)
    p.add_argument("--schedule", type=int, nargs="*", default=None,
                   help="Epoch milestones for ×0.1 LR decay.")
    p.add_argument("--lds_kernel", default=None)
    p.add_argument("--lds_reweight", default=None, choices=["sqrt_inv", "inverse"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_dir = Path(args.config_dir)
    cfg = _resolve_config(args, config_dir)

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    log.info("Device: %s", device)
    log.info("Dataset: %s | Method: %s | Seed: %d", args.dataset, args.method, args.seed)

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    log.info("Loading data from %s …", args.data_dir)
    data = load_muvis_dataset(
        args.dataset,
        data_dir=args.data_dir,
        val_split=0.1,
        seed=args.seed,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    log.info("Train=%d  Val=%d  Test=%d  feat_dim=%d",
             len(y_train), len(y_val), len(y_test), data["feat_dim"])

    train_weights = None
    if args.method == "lds":
        log.info("Computing LDS weights …")
        train_weights = compute_lds_weights(
            y_train,
            kernel=cfg["lds_kernel"],
            ks=cfg["lds_ks"],
            sigma=cfg["lds_sigma"],
            reweight=cfg["lds_reweight"],
            num_bins=cfg["num_bins"],
        )
    elif args.method == "sqinv":
        log.info("Computing SQInv weights …")
        train_weights = compute_sqinv_weights(y_train, num_bins=cfg["num_bins"])
    elif args.method == "uvote":
        log.info("Computing UVote weights (num_branch=%d) …", cfg["uvote_num_branch"])
        train_weights = compute_uvote_weights(
            y_train,
            num_branch=cfg["uvote_num_branch"],
            num_bins=cfg["num_bins"],
        )

    train_ds = TimeSeriesDataset(X_train, y_train, weights=train_weights)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)
    train_loader = _make_loader(train_ds, bs, True, nw, g)
    val_loader = _make_loader(val_ds, bs, False, nw, g)
    test_loader = _make_loader(test_ds, bs, False, nw, g)

    backbone = ResNet1DBackbone(
        dataset_name=args.dataset,
        config_dir=str(config_dir / "dataset"),
        feat_dim=data["feat_dim"],
    ).to(device)
    log.info("Backbone: feature_dim=%d", backbone.feature_dim)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset.replace('/', '_')}_{args.method}_seed{args.seed}"
    pred_path = out_dir / f"{tag}_predictions.npz"

    # ─── RnC: two-stage SGD ──────────────────────────────────────────────
    if args.method == "rnc":
        rnc_bs = cfg["rnc_batch_size"]
        rnc_train_loader = _make_loader(train_ds, rnc_bs, True, nw, g)
        rnc_val_loader = _make_loader(val_ds, rnc_bs, False, nw, g)

        _train_rnc_stage1(backbone, rnc_train_loader, rnc_val_loader, device, cfg)
        _train_rnc_stage2(backbone, rnc_train_loader, rnc_val_loader, device, cfg)

        rnc_test_loader = _make_loader(test_ds, rnc_bs, False, nw, g)
        yhat_test = _predict_all(backbone, rnc_test_loader, device)
        np.savez(pred_path, y_test=y_test, yhat_test=yhat_test, y_train=y_train)
        log.info("Wrote %s", pred_path)
        return

    # ─── UVote: multi-head + gradual schedule ────────────────────────────
    if args.method == "uvote":
        uvote_model = UVOTEModel(backbone, num_branch=cfg["uvote_num_branch"]).to(device)
        param_groups = uvote_model.get_parameter_groups(lr=cfg["lr"])
        optimizer = torch.optim.Adam(param_groups)
        dynamic_loss = bool(cfg.get("uvote_dynamic_loss", 1))

        log.info("UVote: %d epochs, batch=%d, num_branch=%d, dynamic_loss=%s",
                 cfg["epochs"], cfg["batch_size"], cfg["uvote_num_branch"], dynamic_loss)

        best_val_l1, best_state = float("inf"), None
        for epoch in range(1, cfg["epochs"] + 1):
            _stepwise_lr(optimizer, epoch, cfg["lr"], cfg["schedule"])
            _train_epoch_uvote(uvote_model, train_loader, optimizer, device,
                               epoch, cfg["epochs"], dynamic_loss)
            val_l1 = _validate(uvote_model, val_loader, device, is_uvote=True)
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                best_state = {k: v.clone() for k, v in uvote_model.state_dict().items()}
            if epoch % 10 == 0 or epoch == cfg["epochs"]:
                log.info("epoch %3d/%d  val L1 %.4f (best %.4f)",
                         epoch, cfg["epochs"], val_l1, best_val_l1)

        if best_state:
            uvote_model.load_state_dict(best_state)
        yhat_test = _predict_all(uvote_model, test_loader, device, is_uvote=True)
        np.savez(pred_path, y_test=y_test, yhat_test=yhat_test, y_train=y_train)
        log.info("Wrote %s", pred_path)
        return

    # ─── Standard methods: vanilla / lds / sqinv / focal_l1 / conr ───────
    optimizer = torch.optim.Adam(backbone.parameters(), lr=cfg["lr"])
    log.info("%s: %d epochs, batch=%d, lr=%.5f",
             args.method.upper(), cfg["epochs"], cfg["batch_size"], cfg["lr"])

    best_val_l1, best_state = float("inf"), None
    for epoch in range(1, cfg["epochs"] + 1):
        _stepwise_lr(optimizer, epoch, cfg["lr"], cfg["schedule"])
        _train_epoch_standard(backbone, train_loader, optimizer, device, args.method, cfg)
        val_l1 = _validate(backbone, val_loader, device)
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
        if epoch % 20 == 0 or epoch == cfg["epochs"]:
            log.info("epoch %3d/%d  val L1 %.4f (best %.4f)",
                     epoch, cfg["epochs"], val_l1, best_val_l1)

    if best_state:
        backbone.load_state_dict(best_state)
    yhat_test = _predict_all(backbone, test_loader, device)
    np.savez(pred_path, y_test=y_test, yhat_test=yhat_test, y_train=y_train)
    log.info("Wrote %s", pred_path)


if __name__ == "__main__":
    main()
