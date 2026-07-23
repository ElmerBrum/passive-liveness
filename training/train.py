"""
Training loop for MultiFTNet (MiniFASNet + Fourier aux branch).

Loss = (1 - w) * CrossEntropy(cls, label) + w * MSE(ft_map, fourier_target)
(w = cfg.ft_loss_weight, default 0.5 — same balance as the original repo).

Designed for a single GPU (RTX 5070 / Blackwell). Mixed precision (AMP) on by
default. Saves checkpoints in the contract format documented in README.md so
evaluate.py and the ONNX export can consume them.

Usage:
    python training/train.py                     # config.py defaults
    python training/train.py --epochs 5 --batch-size 128
    python training/train.py --device cpu        # tiny CPU smoke-test
"""

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:                      # tensorboard is optional
    SummaryWriter = None
    _HAS_TB = False

from config import TrainConfig, get_default_config
from dataset import build_loader
from models import build_multiftnet


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _NullWriter:
    """No-op stand-in for SummaryWriter when tensorboard isn't installed."""
    def add_scalar(self, *a, **k): pass
    def close(self): pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


@torch.no_grad()
def evaluate_split(model, loader, device) -> tuple[float, float]:
    """Returns (accuracy, live-recall) on a loader. Model is set to eval()."""
    was_training = model.training
    model.eval()
    correct = total = 0
    live_correct = live_total = 0
    for img, _ft, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(img)                     # eval mode → logits only
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
        live_mask = label == 1
        live_total += live_mask.sum().item()
        live_correct += ((pred == label) & live_mask).sum().item()
    if was_training:
        model.train()
    acc = correct / max(total, 1)
    live_recall = live_correct / max(live_total, 1)
    return acc, live_recall


def save_checkpoint(model, cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone_state": model.backbone_state_dict(),
            "config": asdict(cfg),
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = build_loader(cfg, split="train", shuffle=True, augment=True)
    print(f"Train samples: {len(train_loader.dataset)} "
          f"({train_loader.dataset.class_counts()})")

    # Optional val split (may not exist in a smoke-test).
    val_loader = None
    try:
        val_loader = build_loader(cfg, split="val", shuffle=False, augment=False)
        print(f"Val samples:   {len(val_loader.dataset)} "
              f"({val_loader.dataset.class_counts()})")
    except (FileNotFoundError, RuntimeError):
        print("Val split not found — skipping validation.")

    model = build_multiftnet(cfg).to(device)
    cls_criterion = nn.CrossEntropyLoss()
    ft_criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, momentum=cfg.momentum,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.milestones), gamma=cfg.gamma,
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(cfg.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if _HAS_TB:
        writer = SummaryWriter(str(run_dir / "tb"))
    else:
        writer = _NullWriter()
        print("tensorboard not installed — metrics logged to stdout only.")
    print(f"Run dir: {run_dir}")

    w = cfg.ft_loss_weight
    step = 0
    best_metric = -1.0
    best_path = run_dir / "best.pth"

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "cls": 0.0, "ft": 0.0, "acc": 0.0, "n": 0}

        for img, ft, label in train_loader:
            img = img.to(device, non_blocking=True)
            ft = ft.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                cls_logits, ft_map = model(img)         # train mode → tuple
                loss_cls = cls_criterion(cls_logits, label)
                loss_ft = ft_criterion(ft_map, ft)
                loss = (1 - w) * loss_cls + w * loss_ft

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            running["loss"] += loss.item()
            running["cls"] += loss_cls.item()
            running["ft"] += loss_ft.item()
            running["acc"] += accuracy(cls_logits, label)
            running["n"] += 1

            if step % cfg.log_every == 0:
                n = running["n"]
                writer.add_scalar("train/loss", running["loss"] / n, step)
                writer.add_scalar("train/loss_cls", running["cls"] / n, step)
                writer.add_scalar("train/loss_ft", running["ft"] / n, step)
                writer.add_scalar("train/acc", running["acc"] / n, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                print(f"  epoch {epoch} step {step}  "
                      f"loss {running['loss']/n:.4f}  acc {running['acc']/n:.3f}")
                running = {k: 0.0 for k in running}

        scheduler.step()
        dt = time.time() - t0

        msg = f"[epoch {epoch}] {dt:.1f}s  lr={optimizer.param_groups[0]['lr']:.2e}"
        if val_loader is not None:
            acc, live_recall = evaluate_split(model, val_loader, device)
            writer.add_scalar("val/acc", acc, epoch)
            writer.add_scalar("val/live_recall", live_recall, epoch)
            msg += f"  val_acc={acc:.4f}  val_live_recall={live_recall:.4f}"
            if acc > best_metric:
                best_metric = acc
                save_checkpoint(model, cfg, best_path)
                msg += "  <- best"
        print(msg)

        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(model, cfg, run_dir / f"epoch_{epoch:03d}.pth")

    # Always save a final checkpoint.
    final_path = run_dir / "final.pth"
    save_checkpoint(model, cfg, final_path)
    writer.close()
    print(f"Done. Final checkpoint: {final_path}")
    if best_path.exists():
        print(f"Best checkpoint:  {best_path}  (val_acc={best_metric:.4f})")
    return best_path if best_path.exists() else final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config_from_args(args) -> TrainConfig:
    cfg = get_default_config()
    for f in ("data_root", "patch_info", "model_type", "device"):
        v = getattr(args, f)
        if v is not None:
            setattr(cfg, f, v)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.amp = False
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Train MultiFTNet (MiniFASNet + Fourier)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--patch-info", default=None)
    p.add_argument("--model-type", default=None,
                   help="MiniFASNetV1/V2/V1SE/V2SE")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cuda:0 | cpu")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    args = p.parse_args()

    cfg = build_config_from_args(args)
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    train(cfg)


if __name__ == "__main__":
    main()
