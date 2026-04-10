"""
train.py – Supervised training of the 3P mahjong model.

This script trains a Brain + DQN model on human/bot game logs that have been
converted to JSONL format by data_converter.py.

The training objective combines:
  1. Cross-entropy policy loss   (supervised imitation of recorded actions)
  2. MSE value loss              (regression to final-round score delta)
  3. Optional CQL regularisation (conservative offline RL penalty)

Quick start
-----------
    # Train for 1 epoch on a JSONL dataset (smoke test)
    python train.py --data sample.jsonl --epochs 1

    # Full training with GPU and custom hyper-parameters
    python train.py \\
        --data train.jsonl \\
        --val  val.jsonl \\
        --epochs 30 \\
        --batch-size 512 \\
        --lr 1e-3 \\
        --conv-channels 192 \\
        --num-blocks 40 \\
        --device cuda \\
        --out mortal_3p.pth

Output
------
* ``mortal_3p.pth``   – final checkpoint (Mortal-compatible format)
* ``best_3p.pth``     – best checkpoint by validation policy accuracy
* TensorBoard logs in ``./runs/train_3p_<timestamp>``  (if tensorboard installed)

Resuming
--------
    python train.py --data train.jsonl --resume mortal_3p.pth --epochs 60
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import random_split

# Allow running from train_3p/ directory or from repo root
sys.path.insert(0, str(Path(__file__).parent))

from data_converter import ACTION_SPACE_3P, OBS_CHANNELS_3P
from dataset import MahjongDataset3P, build_dataloader
from model import Brain, DQN, build_model, save_checkpoint, load_checkpoint, parameter_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def policy_loss(q_out: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between argmax-policy of Q values and recorded actions."""
    return nn.functional.cross_entropy(q_out, actions)


def value_loss(q_out: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """MSE between Q(s,a) for the taken action and the observed reward."""
    q_taken = q_out[torch.arange(len(actions)), actions]
    return nn.functional.mse_loss(q_taken, rewards)


def cql_loss(q_out: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Conservative Q-Learning penalty: log-sum-exp(Q) - Q(s,a)."""
    q_taken = q_out[torch.arange(len(actions)), actions]
    return q_out.logsumexp(dim=-1).mean() - q_taken.mean()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    mortal: Brain,
    dqn: DQN,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Return policy accuracy and value MAE on a data loader."""
    mortal.eval()
    dqn.eval()

    correct = 0
    total = 0
    val_err = 0.0

    for batch in loader:
        states = batch["state"].to(device)
        masks = batch["legal_actions_mask"].to(device)
        actions = batch["action"].to(device)
        rewards = batch["reward"].to(device)

        phi = mortal(states)
        q_out = dqn(phi, masks)

        preds = q_out.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total += len(actions)
        q_taken = q_out[torch.arange(len(actions)), actions]
        val_err += (q_taken - rewards).abs().sum().item()

    mortal.train()
    dqn.train()

    return {
        "policy_acc": correct / max(total, 1),
        "value_mae": val_err / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    # ------------------------------------------------------------------ Setup
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Device: {device}")
    logger.info(f"Data:   {args.data}")
    logger.info(f"Epochs: {args.epochs}")

    # ------------------------------------------------------------------ Data
    dataset = MahjongDataset3P(args.data, max_samples=args.max_samples)
    logger.info(f"Dataset: {len(dataset):,} examples")

    if args.val:
        train_ds = dataset
        val_ds = MahjongDataset3P(args.val, max_samples=args.max_samples)
    else:
        # Auto-split: 90% train, 10% val
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(f"Train batches per epoch: {len(train_loader):,}")

    # ------------------------------------------------------------------ Model
    mortal, dqn = build_model(
        version=args.model_version,
        conv_channels=args.conv_channels,
        num_blocks=args.num_blocks,
        device=device,
    )

    start_epoch = 0
    best_acc = 0.0

    if args.resume and Path(args.resume).exists():
        mortal, dqn = load_checkpoint(args.resume, device=device)
        mortal.train()
        dqn.train()
        logger.info(f"Resumed from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        start_epoch = state.get("steps", 0) // max(len(train_loader), 1)
        logger.info(f"Resuming from epoch ~{start_epoch}")

    logger.info(f"Brain params: {parameter_count(mortal):,}")
    logger.info(f"DQN params:   {parameter_count(dqn):,}")

    # ------------------------------------------------------------------ Optim
    all_params = list(mortal.parameters()) + list(dqn.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Cosine annealing with linear warm-up
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Optional AMP (automatic mixed precision)
    use_amp = args.enable_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ------------------------------------------------------------------ TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = f"runs/train_3p_{timestamp}"
        writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard: {tb_dir}")
    except ImportError:
        pass

    # ------------------------------------------------------------------ Training
    global_step = start_epoch * len(train_loader)
    best_ckpt_path = Path(args.out).parent / "best_3p.pth"

    mortal.train()
    dqn.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_cql_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            states = batch["state"].to(device)
            masks = batch["legal_actions_mask"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=use_amp):
                phi = mortal(states)
                q_out = dqn(phi, masks)

                p_loss = policy_loss(q_out, actions)
                v_loss = value_loss(q_out, actions, rewards)
                c_loss = cql_loss(q_out, actions) if args.cql_weight > 0 else torch.tensor(0.0, device=device)

                loss = (
                    args.policy_weight * p_loss
                    + args.value_weight * v_loss
                    + args.cql_weight * c_loss
                )

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_policy_loss += p_loss.item()
            epoch_value_loss += v_loss.item()
            epoch_cql_loss += c_loss.item()
            n_batches += 1
            global_step += 1

            if writer and global_step % 50 == 0:
                writer.add_scalar("train/policy_loss", p_loss.item(), global_step)
                writer.add_scalar("train/value_loss", v_loss.item(), global_step)
                writer.add_scalar("train/cql_loss", c_loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_p = epoch_policy_loss / max(n_batches, 1)
        avg_v = epoch_value_loss / max(n_batches, 1)
        avg_c = epoch_cql_loss / max(n_batches, 1)

        # Validation
        metrics = evaluate(mortal, dqn, val_loader, device)
        acc = metrics["policy_acc"]
        mae = metrics["value_mae"]
        mortal.train()
        dqn.train()

        logger.info(
            f"Epoch {epoch + 1:3d}/{args.epochs}"
            f"  policy_loss={avg_p:.4f}"
            f"  value_loss={avg_v:.4f}"
            f"  cql_loss={avg_c:.4f}"
            f"  val_acc={acc:.4f}"
            f"  val_mae={mae:.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if writer:
            writer.add_scalar("val/policy_acc", acc, epoch)
            writer.add_scalar("val/value_mae", mae, epoch)

        # Save best checkpoint
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(mortal, dqn, steps=global_step, path=best_ckpt_path)
            logger.info(f"  ↑ New best val_acc={best_acc:.4f}, saved to {best_ckpt_path}")

    # ------------------------------------------------------------------ Final save
    save_checkpoint(mortal, dqn, steps=global_step, path=args.out)
    logger.info(f"Saved final checkpoint to {args.out}")

    if writer:
        writer.close()

    return mortal, dqn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3P mahjong model (supervised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--data", required=True, help="Training JSONL/NPZ/PT file")
    parser.add_argument("--val", default="", help="Validation data (optional; auto-split if omitted)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate dataset (smoke test)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")

    # Model
    parser.add_argument("--model-version", type=int, default=4,
                        choices=[1, 2, 3, 4],
                        help="Model version (4 recommended)")
    parser.add_argument("--conv-channels", type=int, default=64,
                        help="ResNet conv channels (64=fast, 192=production)")
    parser.add_argument("--num-blocks", type=int, default=6,
                        help="ResNet blocks (6=fast, 40=production)")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--cql-weight", type=float, default=0.0,
                        help="CQL regularisation weight (0=disabled)")
    parser.add_argument("--enable-amp", action="store_true",
                        help="Enable automatic mixed precision (CUDA only)")

    # Misc
    parser.add_argument("--device", default="cpu",
                        help="Training device (cpu / cuda / cuda:0)")
    parser.add_argument("--out", default="mortal_3p.pth",
                        help="Output checkpoint path")
    parser.add_argument("--resume", default="",
                        help="Resume from an existing checkpoint")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    t0 = time.time()
    train(args)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
