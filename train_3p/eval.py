"""
eval.py – Evaluation script for 3P mahjong model.

Computes offline evaluation metrics:
  * Policy accuracy  – fraction of examples where argmax(Q) == recorded action
  * Top-5 accuracy   – fraction where recorded action is in top-5 Q values
  * Value MAE        – mean absolute error between Q(s,a) and observed reward
  * Action distribution – histogram of predicted vs recorded actions

Optionally runs a self-play tournament against a random baseline to produce
online win-rate statistics.

Usage
-----
    # Offline evaluation on a JSONL dataset
    python eval.py --checkpoint mortal_3p.pth --data val.jsonl

    # Self-play evaluation (5 games against random baseline)
    python eval.py --checkpoint mortal_3p.pth --self-play --games 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data_converter import ACTION_SPACE_3P, OBS_CHANNELS_3P, N_TILES, make_legal_actions_mask
from dataset import MahjongDataset3P
from model import load_checkpoint, parameter_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def offline_eval(
    checkpoint_path: str,
    data_path: str,
    device: torch.device,
    batch_size: int = 512,
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    """
    Compute offline metrics for the given checkpoint on a dataset.

    Parameters
    ----------
    checkpoint_path : Path to mortal_3p.pth
    data_path       : Path to JSONL/NPZ/PT evaluation file
    device          : Torch device
    batch_size      : DataLoader batch size
    max_samples     : Truncate dataset (None = use all)

    Returns
    -------
    dict with keys: policy_acc, top5_acc, value_mae, n_examples
    """
    mortal, dqn = load_checkpoint(checkpoint_path, device=device)
    mortal.eval()
    dqn.eval()

    ds = MahjongDataset3P(data_path, max_samples=max_samples)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    correct_top1 = 0
    correct_top5 = 0
    total_mae = 0.0
    n_total = 0

    action_hist_pred = np.zeros(ACTION_SPACE_3P, dtype=np.int64)
    action_hist_true = np.zeros(ACTION_SPACE_3P, dtype=np.int64)

    for batch in loader:
        states = batch["state"].to(device)
        masks = batch["legal_actions_mask"].to(device)
        actions = batch["action"].to(device)
        rewards = batch["reward"].to(device)

        phi = mortal(states)
        q_out = dqn(phi, masks)

        top1_pred = q_out.argmax(dim=-1)
        top5_pred = q_out.topk(min(5, ACTION_SPACE_3P), dim=-1).indices

        correct_top1 += (top1_pred == actions).sum().item()
        correct_top5 += sum(
            a.item() in t5.tolist()
            for a, t5 in zip(actions, top5_pred)
        )

        q_taken = q_out[torch.arange(len(actions)), actions]
        total_mae += (q_taken - rewards).abs().sum().item()
        n_total += len(actions)

        for p in top1_pred.cpu().numpy():
            action_hist_pred[p] += 1
        for a in actions.cpu().numpy():
            action_hist_true[a] += 1

    results = {
        "policy_acc": correct_top1 / max(n_total, 1),
        "top5_acc": correct_top5 / max(n_total, 1),
        "value_mae": total_mae / max(n_total, 1),
        "n_examples": n_total,
    }

    logger.info("=" * 50)
    logger.info("Offline Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"  Examples     : {n_total:,}")
    logger.info(f"  Policy Acc   : {results['policy_acc']:.4f}")
    logger.info(f"  Top-5 Acc    : {results['top5_acc']:.4f}")
    logger.info(f"  Value MAE    : {results['value_mae']:.4f}")
    logger.info("-" * 50)

    # Print top-10 most-predicted vs most-recorded actions
    from data_converter import ACTION_NAMES
    logger.info("Top-10 predicted actions:")
    for idx in action_hist_pred.argsort()[-10:][::-1]:
        logger.info(f"  [{idx:2d}] {ACTION_NAMES[idx]:30s} count={action_hist_pred[idx]}")
    logger.info("Top-10 recorded actions:")
    for idx in action_hist_true.argsort()[-10:][::-1]:
        logger.info(f"  [{idx:2d}] {ACTION_NAMES[idx]:30s} count={action_hist_true[idx]}")

    return results


# ---------------------------------------------------------------------------
# Simple self-play evaluation harness
# ---------------------------------------------------------------------------

def _random_policy(state: np.ndarray, mask: np.ndarray) -> int:
    """Baseline: choose uniformly at random from legal actions."""
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 77  # none
    return int(np.random.choice(legal))


def _model_policy(
    mortal,
    dqn,
    device: torch.device,
    state: np.ndarray,
    mask: np.ndarray,
) -> int:
    """Model: argmax Q over legal actions."""
    with torch.inference_mode():
        obs = torch.from_numpy(
            state.reshape(1, OBS_CHANNELS_3P, N_TILES)
        ).float().to(device)
        m = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        phi = mortal(obs)
        q = dqn(phi, m)
        return int(q.argmax(dim=-1).item())


class SimpleSelfPlayEnv:
    """
    A minimal synthetic 3P game environment for smoke-testing the model.

    This does NOT implement full mahjong rules. It generates random states and
    checks that the model can produce valid (legal) actions without crashing.
    The "win rate" reported is just a random benchmark – it is not a true
    strength measurement. Replace with a real mahjong engine for meaningful
    self-play evaluation.
    """

    def __init__(self, n_players: int = 3):
        self.n_players = n_players

    def reset(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (states, masks) for each player."""
        states = [
            np.random.randn(OBS_CHANNELS_3P, N_TILES).astype(np.float32)
            for _ in range(self.n_players)
        ]
        masks = [
            make_legal_actions_mask(
                can_discard=random.sample(range(34), k=random.randint(1, 13)),
                can_agari=random.random() < 0.05,
            )
            for _ in range(self.n_players)
        ]
        return states, masks

    def step(self, actions: list[int]) -> tuple[list[float], bool]:
        """Return (rewards, done)."""
        # Random outcome: one random player gets +1, others -0.5
        rewards = [-0.5] * self.n_players
        winner = random.randint(0, self.n_players - 1)
        rewards[winner] = 1.0
        done = True  # each episode is a single decision for smoke test
        return rewards, done


def self_play_eval(
    checkpoint_path: str,
    device: torch.device,
    n_games: int = 10,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Run *n_games* episodes of the model (seat 0) vs random baseline.

    Returns
    -------
    dict with keys: model_win_rate, model_avg_reward, baseline_avg_reward
    """
    mortal, dqn = load_checkpoint(checkpoint_path, device=device)
    mortal.eval()
    dqn.eval()

    env = SimpleSelfPlayEnv(n_players=3)
    model_rewards: list[float] = []
    baseline_rewards: list[float] = []

    for game_idx in range(n_games):
        states, masks = env.reset()

        # Seat 0 = model; seats 1, 2 = random
        actions = [
            _model_policy(mortal, dqn, device, states[0], masks[0]),
            _random_policy(states[1], masks[1]),
            _random_policy(states[2], masks[2]),
        ]
        rewards, _ = env.step(actions)
        model_rewards.append(rewards[0])
        baseline_rewards.append(sum(rewards[1:]) / 2)

        if verbose and (game_idx + 1) % max(1, n_games // 5) == 0:
            logger.info(f"  Game {game_idx + 1}/{n_games}: model_reward={rewards[0]:+.1f}")

    model_win_rate = sum(r > 0 for r in model_rewards) / max(len(model_rewards), 1)
    results = {
        "model_win_rate": model_win_rate,
        "model_avg_reward": float(np.mean(model_rewards)),
        "baseline_avg_reward": float(np.mean(baseline_rewards)),
        "n_games": n_games,
    }

    logger.info("=" * 50)
    logger.info("Self-Play Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"  Games        : {n_games}")
    logger.info(f"  Model win%   : {model_win_rate:.2%}")
    logger.info(f"  Model avg R  : {results['model_avg_reward']:+.4f}")
    logger.info(f"  Baseline R   : {results['baseline_avg_reward']:+.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate 3P mahjong model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to mortal_3p.pth")
    parser.add_argument("--data", default="", help="Evaluation JSONL/NPZ file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-play", action="store_true",
                        help="Run self-play evaluation against random baseline")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of self-play games")
    parser.add_argument("--out-json", default="",
                        help="Write results to JSON file (optional)")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    all_results: dict = {}

    if args.data:
        offline_results = offline_eval(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        all_results["offline"] = offline_results

    if args.self_play:
        sp_results = self_play_eval(
            checkpoint_path=args.checkpoint,
            device=device,
            n_games=args.games,
        )
        all_results["self_play"] = sp_results

    if not all_results:
        parser.error("Provide --data and/or --self-play")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        logger.info(f"Results written to {args.out_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
