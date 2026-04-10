"""
selfplay.py – Generate training data via self-play for the 3P mahjong model.

The self-play loop runs episodes in which the current model plays against
copies of itself (or a mix of model + random policies).  Collected
(state, action, reward) tuples are written to JSONL and fed back into
training by ``run_pipeline.py``.

Architecture
------------
                  ┌──────────────────────────────┐
                  │     SelfPlayCollector         │
                  │                               │
  checkpoint ──►  │  model × N seats (+ ε-random) │ ──► episodes.jsonl
                  │  +  GameEngine                │
                  └──────────────────────────────┘

Built-in game engine
--------------------
``SimpleSelfPlayEnv`` generates *synthetic* states so you can run the
pipeline end-to-end without a real mahjong engine.  The "win/loss" signal
is random, so the model trained purely on this data won't improve at
mahjong – but the pipeline is fully exercised and ready to plug in a real
engine.

Plug in a real engine
---------------------
Implement the ``GameEngine`` protocol below and pass it via
``--engine-module`` (a Python module path to a class called
``MahjongEngine``).  Expected interface::

    engine = MahjongEngine()
    states, masks = engine.reset()            # list of 3 state arrays, 3 mask arrays
    states, masks, rewards, done = engine.step(actions)   # list of 3 ints

Usage
-----
    # 50 games, all seats use the model
    python selfplay.py --checkpoint mortal_3p.pth --games 50 --out selfplay.jsonl

    # 100 games; seat 2 plays randomly (adds exploration diversity)
    python selfplay.py --checkpoint mortal_3p.pth --games 100 --random-seats 2 --out selfplay.jsonl

    # Temperature sampling instead of argmax (more diverse actions)
    python selfplay.py --checkpoint mortal_3p.pth --games 200 --temperature 0.5 --out selfplay.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_converter import ACTION_SPACE_3P, OBS_CHANNELS_3P, N_TILES, make_legal_actions_mask
from model import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GameEngine protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GameEngine(Protocol):
    """Interface that a mahjong engine must implement for self-play."""

    def reset(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Start a new game and return initial observations.

        Returns
        -------
        states : list of 3 float32 arrays, each shaped ``(OBS_CHANNELS_3P, N_TILES)``
        masks  : list of 3 bool arrays, each shaped ``(ACTION_SPACE_3P,)``
        """
        ...

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float], bool]:
        """
        Apply one decision step and return next observations.

        Parameters
        ----------
        actions : action index (0..ACTION_SPACE_3P-1) for each of the 3 seats

        Returns
        -------
        states  : next states for each seat
        masks   : next legal-action masks for each seat
        rewards : per-seat reward for this step (0 for intermediate steps)
        done    : True when the game/kyoku is finished
        """
        ...


# ---------------------------------------------------------------------------
# Built-in synthetic engine (smoke-test placeholder)
# ---------------------------------------------------------------------------

class SimpleSelfPlayEnv:
    """
    Synthetic 3P game environment.

    States and outcomes are RANDOM – this is only a correctness smoke test.
    Replace with a real mahjong engine for meaningful RL training.
    """

    N_PLAYERS = 3
    # Average number of decisions per game before a terminal signal is issued.
    _STEPS_PER_GAME = 40

    def __init__(self, steps_per_game: int = _STEPS_PER_GAME):
        self._steps_per_game = steps_per_game
        self._step_count = 0

    def reset(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._step_count = 0
        states = [
            np.random.randn(OBS_CHANNELS_3P, N_TILES).astype(np.float32)
            for _ in range(self.N_PLAYERS)
        ]
        masks = [self._random_mask() for _ in range(self.N_PLAYERS)]
        return states, masks

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float], bool]:
        self._step_count += 1
        done = self._step_count >= self._steps_per_game

        states = [
            np.random.randn(OBS_CHANNELS_3P, N_TILES).astype(np.float32)
            for _ in range(self.N_PLAYERS)
        ]
        masks = [self._random_mask() for _ in range(self.N_PLAYERS)]

        if done:
            # Random final-score outcome: one player wins, two lose.
            winner = random.randint(0, self.N_PLAYERS - 1)
            rewards = [-5000.0] * self.N_PLAYERS
            rewards[winner] = 10000.0
        else:
            rewards = [0.0] * self.N_PLAYERS

        return states, masks, rewards, done

    @staticmethod
    def _random_mask() -> np.ndarray:
        can_discard = random.sample(range(34), k=random.randint(1, 13))
        return make_legal_actions_mask(
            can_discard=can_discard,
            can_agari=random.random() < 0.05,
            can_pon=random.random() < 0.1,
        )


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _argmax_policy(
    mortal,
    dqn,
    device: torch.device,
    state: np.ndarray,
    mask: np.ndarray,
    temperature: float = 0.0,
) -> int:
    """Model policy: argmax Q (temperature=0) or softmax sampling."""
    with torch.inference_mode():
        obs = torch.from_numpy(state.reshape(1, OBS_CHANNELS_3P, N_TILES)).float().to(device)
        m = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        phi = mortal(obs)
        q = dqn(phi, m).squeeze(0)  # (ACTION_SPACE_3P,)

    legal = mask.astype(bool)
    if not legal.any():
        return 77  # none

    if temperature <= 0.0:
        # Greedy
        q_np = q.cpu().float().numpy()
        q_np[~legal] = -1e9
        return int(q_np.argmax())
    else:
        # Temperature-scaled softmax over legal actions
        q_np = q.cpu().float().numpy()
        q_np[~legal] = -1e9
        q_np = q_np - q_np[legal].max()  # numerical stability
        exp_q = np.exp(q_np / temperature)
        exp_q[~legal] = 0.0
        probs = exp_q / exp_q.sum()
        return int(np.random.choice(ACTION_SPACE_3P, p=probs))


def _random_policy(mask: np.ndarray) -> int:
    """Uniform random policy over legal actions."""
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 77
    return int(np.random.choice(legal))


# ---------------------------------------------------------------------------
# Core self-play collector
# ---------------------------------------------------------------------------

class SelfPlayCollector:
    """
    Runs self-play episodes and accumulates training examples.

    Parameters
    ----------
    mortal, dqn : loaded model
    device      : compute device
    random_seats: set of seat indices that play randomly (adds exploration)
    epsilon     : probability of random action (ε-greedy; applies to model seats)
    temperature : softmax temperature for action sampling (0 = argmax)
    """

    def __init__(
        self,
        mortal,
        dqn,
        device: torch.device,
        random_seats: set[int] | None = None,
        epsilon: float = 0.05,
        temperature: float = 0.0,
    ):
        self.mortal = mortal
        self.dqn = dqn
        self.device = device
        self.random_seats = random_seats or set()
        self.epsilon = epsilon
        self.temperature = temperature

    def collect(
        self,
        engine: GameEngine,
        n_games: int,
        game_id_prefix: str = "sp",
    ) -> list[dict]:
        """
        Run *n_games* episodes and return all collected training examples.

        Each example is a dict compatible with the JSONL format expected by
        ``MahjongDataset3P``.
        """
        all_examples: list[dict] = []

        for game_idx in range(n_games):
            examples = self._run_episode(engine, game_id=f"{game_id_prefix}_{game_idx:06d}")
            all_examples.extend(examples)

            if (game_idx + 1) % max(1, n_games // 10) == 0:
                logger.info(
                    f"  Self-play progress: {game_idx + 1}/{n_games} games"
                    f"  ({len(all_examples)} examples so far)"
                )

        return all_examples

    def _run_episode(self, engine: GameEngine, game_id: str) -> list[dict]:
        """Run one episode and return examples for all seats."""
        states, masks = engine.reset()

        # Per-seat trajectory: list of (state, mask, action) per step
        trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(3)]

        step = 0
        done = False
        final_rewards: list[float] = [0.0, 0.0, 0.0]

        while not done:
            actions = []
            for seat in range(3):
                action = self._pick_action(seat, states[seat], masks[seat])
                actions.append(action)
                trajectories[seat].append((
                    states[seat].copy(),
                    masks[seat].copy(),
                    action,
                ))

            next_states, next_masks, rewards, done = engine.step(actions)

            # Accumulate rewards
            for seat in range(3):
                final_rewards[seat] += rewards[seat]

            states = next_states
            masks = next_masks
            step += 1

        # Build training examples: reward for each step = final outcome
        examples = []
        for seat in range(3):
            reward = float(final_rewards[seat])
            for t, (s, m, a) in enumerate(trajectories[seat]):
                examples.append({
                    "state": s.flatten().tolist(),
                    "legal_actions_mask": m.tolist(),
                    "action": a,
                    "reward": reward,
                    "meta": {
                        "game_id": game_id,
                        "player_id": seat,
                        "round": "?",
                        "step": t,
                        "is_3p": True,
                        "source": "selfplay",
                    },
                })

        return examples

    def _pick_action(self, seat: int, state: np.ndarray, mask: np.ndarray) -> int:
        if seat in self.random_seats:
            return _random_policy(mask)
        # ε-greedy
        if random.random() < self.epsilon:
            return _random_policy(mask)
        return _argmax_policy(
            self.mortal, self.dqn, self.device, state, mask,
            temperature=self.temperature,
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[dict], path: Path, append: bool = False) -> int:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")
    return len(examples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3P mahjong self-play training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to mortal_3p.pth (or 'random' to start from scratch)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of self-play games to run")
    parser.add_argument("--out", default="selfplay.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of overwriting")
    parser.add_argument("--random-seats", type=int, nargs="*", default=[],
                        help="Seat indices that play randomly (e.g. --random-seats 1 2)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="ε-greedy exploration rate for model seats")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Softmax temperature for action sampling (0=argmax)")
    parser.add_argument("--device", default="cpu",
                        help="Compute device (cpu / cuda)")
    parser.add_argument("--engine-module", default="",
                        help=(
                            "Python module path to a MahjongEngine class "
                            "(e.g. 'myengine.MahjongEngine'). "
                            "If omitted, uses the built-in synthetic engine."
                        ))
    parser.add_argument("--steps-per-game", type=int, default=40,
                        help="Steps per synthetic game (built-in engine only)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ------------------------------------------------------------------ Model
    if args.checkpoint.lower() == "random":
        logger.info("Using randomly initialised model (no checkpoint)")
        from model import build_model
        mortal, dqn = build_model(device=device)
        mortal.eval()
        dqn.eval()
    else:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        mortal, dqn = load_checkpoint(args.checkpoint, device=device)

    # ------------------------------------------------------------------ Engine
    if args.engine_module:
        # Dynamic import: "package.module.ClassName"
        parts = args.engine_module.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"--engine-module must be 'module.ClassName', got: {args.engine_module}"
            )
        mod = importlib.import_module(parts[0])
        engine_cls = getattr(mod, parts[1])
        engine: GameEngine = engine_cls()
        logger.info(f"Using custom engine: {args.engine_module}")
    else:
        engine = SimpleSelfPlayEnv(steps_per_game=args.steps_per_game)
        logger.info(
            "Using built-in synthetic engine "
            "(outcomes are random – replace for real RL training)"
        )

    # ------------------------------------------------------------------ Collect
    collector = SelfPlayCollector(
        mortal=mortal,
        dqn=dqn,
        device=device,
        random_seats=set(args.random_seats),
        epsilon=args.epsilon,
        temperature=args.temperature,
    )

    logger.info(
        f"Running {args.games} self-play games "
        f"(ε={args.epsilon}, T={args.temperature}, random_seats={args.random_seats})"
    )
    examples = collector.collect(engine, n_games=args.games)

    # ------------------------------------------------------------------ Write
    out_path = Path(args.out)
    n_written = write_jsonl(examples, out_path, append=args.append)
    logger.info(f"Written {n_written} examples to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
