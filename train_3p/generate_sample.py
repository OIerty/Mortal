"""
generate_sample.py – Generate a synthetic MJAI log and training sample file.

Produces:
  * sample_game.json  – A minimal synthetic MJAI log (3-player)
  * sample.jsonl      – 20 training examples extracted from the log

These are used for smoke tests:

    python generate_sample.py
    python train.py --data sample.jsonl --epochs 1

The generated game is intentionally simple and does NOT represent realistic
mahjong play; it is only meant to verify that the pipeline runs end-to-end
without errors.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_converter import (
    N_TILES, OBS_CHANNELS_3P, ACTION_SPACE_3P,
    encode_state_3p, make_legal_actions_mask,
    ACTION_MAPPING, ACTION_NAMES, id_to_tile, tile_to_id,
)

random.seed(42)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def _deal_hand(tiles: list[int], n: int = 13) -> list[int]:
    """Pop n tiles from the wall and return them as a hand."""
    hand = tiles[:n]
    del tiles[:n]
    return hand


def _tile_id_to_mjai(tid: int) -> str:
    return id_to_tile(tid)


# ---------------------------------------------------------------------------
# Synthetic game generator
# ---------------------------------------------------------------------------

def generate_mjai_game(n_moves: int = 30) -> list[dict]:
    """
    Generate a minimal synthetic 3-player MJAI event list.

    The game plays through a single kyoku (round) with:
    * One start_game
    * One start_kyoku
    * n_moves tsumo+dahai pairs (round-robin across 3 players)
    * One end_kyoku (ryukyoku)
    * One end_game
    """
    events: list[dict] = []

    # Full wall: 4 copies of each tile (0-33), minus the 4 North (tile 30) tiles
    # In 3P mahjong, North tiles are removed from the draw pile.
    wall: list[int] = []
    for tid in range(N_TILES):
        copies = 4 if tid != 30 else 0  # no North in 3P draw
        wall.extend([tid] * copies)
    random.shuffle(wall)

    # Deal hands: 13 tiles each
    hands = [_deal_hand(wall) for _ in range(3)]

    # Dora indicator
    dora_id = wall.pop(0) if wall else 0
    dora_marker = _tile_id_to_mjai(dora_id)

    scores = [35000, 35000, 35000]

    events.append({"type": "start_game", "names": ["player0", "player1", "player2"], "id": 0})
    events.append({
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": scores,
        "dora_marker": dora_marker,
        "tehais": [
            [_tile_id_to_mjai(t) for t in hands[0]],
            ["?"] * 13,
            ["?"] * 13,
        ],
    })

    current_player = 0
    for move in range(n_moves):
        if not wall:
            break

        # Tsumo
        drawn = wall.pop(0)
        tile_str = _tile_id_to_mjai(drawn)
        events.append({"type": "tsumo", "actor": current_player, "pai": tile_str})

        if current_player == 0:
            hands[0].append(drawn)

        # Discard (choose a random tile from hand, or the drawn tile)
        if current_player == 0 and hands[0]:
            discard_tid = random.choice(hands[0])
            hands[0].remove(discard_tid)
        else:
            discard_tid = drawn

        discard_str = _tile_id_to_mjai(discard_tid)
        events.append({
            "type": "dahai",
            "actor": current_player,
            "pai": discard_str,
            "tsumogiri": discard_tid == drawn,
        })

        current_player = (current_player + 1) % 3

    # End kyoku
    events.append({
        "type": "ryukyoku",
        "reason": "fanpai",
        "tehais": [
            [_tile_id_to_mjai(t) for t in hands[0]],
            ["?"] * 13,
            ["?"] * 13,
        ],
        "tenpais": [False, False, False],
        "scores": scores,
        "deltas": [0, 0, 0],
    })
    events.append({"type": "end_game"})

    return events


# ---------------------------------------------------------------------------
# Direct synthetic example generator (no log parsing required)
# ---------------------------------------------------------------------------

def generate_synthetic_examples(n: int = 20) -> list[dict]:
    """
    Generate *n* synthetic training examples with valid state/mask/action/reward.

    This bypasses the MJAI parser and directly calls encode_state_3p for speed.
    """
    examples = []
    for i in range(n):
        # Random hand (13 tiles)
        hand = random.sample(range(N_TILES), min(13, N_TILES))

        # Random discards (0-8 tiles per player)
        discards = [
            random.sample(range(N_TILES), random.randint(0, 8))
            for _ in range(3)
        ]

        # Random melds (0-2 per player)
        melds = [
            [[random.randint(0, N_TILES - 1) for _ in range(3)]
             for _ in range(random.randint(0, 2))]
            for _ in range(3)
        ]

        dora_indicators = random.sample(range(N_TILES), random.randint(1, 3))
        remaining = random.randint(5, 70)
        riichi = [random.random() < 0.1 for _ in range(3)]
        scores = [random.uniform(0, 70000) for _ in range(3)]
        round_wind = random.randint(0, 1)
        seat_wind = random.randint(0, 2)
        kyoku = random.randint(0, 3)
        honba = random.randint(0, 3)
        kyotaku = random.randint(0, 3)
        nukidora_counts = [random.randint(0, 2) for _ in range(3)]
        last_discard = random.choice([-1] + list(range(N_TILES)))

        obs = encode_state_3p(
            hand=hand,
            discards=discards,
            melds=melds,
            dora_indicators=dora_indicators,
            remaining=remaining,
            riichi=riichi,
            scores=scores,
            round_wind=round_wind,
            seat_wind=seat_wind,
            kyoku=kyoku,
            honba=honba,
            kyotaku=kyotaku,
            nukidora_counts=nukidora_counts,
            last_discard_tile=last_discard,
        )

        # Legal actions: can discard any tile in hand
        can_discard = [t for t in hand if t < N_TILES]
        can_riichi = can_discard[:1] if random.random() < 0.15 else []
        can_agari = random.random() < 0.05
        can_nukidora = (tile_to_id("4z") in hand) and random.random() < 0.5

        mask = make_legal_actions_mask(
            can_discard=can_discard or list(range(34)),  # always allow some discard
            can_riichi=can_riichi,
            can_agari=can_agari,
            can_pon=random.random() < 0.1,
            can_chi_low=random.random() < 0.05,
            can_chi_mid=random.random() < 0.05,
            can_chi_high=random.random() < 0.05,
            can_nukidora=can_nukidora,
        )

        # Choose a random legal action
        legal_indices = mask.nonzero()[0].tolist()
        action = int(random.choice(legal_indices)) if legal_indices else 77

        reward = random.uniform(-30000, 30000)

        examples.append({
            "state": obs.flatten().tolist(),
            "legal_actions_mask": mask.tolist(),
            "action": action,
            "reward": float(reward),
            "meta": {
                "game_id": f"synthetic_{i:04d}",
                "player_id": 0,
                "round": f"E{kyoku + 1}",
                "step": i,
                "is_3p": True,
            },
        })

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3P training data for smoke tests",
    )
    parser.add_argument("--n", type=int, default=20,
                        help="Number of training examples to generate")
    parser.add_argument("--out", default="sample.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--game-log", default="",
                        help="Also write a synthetic MJAI game log to this path")
    args = parser.parse_args(argv)

    examples = generate_synthetic_examples(args.n)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")
    print(f"Written {len(examples)} examples to {out_path}", file=sys.stderr)

    if args.game_log:
        events = generate_mjai_game()
        game_path = Path(args.game_log)
        game_path.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Written MJAI game log ({len(events)} events) to {game_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
