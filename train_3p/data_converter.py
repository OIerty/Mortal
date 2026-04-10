"""
data_converter.py – Convert mahjong log files to 3P training examples.

Supported input formats
-----------------------
1. MJAI JSON-lines log  (one JSON object per line, or a list in a file)
2. mahjong_soul_api JSON log  (game record with ``actions`` list)
3. Simple bot action log  (JSON-lines with fields: ``state``, ``action``, ``legal``)

Output formats
--------------
* ``jsonl`` – newline-delimited JSON, one example per line:
    {
      "state":            [float, ...],   # OBS_CHANNELS_3P × 34 flattened
      "legal_actions_mask": [bool, ...],  # ACTION_SPACE_3P booleans
      "action":           int,            # action index
      "reward":           float,          # final round score delta (or 0)
      "meta": {
          "game_id":    str,
          "player_id":  int,
          "round":      str,
          "step":       int,
          "is_3p":      true
      }
    }
* ``npz``  – numpy archive with keys matching the jsonl fields
* ``pt``   – torch file (dict of tensors)

Action-index ↔ MJAI action mapping
------------------------------------
See ACTION_MAPPING at the bottom of this file, and the module-level constant
ACTION_SPACE_3P = 79.

    Index   MJAI type / tile       Notes
    ------  ---------------------  ----------------------------------------
    0-33    dahai tile_id           Normal discard; tile_id = TILE_ENCODING
    34-67   dahai tile_id + riichi  Riichi declaration + discard; tile_id - 34
    68      agari                   Covers tsumo AND ron (context-resolved)
    69      pon                     Pon call on last discard
    70      chi_low                 Chi: caller uses tiles lower (e.g. 3-4+5)
    71      chi_mid                 Chi: caller uses tiles mid  (e.g. 3+4-5)
    72      chi_high                Chi: caller uses tiles high (e.g. 3+4+5-6 wait)
    73      daiminkan               Open kan on opponent discard
    74      kakan                   Added kan (shouminkan)
    75      ankan                   Closed kan
    76      ryukyoku                Abortive / exhaustive draw declaration
    77      none                    Pass / no action
    78      nukidora                3P-specific: declare N tile as extra dora

Tile-id encoding (0-33):
    0-8   → 1m-9m  (man/character suit)
    9-17  → 1p-9p  (pin/circle suit)
    18-26 → 1s-9s  (sou/bamboo suit)
    27    → 1z (East)
    28    → 2z (South)
    29    → 3z (West)
    30    → 4z (North / nukidora tile)
    31    → 5z (Haku / White dragon)
    32    → 6z (Hatsu / Green dragon)
    33    → 7z (Chun / Red dragon)

Usage
-----
    python data_converter.py --input game.json --fmt mjai --output train.jsonl
    python data_converter.py --input game.json --fmt mjai --output data.npz --out-fmt npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of distinct tile types in Japanese mahjong
N_TILES = 34

# 3P observation: OBS_CHANNELS_3P × 34 tensor (see encode_state_3p for layout)
OBS_CHANNELS_3P = 54

# Number of distinct actions in the 3P action space
ACTION_SPACE_3P = 79

# Default starting score for 3P mahjong (most rule sets use 35000)
INITIAL_SCORE_3P = 35000.0

# Human-readable name for each action index
ACTION_NAMES: list[str] = (
    [f"dahai_{i}" for i in range(34)]        # 0-33
    + [f"riichi_dahai_{i}" for i in range(34)]  # 34-67
    + ["agari", "pon", "chi_low", "chi_mid", "chi_high",
       "daiminkan", "kakan", "ankan", "ryukyoku", "none", "nukidora"]  # 68-78
)

# ACTION_MAPPING: maps (mjai_type, extra_info) → action_index
# Used by the converter; referenced by model inference adapters.
ACTION_MAPPING: dict[tuple[str, Any], int] = {}

# Build ACTION_MAPPING
for _i in range(34):
    ACTION_MAPPING[("dahai", _i)] = _i
    ACTION_MAPPING[("riichi", _i)] = 34 + _i
ACTION_MAPPING[("agari", None)] = 68
ACTION_MAPPING[("pon", None)] = 69
ACTION_MAPPING[("chi_low", None)] = 70
ACTION_MAPPING[("chi_mid", None)] = 71
ACTION_MAPPING[("chi_high", None)] = 72
ACTION_MAPPING[("daiminkan", None)] = 73
ACTION_MAPPING[("kakan", None)] = 74
ACTION_MAPPING[("ankan", None)] = 75
ACTION_MAPPING[("ryukyoku", None)] = 76
ACTION_MAPPING[("none", None)] = 77
ACTION_MAPPING[("nukidora", None)] = 78

# Reverse mapping: action_index → (mjai_type, extra_info)
INV_ACTION_MAPPING: dict[int, tuple[str, Any]] = {v: k for k, v in ACTION_MAPPING.items()}

# ---------------------------------------------------------------------------
# Tile encoding helpers
# ---------------------------------------------------------------------------

# MJAI tile string → tile_id (0-33)
_MJAI_TO_ID: dict[str, int] = {}
for _n in range(1, 10):
    _MJAI_TO_ID[f"{_n}m"] = _n - 1          # 0-8
    _MJAI_TO_ID[f"{_n}mr"] = _n - 1         # red five treated same
    _MJAI_TO_ID[f"{_n}p"] = 9 + _n - 1      # 9-17
    _MJAI_TO_ID[f"{_n}pr"] = 9 + _n - 1
    _MJAI_TO_ID[f"{_n}s"] = 18 + _n - 1     # 18-26
    _MJAI_TO_ID[f"{_n}sr"] = 18 + _n - 1
for _n, _z in enumerate(["E", "S", "W", "N", "P", "F", "C"]):
    _MJAI_TO_ID[f"{_n + 1}z"] = 27 + _n     # 27-33
_MJAI_TO_ID["?"] = -1  # unknown tile

# tile_id → canonical MJAI tile string
_ID_TO_MJAI: dict[int, str] = {v: k for k, v in _MJAI_TO_ID.items() if not k.endswith("r") and k != "?"}


def tile_to_id(tile_str: str) -> int:
    """Return the tile-id (0-33) for a MJAI tile string, or -1 for unknown."""
    return _MJAI_TO_ID.get(tile_str, -1)


def id_to_tile(tile_id: int) -> str:
    """Return the canonical MJAI tile string for a tile-id."""
    return _ID_TO_MJAI.get(tile_id, "?")


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

def encode_state_3p(
    hand: list[int],             # tile-ids in own hand (may include -1 for unknown)
    discards: list[list[int]],   # discards[player_idx] = list of tile-ids
    melds: list[list[list[int]]],# melds[player_idx] = list of meld (each meld is a list of tile-ids)
    dora_indicators: list[int],  # tile-ids of dora-indicator tiles
    remaining: int,              # tiles remaining in the wall
    riichi: list[bool],          # riichi/tenpai state per player
    scores: list[float],         # scores per player (normalized by 25000)
    round_wind: int,             # 0=East, 1=South
    seat_wind: int,              # own seat wind 0=East, 1=South, 2=West
    kyoku: int,                  # round number (0-indexed: 0=E1, 1=E2, …)
    honba: int,                  # honba count
    kyotaku: int,                # riichi sticks on table
    nukidora_counts: list[int],  # nukidora (N tile) count per player
    last_discard_tile: int = -1, # tile-id of last discarded tile (-1 if none)
    current_player: int = 0,     # which player is acting (0, 1, 2)
) -> np.ndarray:
    """
    Encode a 3-player mahjong game state into an (OBS_CHANNELS_3P, 34) float32 array.

    Channel layout (54 channels total):
        0-3   : own hand binary counts (≥1, ≥2, ≥3, ≥4 copies per tile type)
        4-7   : own discard history (4 temporal buckets of ~6 tiles each)
        8-11  : opponent-1 discard history (4 buckets)
        12-15 : opponent-2 discard history (4 buckets)
        16-18 : own melds (up to 3 active melds, each as tile-presence)
        19-21 : opponent-1 melds
        22-24 : opponent-2 melds
        25-29 : dora indicators (up to 5 slots)
        30    : remaining-tile count (normalized, broadcast across 34 positions)
        31-33 : riichi/tenpai state (self, opp1, opp2; broadcast)
        34-36 : score (self, opp1, opp2; normalized by 25000; broadcast)
        37    : round wind (broadcast)
        38    : seat wind (broadcast)
        39    : kyoku / honba info (broadcast)
        40-42 : nukidora count (self, opp1, opp2; broadcast)
        43    : last-discard tile (one-hot over 34 positions)
        44    : current player acting (broadcast)
        45-53 : reserved / padding (zeros)
    """
    obs = np.zeros((OBS_CHANNELS_3P, N_TILES), dtype=np.float32)

    # Helper: count copies of each tile in a list
    def _count_tiles(tile_list: list[int]) -> np.ndarray:
        counts = np.zeros(N_TILES, dtype=np.float32)
        for t in tile_list:
            if 0 <= t < N_TILES:
                counts[t] += 1.0
        return counts

    # --- Channels 0-3: own hand ---
    hand_counts = _count_tiles(hand)
    for c in range(4):
        obs[c] = (hand_counts >= c + 1).astype(np.float32)

    # --- Channels 4-15: discard history (4 buckets per player) ---
    for p_rel, p_discards in enumerate(discards[:3]):
        base_ch = 4 + p_rel * 4
        bucket_size = max(1, len(p_discards) // 4 + 1)
        for b in range(4):
            start = b * bucket_size
            end = min(start + bucket_size, len(p_discards))
            bucket = p_discards[start:end]
            obs[base_ch + b] = _count_tiles(bucket).clip(0, 1)

    # --- Channels 16-24: meld data (3 meld slots per player) ---
    for p_rel, p_melds in enumerate(melds[:3]):
        base_ch = 16 + p_rel * 3
        for m_idx, meld_tiles in enumerate(p_melds[:3]):
            obs[base_ch + m_idx] = _count_tiles(meld_tiles).clip(0, 1)

    # --- Channels 25-29: dora indicators ---
    for d_idx, dora in enumerate(dora_indicators[:5]):
        if 0 <= dora < N_TILES:
            obs[25 + d_idx, dora] = 1.0

    # --- Channel 30: remaining tiles (broadcast) ---
    obs[30, :] = float(remaining) / 70.0  # ~70 tiles remain at game start in 3P

    # --- Channels 31-33: riichi/tenpai (broadcast) ---
    for p_rel, r in enumerate(riichi[:3]):
        obs[31 + p_rel, :] = float(r)

    # --- Channels 34-36: score (broadcast) ---
    for p_rel, s in enumerate(scores[:3]):
        obs[34 + p_rel, :] = float(s) / 25000.0

    # --- Channel 37: round wind (broadcast) ---
    obs[37, :] = float(round_wind)

    # --- Channel 38: seat wind (broadcast) ---
    obs[38, :] = float(seat_wind) / 2.0

    # --- Channel 39: kyoku / honba (broadcast) ---
    obs[39, :] = (float(kyoku) + float(honba) * 0.1 + float(kyotaku) * 0.01) / 8.0

    # --- Channels 40-42: nukidora (broadcast) ---
    for p_rel, nd in enumerate(nukidora_counts[:3]):
        obs[40 + p_rel, :] = float(nd) / 4.0

    # --- Channel 43: last discard tile (one-hot) ---
    if 0 <= last_discard_tile < N_TILES:
        obs[43, last_discard_tile] = 1.0

    # --- Channel 44: current player (broadcast) ---
    obs[44, :] = float(current_player) / 2.0

    # channels 45-53 remain zero (reserved)
    return obs


def make_legal_actions_mask(
    can_discard: list[int],
    can_riichi: list[int] | None = None,
    can_agari: bool = False,
    can_pon: bool = False,
    can_chi_low: bool = False,
    can_chi_mid: bool = False,
    can_chi_high: bool = False,
    can_daiminkan: bool = False,
    can_kakan: bool = False,
    can_ankan: bool = False,
    can_ryukyoku: bool = False,
    can_nukidora: bool = False,
) -> np.ndarray:
    """
    Return a boolean mask of shape (ACTION_SPACE_3P,) marking legal actions.

    Parameters
    ----------
    can_discard : list of tile-ids that can be discarded normally
    can_riichi  : list of tile-ids that trigger a riichi declaration on discard
    can_agari   : whether agari (win) is available
    can_pon     : whether pon call is available
    can_chi_*   : whether chi variants are available
    can_daiminkan / kakan / ankan: kan variants
    can_ryukyoku : whether ryukyoku declaration is available
    can_nukidora : 3P only – whether the player holds a North tile to declare
    """
    mask = np.zeros(ACTION_SPACE_3P, dtype=bool)
    if can_riichi is None:
        can_riichi = []
    for t in can_discard:
        if 0 <= t < N_TILES:
            mask[t] = True
    for t in can_riichi:
        if 0 <= t < N_TILES:
            mask[34 + t] = True
    if can_agari:
        mask[68] = True
    if can_pon:
        mask[69] = True
    if can_chi_low:
        mask[70] = True
    if can_chi_mid:
        mask[71] = True
    if can_chi_high:
        mask[72] = True
    if can_daiminkan:
        mask[73] = True
    if can_kakan:
        mask[74] = True
    if can_ankan:
        mask[75] = True
    if can_ryukyoku:
        mask[76] = True
    if not mask.any():
        mask[77] = True  # none is always a fallback
    if can_nukidora:
        mask[78] = True
    return mask


# ---------------------------------------------------------------------------
# MJAI log parser
# ---------------------------------------------------------------------------

class GameState3P:
    """Accumulates MJAI events to track 3P game state for state encoding."""

    N_PLAYERS = 3

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.reset()

    def reset(self):
        self.hand: list[int] = []
        self.discards: list[list[int]] = [[] for _ in range(self.N_PLAYERS)]
        self.melds: list[list[list[int]]] = [[] for _ in range(self.N_PLAYERS)]
        self.dora_indicators: list[int] = []
        self.remaining: int = 70
        self.riichi: list[bool] = [False] * self.N_PLAYERS
        self.scores: list[float] = [INITIAL_SCORE_3P] * self.N_PLAYERS
        self.round_wind: int = 0
        self.seat_wind: int = 0
        self.kyoku: int = 0
        self.honba: int = 0
        self.kyotaku: int = 0
        self.nukidora_counts: list[int] = [0] * self.N_PLAYERS
        self.last_discard_tile: int = -1
        self.step: int = 0
        self.round_str: str = "E1"

    def apply_event(self, event: dict):
        """Update internal state from a single MJAI event dict."""
        t = event.get("type", "")

        if t == "start_kyoku":
            self.round_wind = {"E": 0, "S": 1, "W": 2, "N": 3}.get(
                event.get("bakaze", "E"), 0
            )
            self.kyoku = event.get("kyoku", 1) - 1
            self.honba = event.get("honba", 0)
            self.kyotaku = event.get("kyotaku", 0)
            oya = event.get("oya", 0)
            # Rotate so seat_wind is relative to self
            self.seat_wind = (self.player_id - oya) % self.N_PLAYERS
            self.scores = [float(s) for s in event.get("scores", [INITIAL_SCORE_3P] * 4)[:3]]
            tehais = event.get("tehais", [[] for _ in range(self.N_PLAYERS)])
            self.hand = [tile_to_id(t) for t in tehais[self.player_id]
                         if tile_to_id(t) >= 0]
            self.discards = [[] for _ in range(self.N_PLAYERS)]
            self.melds = [[] for _ in range(self.N_PLAYERS)]
            self.riichi = [False] * self.N_PLAYERS
            self.nukidora_counts = [0] * self.N_PLAYERS
            self.remaining = 70
            self.last_discard_tile = -1
            dora_marker = event.get("dora_marker", "?")
            dora_id = tile_to_id(dora_marker)
            self.dora_indicators = [dora_id] if dora_id >= 0 else []
            bakaze = event.get("bakaze", "E")
            self.round_str = f"{bakaze}{self.kyoku + 1}"

        elif t == "dora":
            dora_marker = event.get("dora_marker", "?")
            dora_id = tile_to_id(dora_marker)
            if dora_id >= 0:
                self.dora_indicators.append(dora_id)

        elif t == "tsumo":
            actor = event.get("actor", 0)
            pai = event.get("pai", "?")
            self.remaining = max(0, self.remaining - 1)
            if actor == self.player_id:
                tid = tile_to_id(pai)
                if tid >= 0:
                    self.hand.append(tid)

        elif t == "dahai":
            actor = event.get("actor", 0)
            pai = event.get("pai", "?")
            tid = tile_to_id(pai)
            if 0 <= actor < self.N_PLAYERS:
                self.discards[actor].append(tid)
                if actor == self.player_id and tid in self.hand:
                    self.hand.remove(tid)
            self.last_discard_tile = tid

        elif t == "riichi":
            actor = event.get("actor", 0)
            if 0 <= actor < self.N_PLAYERS:
                self.riichi[actor] = True

        elif t in ("chi", "pon", "daiminkan", "kakan", "ankan"):
            actor = event.get("actor", 0)
            consumed = [tile_to_id(p) for p in event.get("consumed", [])]
            pai = tile_to_id(event.get("pai", "?"))
            meld = [pai] + consumed if pai >= 0 else consumed
            if 0 <= actor < self.N_PLAYERS:
                self.melds[actor].append([x for x in meld if x >= 0])
                if actor == self.player_id:
                    for c in consumed:
                        if c in self.hand:
                            self.hand.remove(c)

        elif t == "nukidora":
            actor = event.get("actor", 0)
            if 0 <= actor < self.N_PLAYERS:
                self.nukidora_counts[actor] += 1
                if actor == self.player_id:
                    north_id = tile_to_id("4z")
                    if north_id in self.hand:
                        self.hand.remove(north_id)

        self.step += 1

    def encode(self) -> np.ndarray:
        """Return encoded state as (OBS_CHANNELS_3P, 34) float32 array."""
        return encode_state_3p(
            hand=self.hand,
            discards=self.discards,
            melds=self.melds,
            dora_indicators=self.dora_indicators,
            remaining=self.remaining,
            riichi=self.riichi,
            scores=self.scores,
            round_wind=self.round_wind,
            seat_wind=self.seat_wind,
            kyoku=self.kyoku,
            honba=self.honba,
            kyotaku=self.kyotaku,
            nukidora_counts=self.nukidora_counts,
            last_discard_tile=self.last_discard_tile,
            current_player=self.player_id,
        )


def _event_to_action_index(event: dict, player_id: int) -> int | None:
    """
    Map a MJAI event dict that represents an **actor == player_id** action to
    an action index.  Returns None if the event is not an action or is not
    by the specified player.
    """
    t = event.get("type", "")
    actor = event.get("actor", -1)

    if actor != player_id and t not in ("agari",):
        return None

    if t == "dahai":
        tid = tile_to_id(event.get("pai", "?"))
        if tid < 0:
            return None
        return ACTION_MAPPING.get(("dahai", tid))

    if t == "riichi":
        # riichi action itself is followed by a dahai; we encode both together
        # by waiting for the subsequent dahai event.
        return None  # handled in converter loop

    if t == "agari":
        return ACTION_MAPPING[("agari", None)]

    if t == "pon":
        return ACTION_MAPPING[("pon", None)]

    if t == "chi":
        # Determine chi variant from consumed vs pai
        pai = tile_to_id(event.get("pai", "?"))
        consumed = sorted([tile_to_id(p) for p in event.get("consumed", [])])
        if len(consumed) == 2 and pai >= 0:
            c0, c1 = consumed
            p = pai
            if p == c0 + 2 and p == c1 + 1:   # low wait: p is highest
                return ACTION_MAPPING[("chi_low", None)]
            elif c0 < p < c1:                   # mid wait
                return ACTION_MAPPING[("chi_mid", None)]
            else:                               # high wait: p is lowest
                return ACTION_MAPPING[("chi_high", None)]
        return ACTION_MAPPING[("chi_low", None)]

    if t == "daiminkan":
        return ACTION_MAPPING[("daiminkan", None)]
    if t == "kakan":
        return ACTION_MAPPING[("kakan", None)]
    if t == "ankan":
        return ACTION_MAPPING[("ankan", None)]
    if t == "ryukyoku":
        return ACTION_MAPPING[("ryukyoku", None)]
    if t == "none":
        return ACTION_MAPPING[("none", None)]
    if t == "nukidora":
        return ACTION_MAPPING[("nukidora", None)]

    return None


# ---------------------------------------------------------------------------
# MJAI log converter
# ---------------------------------------------------------------------------

def _load_events(path: Path) -> list[dict]:
    """Load a MJAI log file as a flat list of event dicts."""
    text = path.read_text(encoding="utf-8")
    # Try whole-file JSON first (list or {"log": [...]} wrapper)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "log" in data:
            return data["log"]
        # Unknown single-object format – fall through to JSONL
    except json.JSONDecodeError:
        pass
    # Newline-delimited JSON (jsonl)
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _hand_tile_ids(
    state: "GameState3P",
    discarded_tid: int,
) -> list[int]:
    """
    Return the tile-ids that were in hand just before this dahai event.

    ``state.hand`` has already been updated (tile removed by apply_event), so
    we reconstruct the pre-action hand by adding the discarded tile back.
    """
    hand = list(state.hand) + [discarded_tid]
    return sorted(set(t for t in hand if t >= 0))


def _call_event_mask(event_type: str, action_idx: int) -> "np.ndarray":
    """
    Build a legal-actions mask for a call / special event.

    For all call-type events the player's only choices were to make the call
    *or* pass (none / 77).  We mark both as legal so the network sees a
    meaningful contrast.
    """
    mask = np.zeros(ACTION_SPACE_3P, dtype=bool)
    mask[action_idx] = True
    mask[77] = True  # 'none' (pass) is always an alternative
    return mask


def convert_mjai_log(
    path: Path,
    game_id: str = "",
) -> Iterator[dict]:
    """
    Parse a MJAI log file and yield one training example dict per decision
    point for all players.
    """
    events = _load_events(path)

    # Determine number of players
    n_players = 3
    for evt in events:
        if evt.get("type") == "start_game":
            names = evt.get("names", [])
            n_players = len(names)
            break
    if n_players != 3:
        # Only process 3P games
        return

    # Final scores (for reward computation).
    # Primary source: end_game event with a "scores" field.
    # Fallback: last observed scores from the final agari/ryukyoku/end_kyoku event.
    final_scores: list[float] | None = None
    last_seen_scores: list[float] = [INITIAL_SCORE_3P] * 3
    for evt in events:
        t_e = evt.get("type", "")
        if t_e in ("agari", "ryukyoku", "end_kyoku") and "scores" in evt:
            last_seen_scores = [float(s) for s in evt["scores"][:3]]
        elif t_e == "end_game":
            if "scores" in evt:
                final_scores = [float(s) for s in evt["scores"][:3]]
            break
    if final_scores is None:
        final_scores = last_seen_scores

    initial_scores: list[float] = [INITIAL_SCORE_3P] * 3
    for evt in events:
        if evt.get("type") == "start_kyoku":
            initial_scores = [float(s) for s in evt.get("scores", [INITIAL_SCORE_3P] * 3)[:3]]
            break

    riichi_pending: list[bool] = [False, False, False]

    for player_id in range(n_players):
        state = GameState3P(player_id)
        step_global = 0
        riichi_pending_local: dict[int, bool] = {}

        for evt in events:
            t = evt.get("type", "")
            actor = evt.get("actor", -1)

            # Capture pre-action observation BEFORE modifying state.
            # This ensures the (state → action) pair is correctly aligned:
            # the observation reflects what the player saw when they acted.
            pre_obs = state.encode()

            # Apply state update
            state.apply_event(evt)
            step_global += 1

            action_idx: int | None = None

            if t == "riichi" and actor == player_id:
                riichi_pending_local[actor] = True
                continue  # wait for the following dahai

            if t == "dahai" and actor == player_id:
                tid = tile_to_id(evt.get("pai", "?"))
                if tid < 0:
                    riichi_pending_local.pop(actor, None)
                    continue
                is_riichi = riichi_pending_local.pop(actor, False)
                if is_riichi:
                    action_idx = ACTION_MAPPING.get(("riichi", tid))
                    # Legal mask: riichi-discard for every tile in the pre-action hand
                    hand_ids = _hand_tile_ids(state, tid)
                    mask = make_legal_actions_mask(
                        can_discard=[],
                        can_riichi=hand_ids,
                    )
                else:
                    action_idx = ACTION_MAPPING.get(("dahai", tid))
                    # Legal mask: any tile that was in hand can be discarded
                    hand_ids = _hand_tile_ids(state, tid)
                    mask = make_legal_actions_mask(can_discard=hand_ids)

            elif t in ("pon", "chi", "daiminkan", "kakan", "ankan",
                       "agari", "ryukyoku", "nukidora", "none"):
                if t == "agari":
                    # Agari can be multi-player (multi-ron) or tsumo
                    winners = evt.get("who", [actor])
                    if isinstance(winners, int):
                        winners = [winners]
                    if player_id in winners:
                        action_idx = ACTION_MAPPING[("agari", None)]
                        # Both agari and none (pass) are available
                        mask = make_legal_actions_mask(can_agari=True)
                elif actor == player_id:
                    action_idx = _event_to_action_index(evt, player_id)
                    if action_idx is not None:
                        mask = _call_event_mask(t, action_idx)

            if action_idx is None:
                continue

            reward = float(final_scores[player_id] - initial_scores[player_id])

            yield {
                "state": pre_obs.flatten().tolist(),
                "legal_actions_mask": mask.tolist(),
                "action": int(action_idx),
                "reward": reward,
                "meta": {
                    "game_id": game_id or path.stem,
                    "player_id": player_id,
                    "round": state.round_str,
                    "step": step_global,
                    "is_3p": True,
                },
            }


# ---------------------------------------------------------------------------
# mahjong_soul_api log converter
# ---------------------------------------------------------------------------

def convert_majsoul_log(
    path: Path,
    game_id: str = "",
) -> Iterator[dict]:
    """
    Convert a mahjong_soul_api game record to 3P training examples.

    The mahjong_soul_api JSON has a nested ``actions`` list; this function
    normalises the structure to a flat MJAI event list and delegates to
    :func:`convert_mjai_log`.

    Supported schemas:
    * ``{"actions": [...]}``  (direct actions list)
    * ``{"data": {"actions": [...]}}``  (wrapped)
    * A list of dicts with "action_type" / "result" fields (raw proto-JSON)
    """
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    if isinstance(data, list):
        actions = data
    elif isinstance(data, dict):
        actions = data.get("actions", data.get("data", {}).get("actions", []))
    else:
        return

    # Convert actions to approximate MJAI events
    mjai_events: list[dict] = []
    for action in actions:
        evt = _majsoul_action_to_mjai(action)
        if evt:
            mjai_events.extend(evt if isinstance(evt, list) else [evt])

    if not mjai_events:
        return

    # Serialise the normalised event list to a cross-platform temporary file
    # and delegate to convert_mjai_log.
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as tmp_f:
        json.dump(mjai_events, tmp_f)
        tmp_path = Path(tmp_f.name)

    try:
        yield from convert_mjai_log(tmp_path, game_id=game_id or path.stem)
    finally:
        tmp_path.unlink(missing_ok=True)


def _majsoul_action_to_mjai(action: dict) -> list[dict] | dict | None:
    """
    Best-effort conversion of a mahjong_soul_api action dict to MJAI event(s).
    Only handles the most common action types.
    """
    atype = action.get("action_type", action.get("type", ""))
    if not atype:
        return None

    # Mapping of soul action names to MJAI types
    type_map = {
        "ActionMJStart": lambda a: {"type": "start_game",
                                    "names": ["0", "1", "2"],
                                    "id": 0},
        "ActionNewRound": lambda a: {"type": "start_kyoku",
                                     "bakaze": a.get("chang", "E"),
                                     "kyoku": a.get("ju", 1),
                                     "honba": a.get("ben", 0),
                                     "kyotaku": a.get("liqibang", 0),
                                     "oya": a.get("ju", 1) - 1,
                                     "scores": a.get("scores", [35000] * 3),
                                     "dora_marker": a.get("doras", ["?"])[0]
                                                    if a.get("doras") else "?",
                                     "tehais": [[] for _ in range(3)]},
        "ActionDealTile": lambda a: {"type": "tsumo",
                                     "actor": a.get("seat", 0),
                                     "pai": a.get("tile", "?")},
        "ActionDiscardTile": lambda a: {"type": "dahai",
                                        "actor": a.get("seat", 0),
                                        "pai": a.get("tile", "?"),
                                        "tsumogiri": a.get("moqie", False)},
        "ActionChiPengGang": lambda a: {"type": "pon",
                                        "actor": a.get("seat", 0),
                                        "consumed": a.get("tiles", [])},
        "ActionAnGangAddGang": lambda a: {"type": "kakan",
                                          "actor": a.get("seat", 0),
                                          "pai": a.get("tiles", ["?"])[0]},
        "ActionHu": lambda a: {"type": "agari",
                               "actor": a.get("seat", 0),
                               "who": a.get("seat", 0)},
        "ActionLiuJu": lambda a: {"type": "ryukyoku"},
        "ActionBabei": lambda a: {"type": "nukidora",
                                  "actor": a.get("seat", 0),
                                  "pai": "4z"},
    }
    fn = type_map.get(atype)
    if fn:
        return fn(action)
    return None


# ---------------------------------------------------------------------------
# Bot action log converter
# ---------------------------------------------------------------------------

def convert_bot_log(path: Path, game_id: str = "") -> Iterator[dict]:
    """
    Convert a simple bot action log (jsonl) to training examples.

    Each line must be a JSON object with:
        state  : list[float]  – flat state vector of length OBS_CHANNELS_3P*34
        action : int          – action index (0..ACTION_SPACE_3P-1)
        legal  : list[bool]   – legal actions mask
        reward : float        – (optional) round reward
    """
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = obj.get("state", [])
            action = int(obj.get("action", 0))
            legal = obj.get("legal", [True] * ACTION_SPACE_3P)
            reward = float(obj.get("reward", 0.0))
            if len(state) != OBS_CHANNELS_3P * N_TILES:
                continue
            yield {
                "state": state,
                "legal_actions_mask": legal,
                "action": action,
                "reward": reward,
                "meta": {
                    "game_id": game_id or path.stem,
                    "player_id": obj.get("player_id", 0),
                    "round": obj.get("round", "E1"),
                    "step": line_no,
                    "is_3p": True,
                },
            }


# ---------------------------------------------------------------------------
# Writer utilities
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[dict], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


def write_npz(examples: list[dict], out_path: Path):
    states = np.array([ex["state"] for ex in examples], dtype=np.float32)
    masks = np.array([ex["legal_actions_mask"] for ex in examples], dtype=bool)
    actions = np.array([ex["action"] for ex in examples], dtype=np.int64)
    rewards = np.array([ex["reward"] for ex in examples], dtype=np.float32)
    np.savez_compressed(
        out_path,
        state=states,
        legal_actions_mask=masks,
        action=actions,
        reward=rewards,
    )


def write_pt(examples: list[dict], out_path: Path):
    import torch
    states = torch.tensor([ex["state"] for ex in examples], dtype=torch.float32)
    masks = torch.tensor([ex["legal_actions_mask"] for ex in examples], dtype=torch.bool)
    actions = torch.tensor([ex["action"] for ex in examples], dtype=torch.long)
    rewards = torch.tensor([ex["reward"] for ex in examples], dtype=torch.float32)
    torch.save(
        {"state": states, "legal_actions_mask": masks,
         "action": actions, "reward": rewards},
        out_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Convert 3P mahjong logs to training examples"
    )
    parser.add_argument("--input", required=True, help="Input log file path")
    parser.add_argument(
        "--fmt",
        choices=["mjai", "majsoul", "bot"],
        default="mjai",
        help="Input format (default: mjai)",
    )
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--out-fmt",
        choices=["jsonl", "npz", "pt"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument("--game-id", default="", help="Optional game identifier")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    converters = {
        "mjai": convert_mjai_log,
        "majsoul": convert_majsoul_log,
        "bot": convert_bot_log,
    }
    gen = converters[args.fmt](in_path, game_id=args.game_id)
    examples = list(gen)
    print(f"Converted {len(examples)} examples from {in_path}", file=sys.stderr)

    if args.out_fmt == "jsonl":
        write_jsonl(examples, out_path)
    elif args.out_fmt == "npz":
        write_npz(examples, out_path)
    elif args.out_fmt == "pt":
        write_pt(examples, out_path)

    print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
