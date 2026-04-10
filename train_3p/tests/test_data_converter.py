"""
tests/test_data_converter.py – Unit tests for data_converter.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow running from the train_3p directory or repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_converter import (
    N_TILES,
    OBS_CHANNELS_3P,
    ACTION_SPACE_3P,
    ACTION_NAMES,
    ACTION_MAPPING,
    INV_ACTION_MAPPING,
    tile_to_id,
    id_to_tile,
    encode_state_3p,
    make_legal_actions_mask,
    GameState3P,
    convert_mjai_log,
    convert_bot_log,
    write_jsonl,
    write_npz,
)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

class TestTileHelpers:
    def test_tile_to_id_man(self):
        assert tile_to_id("1m") == 0
        assert tile_to_id("9m") == 8

    def test_tile_to_id_pin(self):
        assert tile_to_id("1p") == 9
        assert tile_to_id("9p") == 17

    def test_tile_to_id_sou(self):
        assert tile_to_id("1s") == 18
        assert tile_to_id("9s") == 26

    def test_tile_to_id_honors(self):
        assert tile_to_id("1z") == 27  # East
        assert tile_to_id("4z") == 30  # North
        assert tile_to_id("7z") == 33  # Chun

    def test_tile_to_id_red_five(self):
        assert tile_to_id("5mr") == tile_to_id("5m")
        assert tile_to_id("5pr") == tile_to_id("5p")

    def test_tile_to_id_unknown(self):
        assert tile_to_id("?") == -1
        assert tile_to_id("") == -1

    def test_id_to_tile_roundtrip(self):
        for i in range(N_TILES):
            mjai = id_to_tile(i)
            assert tile_to_id(mjai) == i, f"Roundtrip failed for tile {i}: {mjai}"

    def test_n_tiles(self):
        assert N_TILES == 34


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

class TestActionMapping:
    def test_action_space_size(self):
        assert ACTION_SPACE_3P == 79

    def test_action_names_length(self):
        assert len(ACTION_NAMES) == ACTION_SPACE_3P

    def test_dahai_mapping(self):
        for i in range(N_TILES):
            assert ACTION_MAPPING[("dahai", i)] == i

    def test_riichi_mapping(self):
        for i in range(N_TILES):
            assert ACTION_MAPPING[("riichi", i)] == 34 + i

    def test_special_actions(self):
        assert ACTION_MAPPING[("agari", None)] == 68
        assert ACTION_MAPPING[("pon", None)] == 69
        assert ACTION_MAPPING[("chi_low", None)] == 70
        assert ACTION_MAPPING[("chi_mid", None)] == 71
        assert ACTION_MAPPING[("chi_high", None)] == 72
        assert ACTION_MAPPING[("daiminkan", None)] == 73
        assert ACTION_MAPPING[("kakan", None)] == 74
        assert ACTION_MAPPING[("ankan", None)] == 75
        assert ACTION_MAPPING[("ryukyoku", None)] == 76
        assert ACTION_MAPPING[("none", None)] == 77
        assert ACTION_MAPPING[("nukidora", None)] == 78

    def test_inverse_mapping(self):
        for k, v in ACTION_MAPPING.items():
            assert INV_ACTION_MAPPING[v] == k


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

class TestEncodeState:
    def _default_state(self):
        return encode_state_3p(
            hand=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            discards=[[9, 10, 11], [12, 13], [14]],
            melds=[[[15, 16, 17]], [], []],
            dora_indicators=[18],
            remaining=50,
            riichi=[False, False, False],
            scores=[35000.0, 35000.0, 35000.0],
            round_wind=0,
            seat_wind=0,
            kyoku=0,
            honba=0,
            kyotaku=0,
            nukidora_counts=[0, 0, 0],
            last_discard_tile=-1,
        )

    def test_shape(self):
        obs = self._default_state()
        assert obs.shape == (OBS_CHANNELS_3P, N_TILES)

    def test_dtype(self):
        obs = self._default_state()
        assert obs.dtype == np.float32

    def test_hand_channels(self):
        obs = self._default_state()
        # Tiles 0-8 are in hand (1 copy each) → channel 0 should be 1
        for t in range(9):
            assert obs[0, t] == 1.0, f"ch0 tile {t} should be 1"
        # Channel 1 (≥2 copies) should be 0 for tiles 0-8
        for t in range(9):
            assert obs[1, t] == 0.0, f"ch1 tile {t} should be 0"

    def test_dora_channel(self):
        obs = self._default_state()
        # Dora indicator is tile 18 (1s)
        assert obs[25, 18] == 1.0
        # All other dora channels should be 0 for tile 18
        assert obs[26, 18] == 0.0

    def test_riichi_channel(self):
        obs = encode_state_3p(
            hand=[0], discards=[[], [], []], melds=[[], [], []],
            dora_indicators=[], remaining=10,
            riichi=[True, False, True],
            scores=[35000.0] * 3,
            round_wind=0, seat_wind=0, kyoku=0, honba=0, kyotaku=0,
            nukidora_counts=[0, 0, 0],
        )
        assert (obs[31] == 1.0).all()
        assert (obs[32] == 0.0).all()
        assert (obs[33] == 1.0).all()

    def test_last_discard_one_hot(self):
        obs = encode_state_3p(
            hand=[0], discards=[[], [], []], melds=[[], [], []],
            dora_indicators=[], remaining=10,
            riichi=[False, False, False],
            scores=[35000.0] * 3,
            round_wind=0, seat_wind=0, kyoku=0, honba=0, kyotaku=0,
            nukidora_counts=[0, 0, 0],
            last_discard_tile=5,
        )
        assert obs[43, 5] == 1.0
        assert obs[43, 0] == 0.0

    def test_remaining_normalized(self):
        obs = self._default_state()  # remaining=50
        # Channel 30 should be 50/70 ≈ 0.714
        expected = 50.0 / 70.0
        assert abs(obs[30, 0] - expected) < 1e-5


# ---------------------------------------------------------------------------
# Legal actions mask
# ---------------------------------------------------------------------------

class TestLegalActionsMask:
    def test_shape(self):
        mask = make_legal_actions_mask(can_discard=[0, 1, 2])
        assert mask.shape == (ACTION_SPACE_3P,)
        assert mask.dtype == bool

    def test_can_discard(self):
        mask = make_legal_actions_mask(can_discard=[0, 5, 10])
        assert mask[0] and mask[5] and mask[10]
        assert not mask[1]

    def test_can_riichi(self):
        mask = make_legal_actions_mask(can_discard=[0], can_riichi=[3])
        assert mask[34 + 3]

    def test_can_agari(self):
        mask = make_legal_actions_mask(can_discard=[0], can_agari=True)
        assert mask[68]

    def test_none_always_available_when_no_other(self):
        # If no actions specified, 'none' should be set
        mask = make_legal_actions_mask(can_discard=[])
        assert mask[77]

    def test_nukidora(self):
        mask = make_legal_actions_mask(can_discard=[0], can_nukidora=True)
        assert mask[78]

    def test_pon(self):
        mask = make_legal_actions_mask(can_discard=[0], can_pon=True)
        assert mask[69]


# ---------------------------------------------------------------------------
# GameState3P
# ---------------------------------------------------------------------------

class TestGameState3P:
    def _start_events(self):
        return [
            {
                "type": "start_game",
                "names": ["A", "B", "C"],
                "id": 0,
            },
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "scores": [35000, 35000, 35000],
                "dora_marker": "1p",
                "tehais": [
                    ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                    ["?"] * 13,
                    ["?"] * 13,
                ],
            },
        ]

    def test_hand_dealt(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        assert len(gs.hand) == 13

    def test_tsumo_adds_to_hand(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        gs.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        assert len(gs.hand) == 14

    def test_dahai_removes_from_hand(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        gs.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        gs.apply_event({"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True})
        assert len(gs.hand) == 13
        assert tile_to_id("5p") not in gs.hand

    def test_riichi_state(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        gs.apply_event({"type": "riichi", "actor": 1})
        assert gs.riichi[1] is True
        assert gs.riichi[0] is False

    def test_dora_update(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        initial_len = len(gs.dora_indicators)
        gs.apply_event({"type": "dora", "dora_marker": "2p"})
        assert len(gs.dora_indicators) == initial_len + 1

    def test_nukidora(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        # Give player 0 a North tile
        gs.hand.append(tile_to_id("4z"))
        count_before = gs.nukidora_counts[0]
        gs.apply_event({"type": "nukidora", "actor": 0, "pai": "4z"})
        assert gs.nukidora_counts[0] == count_before + 1
        assert tile_to_id("4z") not in gs.hand

    def test_encode_shape(self):
        gs = GameState3P(player_id=0)
        for evt in self._start_events():
            gs.apply_event(evt)
        obs = gs.encode()
        assert obs.shape == (OBS_CHANNELS_3P, N_TILES)


# ---------------------------------------------------------------------------
# MJAI log converter (integration)
# ---------------------------------------------------------------------------

class TestConvertMjaiLog:
    def _make_simple_log(self) -> list[dict]:
        """Minimal 3-player MJAI log."""
        return [
            {"type": "start_game", "names": ["A", "B", "C"], "id": 0},
            {
                "type": "start_kyoku",
                "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
                "oya": 0, "scores": [35000, 35000, 35000],
                "dora_marker": "1p",
                "tehais": [
                    ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                     "1p", "2p", "3p", "4p"],
                    ["?"] * 13, ["?"] * 13,
                ],
            },
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": False},
            {"type": "tsumo", "actor": 1, "pai": "?"},
            {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": True},
            {"type": "tsumo", "actor": 2, "pai": "?"},
            {"type": "dahai", "actor": 2, "pai": "1s", "tsumogiri": True},
            {"type": "end_game"},
        ]

    def test_yields_examples(self, tmp_path):
        log = self._make_simple_log()
        log_file = tmp_path / "game.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")

        examples = list(convert_mjai_log(log_file))
        assert len(examples) > 0

    def test_example_structure(self, tmp_path):
        log = self._make_simple_log()
        log_file = tmp_path / "game.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")

        examples = list(convert_mjai_log(log_file))
        for ex in examples:
            assert "state" in ex
            assert "legal_actions_mask" in ex
            assert "action" in ex
            assert "reward" in ex
            assert "meta" in ex
            assert ex["meta"]["is_3p"] is True

    def test_state_shape(self, tmp_path):
        log = self._make_simple_log()
        log_file = tmp_path / "game.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")

        examples = list(convert_mjai_log(log_file))
        for ex in examples:
            assert len(ex["state"]) == OBS_CHANNELS_3P * N_TILES

    def test_mask_shape(self, tmp_path):
        log = self._make_simple_log()
        log_file = tmp_path / "game.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")

        examples = list(convert_mjai_log(log_file))
        for ex in examples:
            assert len(ex["legal_actions_mask"]) == ACTION_SPACE_3P

    def test_action_in_range(self, tmp_path):
        log = self._make_simple_log()
        log_file = tmp_path / "game.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")

        examples = list(convert_mjai_log(log_file))
        for ex in examples:
            assert 0 <= ex["action"] < ACTION_SPACE_3P

    def test_4p_game_skipped(self, tmp_path):
        """4-player games should produce no examples."""
        log = [
            {"type": "start_game", "names": ["A", "B", "C", "D"], "id": 0},
            {
                "type": "start_kyoku",
                "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
                "oya": 0, "scores": [25000, 25000, 25000, 25000],
                "dora_marker": "1p",
                "tehais": [["?"] * 13] * 4,
            },
            {"type": "end_game"},
        ]
        log_file = tmp_path / "4p.json"
        log_file.write_text(json.dumps(log), encoding="utf-8")
        examples = list(convert_mjai_log(log_file))
        assert len(examples) == 0


# ---------------------------------------------------------------------------
# Bot log converter
# ---------------------------------------------------------------------------

class TestConvertBotLog:
    def test_yields_examples(self, tmp_path):
        state = [0.0] * (OBS_CHANNELS_3P * N_TILES)
        mask = [True] + [False] * (ACTION_SPACE_3P - 1)
        examples_in = [
            {"state": state, "action": 0, "legal": mask, "reward": 1000.0},
            {"state": state, "action": 1, "legal": mask, "reward": -500.0},
        ]
        bot_log = tmp_path / "bot.jsonl"
        with bot_log.open("w") as f:
            for ex in examples_in:
                f.write(json.dumps(ex) + "\n")

        examples = list(convert_bot_log(bot_log))
        assert len(examples) == 2

    def test_skips_wrong_shape(self, tmp_path):
        bad = {"state": [0.0] * 10, "action": 0, "legal": [True], "reward": 0}
        bot_log = tmp_path / "bad.jsonl"
        bot_log.write_text(json.dumps(bad) + "\n")
        examples = list(convert_bot_log(bot_log))
        assert len(examples) == 0


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class TestWriters:
    def _make_example(self) -> dict:
        return {
            "state": [0.0] * (OBS_CHANNELS_3P * N_TILES),
            "legal_actions_mask": [True] * ACTION_SPACE_3P,
            "action": 0,
            "reward": 0.0,
            "meta": {"game_id": "x", "player_id": 0, "round": "E1", "step": 0, "is_3p": True},
        }

    def test_write_jsonl(self, tmp_path):
        out = tmp_path / "out.jsonl"
        examples = [self._make_example(), self._make_example()]
        write_jsonl(examples, out)
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "state" in obj

    def test_write_npz(self, tmp_path):
        out = tmp_path / "out.npz"
        examples = [self._make_example(), self._make_example()]
        write_npz(examples, out)
        data = np.load(out)
        assert "state" in data
        assert data["state"].shape == (2, OBS_CHANNELS_3P * N_TILES)
        assert data["action"].shape == (2,)
        assert data["reward"].shape == (2,)
