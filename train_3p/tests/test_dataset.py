"""
tests/test_dataset.py – Unit tests for dataset.py and model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_converter import OBS_CHANNELS_3P, ACTION_SPACE_3P, N_TILES
from dataset import MahjongDataset3P, build_dataloader
from model import Brain, DQN, build_model, save_checkpoint, load_checkpoint, parameter_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jsonl(tmp_path: Path, n: int = 10) -> Path:
    """Write a minimal JSONL file with n synthetic examples."""
    out = tmp_path / "test.jsonl"
    with out.open("w") as f:
        for i in range(n):
            ex = {
                "state": [float(i % 2)] * (OBS_CHANNELS_3P * N_TILES),
                "legal_actions_mask": [True] * ACTION_SPACE_3P,
                "action": i % ACTION_SPACE_3P,
                "reward": float(i),
                "meta": {"game_id": "t", "player_id": 0, "round": "E1",
                         "step": i, "is_3p": True},
            }
            f.write(json.dumps(ex) + "\n")
    return out


def _make_npz(tmp_path: Path, n: int = 10) -> Path:
    out = tmp_path / "test.npz"
    np.savez_compressed(
        out,
        state=np.zeros((n, OBS_CHANNELS_3P * N_TILES), dtype=np.float32),
        legal_actions_mask=np.ones((n, ACTION_SPACE_3P), dtype=bool),
        action=np.zeros(n, dtype=np.int64),
        reward=np.zeros(n, dtype=np.float32),
    )
    return out


# ---------------------------------------------------------------------------
# MahjongDataset3P
# ---------------------------------------------------------------------------

class TestMahjongDataset3P:
    def test_len_jsonl(self, tmp_path):
        path = _make_jsonl(tmp_path, n=15)
        ds = MahjongDataset3P(path)
        assert len(ds) == 15

    def test_len_npz(self, tmp_path):
        path = _make_npz(tmp_path, n=8)
        ds = MahjongDataset3P(path)
        assert len(ds) == 8

    def test_max_samples(self, tmp_path):
        path = _make_jsonl(tmp_path, n=20)
        ds = MahjongDataset3P(path, max_samples=5)
        assert len(ds) == 5

    def test_sample_shape(self, tmp_path):
        path = _make_jsonl(tmp_path, n=5)
        ds = MahjongDataset3P(path)
        sample = ds[0]
        assert sample["state"].shape == (OBS_CHANNELS_3P, N_TILES)
        assert sample["legal_actions_mask"].shape == (ACTION_SPACE_3P,)
        assert sample["action"].shape == ()
        assert sample["reward"].shape == ()

    def test_sample_dtype(self, tmp_path):
        path = _make_jsonl(tmp_path, n=5)
        ds = MahjongDataset3P(path)
        sample = ds[0]
        assert sample["state"].dtype == torch.float32
        assert sample["legal_actions_mask"].dtype == torch.bool
        assert sample["action"].dtype == torch.long
        assert sample["reward"].dtype == torch.float32

    def test_action_values(self, tmp_path):
        path = _make_jsonl(tmp_path, n=10)
        ds = MahjongDataset3P(path)
        for i in range(len(ds)):
            a = int(ds[i]["action"])
            assert 0 <= a < ACTION_SPACE_3P

    def test_augment_doubles_size(self, tmp_path):
        path = _make_jsonl(tmp_path, n=10)
        ds_base = MahjongDataset3P(path, augment=False)
        ds_aug = MahjongDataset3P(path, augment=True)
        assert len(ds_aug) == 2 * len(ds_base)


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------

class TestBuildDataloader:
    def test_batch_shape(self, tmp_path):
        path = _make_jsonl(tmp_path, n=20)
        loader = build_dataloader(path, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["state"].shape == (4, OBS_CHANNELS_3P, N_TILES)
        assert batch["legal_actions_mask"].shape == (4, ACTION_SPACE_3P)
        assert batch["action"].shape == (4,)
        assert batch["reward"].shape == (4,)

    def test_all_batches(self, tmp_path):
        n = 17
        path = _make_jsonl(tmp_path, n=n)
        loader = build_dataloader(path, batch_size=5, shuffle=False)
        total = sum(len(b["action"]) for b in loader)
        assert total == n


# ---------------------------------------------------------------------------
# Brain model
# ---------------------------------------------------------------------------

class TestBrain:
    def test_output_shape_v4(self):
        mortal = Brain(version=4, conv_channels=32, num_blocks=2)
        x = torch.randn(2, OBS_CHANNELS_3P, N_TILES)
        out = mortal(x)
        assert out.shape == (2, 1024)

    def test_output_shape_v2(self):
        mortal = Brain(version=2, conv_channels=32, num_blocks=2)
        x = torch.randn(3, OBS_CHANNELS_3P, N_TILES)
        out = mortal(x)
        assert out.shape == (3, 1024)

    def test_eval_mode(self):
        mortal = Brain(version=4, conv_channels=32, num_blocks=2)
        mortal.eval()
        x = torch.randn(1, OBS_CHANNELS_3P, N_TILES)
        with torch.inference_mode():
            out = mortal(x)
        assert out.shape == (1, 1024)

    def test_parameter_count(self):
        mortal = Brain(version=4, conv_channels=32, num_blocks=2)
        count = parameter_count(mortal)
        assert count > 0


# ---------------------------------------------------------------------------
# DQN model
# ---------------------------------------------------------------------------

class TestDQN:
    def test_output_shape_v4(self):
        dqn = DQN(version=4)
        phi = torch.randn(2, 1024)
        mask = torch.ones(2, ACTION_SPACE_3P, dtype=torch.bool)
        q = dqn(phi, mask)
        assert q.shape == (2, ACTION_SPACE_3P)

    def test_masked_actions_are_neginf(self):
        dqn = DQN(version=4)
        phi = torch.randn(1, 1024)
        mask = torch.zeros(1, ACTION_SPACE_3P, dtype=torch.bool)
        mask[0, 0] = True  # only action 0 is legal
        q = dqn(phi, mask)
        assert q[0, 0] != -torch.inf
        for i in range(1, ACTION_SPACE_3P):
            assert q[0, i].item() == -torch.inf

    def test_argmax_respects_mask(self):
        dqn = DQN(version=4)
        phi = torch.randn(1, 1024)
        mask = torch.zeros(1, ACTION_SPACE_3P, dtype=torch.bool)
        legal_idx = 5
        mask[0, legal_idx] = True
        q = dqn(phi, mask)
        assert q.argmax(dim=-1).item() == legal_idx


# ---------------------------------------------------------------------------
# build_model / save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_build_model_returns_brain_dqn(self):
        mortal, dqn = build_model(version=4, conv_channels=32, num_blocks=2)
        assert isinstance(mortal, Brain)
        assert isinstance(dqn, DQN)

    def test_save_load_roundtrip(self, tmp_path):
        mortal, dqn = build_model(version=4, conv_channels=32, num_blocks=2)

        # Set some non-zero weights so we can verify they round-trip
        with torch.no_grad():
            for p in mortal.parameters():
                p.fill_(1.0)

        ckpt = tmp_path / "mortal_3p.pth"
        save_checkpoint(mortal, dqn, steps=100, path=ckpt)
        assert ckpt.exists()

        mortal2, dqn2 = load_checkpoint(ckpt)
        # Verify weights match
        for p1, p2 in zip(mortal.parameters(), mortal2.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_has_config(self, tmp_path):
        import torch as _torch
        mortal, dqn = build_model(version=4, conv_channels=32, num_blocks=2)
        ckpt = tmp_path / "mortal_3p.pth"
        save_checkpoint(mortal, dqn, path=ckpt)
        state = _torch.load(ckpt, map_location="cpu", weights_only=True)
        assert "config" in state
        assert state["config"]["control"]["version"] == 4
        assert state["config"]["is_3p"] is True

    def test_checkpoint_forward_pass(self, tmp_path):
        """Verify that a loaded checkpoint produces the same output."""
        mortal, dqn = build_model(version=4, conv_channels=32, num_blocks=2)
        mortal.eval()
        dqn.eval()

        x = torch.randn(1, OBS_CHANNELS_3P, N_TILES)
        mask = torch.ones(1, ACTION_SPACE_3P, dtype=torch.bool)

        with torch.inference_mode():
            phi1 = mortal(x)
            q1 = dqn(phi1, mask)

        ckpt = tmp_path / "test.pth"
        save_checkpoint(mortal, dqn, path=ckpt)
        m2, d2 = load_checkpoint(ckpt)

        with torch.inference_mode():
            phi2 = m2(x)
            q2 = d2(phi2, mask)

        assert torch.allclose(q1, q2)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_generate_sample_train(self, tmp_path):
        """Verify the full generate → train pipeline runs without errors."""
        import subprocess
        import sys

        train_3p = Path(__file__).parent.parent

        # Generate sample data
        sample_path = tmp_path / "sample.jsonl"
        result = subprocess.run(
            [sys.executable, str(train_3p / "generate_sample.py"),
             "--n", "20", "--out", str(sample_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert sample_path.exists()

        # Train for 1 epoch
        ckpt_path = tmp_path / "mortal_3p.pth"
        result = subprocess.run(
            [sys.executable, str(train_3p / "train.py"),
             "--data", str(sample_path),
             "--epochs", "1",
             "--batch-size", "8",
             "--conv-channels", "16",
             "--num-blocks", "2",
             "--out", str(ckpt_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert ckpt_path.exists(), "mortal_3p.pth should be created"

    def test_eval_after_train(self, tmp_path):
        """Verify that eval.py runs on the trained checkpoint."""
        import subprocess
        import sys

        train_3p = Path(__file__).parent.parent

        sample_path = tmp_path / "sample.jsonl"
        subprocess.run(
            [sys.executable, str(train_3p / "generate_sample.py"),
             "--n", "10", "--out", str(sample_path)],
            capture_output=True,
        )

        ckpt_path = tmp_path / "mortal_3p.pth"
        subprocess.run(
            [sys.executable, str(train_3p / "train.py"),
             "--data", str(sample_path),
             "--epochs", "1",
             "--batch-size", "5",
             "--conv-channels", "16",
             "--num-blocks", "2",
             "--out", str(ckpt_path)],
            capture_output=True,
        )

        result_path = tmp_path / "results.json"
        result = subprocess.run(
            [sys.executable, str(train_3p / "eval.py"),
             "--checkpoint", str(ckpt_path),
             "--data", str(sample_path),
             "--batch-size", "5",
             "--out-json", str(result_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert result_path.exists()
        results = json.loads(result_path.read_text())
        assert "offline" in results
        assert "policy_acc" in results["offline"]
