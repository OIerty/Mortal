"""
dataset.py – PyTorch Dataset for 3P mahjong supervised training.

Supports loading from:
* JSONL files  (written by data_converter.py)
* NPZ archives (numpy compressed arrays)
* PyTorch .pt files

Each sample returned is a dict with keys:
    state               : FloatTensor of shape (OBS_CHANNELS_3P, 34)
    legal_actions_mask  : BoolTensor  of shape (ACTION_SPACE_3P,)
    action              : LongTensor  scalar
    reward              : FloatTensor scalar

Usage
-----
    from dataset import MahjongDataset3P
    ds = MahjongDataset3P("train.jsonl")
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    for batch in loader:
        states = batch["state"]          # (B, 54, 34)
        masks  = batch["legal_actions_mask"]  # (B, 79)
        acts   = batch["action"]         # (B,)
        rews   = batch["reward"]         # (B,)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from data_converter import OBS_CHANNELS_3P, ACTION_SPACE_3P, N_TILES


class MahjongDataset3P(Dataset):
    """
    Dataset that loads 3P training examples from JSONL, NPZ, or PT files.

    Parameters
    ----------
    path : str or Path
        Path to a ``.jsonl``, ``.npz``, or ``.pt`` file.
    max_samples : int, optional
        Truncate the dataset to at most this many examples (useful for quick
        smoke tests).
    augment : bool
        If True, apply tile-suit rotation augmentation (man↔pin swapping) to
        triple the effective dataset size. Default False.
    """

    def __init__(
        self,
        path: Union[str, Path],
        max_samples: int | None = None,
        augment: bool = False,
    ):
        self.path = Path(path)
        self.augment = augment

        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            self._load_jsonl(max_samples)
        elif suffix == ".npz":
            self._load_npz(max_samples)
        elif suffix in (".pt", ".pth"):
            self._load_pt(max_samples)
        else:
            # Try jsonl as default
            self._load_jsonl(max_samples)

        if augment:
            # Augment by suit rotation (man←→pin, sou unchanged)
            aug_states, aug_masks, aug_acts, aug_rews = self._augment_suit_rotation()
            self.states = torch.cat([self.states, aug_states], dim=0)
            self.masks = torch.cat([self.masks, aug_masks], dim=0)
            self.actions = torch.cat([self.actions, aug_acts], dim=0)
            self.rewards = torch.cat([self.rewards, aug_rews], dim=0)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_jsonl(self, max_samples: int | None):
        states, masks, actions, rewards = [], [], [], []
        with self.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                states.append(obj["state"])
                masks.append(obj["legal_actions_mask"])
                actions.append(obj["action"])
                rewards.append(obj.get("reward", 0.0))

        self._store(states, masks, actions, rewards)

    def _load_npz(self, max_samples: int | None):
        data = np.load(self.path)
        n = len(data["state"])
        if max_samples is not None:
            n = min(n, max_samples)
        self.states = torch.from_numpy(data["state"][:n]).float()
        # Reshape if stored as flat
        if self.states.ndim == 2:
            self.states = self.states.view(-1, OBS_CHANNELS_3P, N_TILES)
        self.masks = torch.from_numpy(data["legal_actions_mask"][:n]).bool()
        self.actions = torch.from_numpy(data["action"][:n]).long()
        self.rewards = torch.from_numpy(data["reward"][:n]).float()

    def _load_pt(self, max_samples: int | None):
        data = torch.load(self.path, weights_only=True)
        n = len(data["state"])
        if max_samples is not None:
            n = min(n, max_samples)
        self.states = data["state"][:n].float()
        if self.states.ndim == 2:
            self.states = self.states.view(-1, OBS_CHANNELS_3P, N_TILES)
        self.masks = data["legal_actions_mask"][:n].bool()
        self.actions = data["action"][:n].long()
        self.rewards = data["reward"][:n].float()

    def _store(
        self,
        states: list,
        masks: list,
        actions: list,
        rewards: list,
    ):
        arr = np.array(states, dtype=np.float32)
        self.states = torch.from_numpy(arr).view(-1, OBS_CHANNELS_3P, N_TILES)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    _MAN_SLICE = slice(0, 9)   # tile indices 0-8
    _PIN_SLICE = slice(9, 18)  # tile indices 9-17

    def _augment_suit_rotation(self):
        """
        Swap man (0-8) ↔ pin (9-17) tile columns.
        Also remap action indices for dahai/riichi_dahai involving man/pin.
        """
        aug_states = self.states.clone()
        # Swap tile columns 0-8 with 9-17 in all channels
        tmp = aug_states[:, :, self._MAN_SLICE].clone()
        aug_states[:, :, self._MAN_SLICE] = aug_states[:, :, self._PIN_SLICE]
        aug_states[:, :, self._PIN_SLICE] = tmp

        aug_acts = self.actions.clone()

        def remap_action(a: int) -> int:
            if 0 <= a <= 8:        # dahai man → dahai pin
                return a + 9
            if 9 <= a <= 17:       # dahai pin → dahai man
                return a - 9
            if 34 <= a <= 42:      # riichi+dahai man → riichi+dahai pin
                return a + 9
            if 43 <= a <= 51:      # riichi+dahai pin → riichi+dahai man
                return a - 9
            return a

        aug_acts = torch.tensor(
            [remap_action(int(a)) for a in aug_acts], dtype=torch.long
        )

        aug_masks = self.masks.clone()
        for idx in range(len(aug_masks)):
            new_mask = aug_masks[idx].clone()
            for src, dst in [
                (range(0, 9), range(9, 18)),
                (range(9, 18), range(0, 9)),
                (range(34, 43), range(43, 52)),
                (range(43, 52), range(34, 43)),
            ]:
                for s, d in zip(src, dst):
                    new_mask[d] = aug_masks[idx][s]
            aug_masks[idx] = new_mask

        return aug_states, aug_masks, aug_acts, self.rewards.clone()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "state": self.states[idx],
            "legal_actions_mask": self.masks[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
        }


def build_dataloader(
    path: Union[str, Path],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int | None = None,
    augment: bool = False,
) -> DataLoader:
    """
    Convenience factory that creates a DataLoader from a training file.

    Parameters
    ----------
    path        : Path to jsonl / npz / pt file
    batch_size  : Batch size
    shuffle     : Whether to shuffle examples each epoch
    num_workers : DataLoader worker processes
    max_samples : Truncate dataset (None = use all)
    augment     : Apply suit-rotation augmentation
    """
    ds = MahjongDataset3P(path, max_samples=max_samples, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
