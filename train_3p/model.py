"""
model.py – PyTorch model for 3P mahjong (policy + value heads).

Architecture
------------
This module implements the same **Brain (ResNet encoder) + DQN (policy/value)**
architecture used by the production Mortal model (Equim-chan/Mortal), adapted
for three-player mahjong observation and action spaces.

The checkpoint format produced by :func:`save_checkpoint` is compatible with
the loader in ``OIerty/Akagi`` → ``mjai_bot/mortal3p/model.py``:

    state = torch.load("mortal_3p.pth", map_location=device)
    mortal = Brain(
        version=state["config"]["control"]["version"],
        conv_channels=state["config"]["resnet"]["conv_channels"],
        num_blocks=state["config"]["resnet"]["num_blocks"],
    )
    dqn = DQN(version=state["config"]["control"]["version"])
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])

Typical usage
-------------
    from model import build_model, save_checkpoint, load_checkpoint

    mortal, dqn = build_model(device=torch.device("cpu"))

    # … training loop …

    save_checkpoint(mortal, dqn, steps=1000, path="mortal_3p.pth")
    mortal2, dqn2 = load_checkpoint("mortal_3p.pth", device=torch.device("cpu"))
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from data_converter import OBS_CHANNELS_3P, ACTION_SPACE_3P, N_TILES

# ---------------------------------------------------------------------------
# Default architecture hyper-parameters
# Tuned for a fast baseline that fits in ~2 GB RAM on CPU.
# For better quality, increase conv_channels to 192 and num_blocks to 40
# (matching the production Mortal model size).
# ---------------------------------------------------------------------------
DEFAULT_VERSION = 4
DEFAULT_CONV_CHANNELS = 64
DEFAULT_NUM_BLOCKS = 6


# ---------------------------------------------------------------------------
# Building blocks (identical to Equim-chan/Mortal for drop-in compatibility)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention over a 1-D feature map."""

    def __init__(self, channels: int, ratio: int = 16, actv_builder=nn.ReLU, bias: bool = True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, max(1, channels // ratio), bias=bias),
            actv_builder(),
            nn.Linear(max(1, channels // ratio), channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        return weight.unsqueeze(-1) * x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        norm_builder=nn.Identity,
        actv_builder=nn.ReLU,
        pre_actv: bool = False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()

        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        num_blocks: int,
        *,
        norm_builder=nn.Identity,
        actv_builder=nn.ReLU,
        pre_actv: bool = False,
    ):
        super().__init__()

        blocks = [
            ResBlock(
                conv_channels,
                norm_builder=norm_builder,
                actv_builder=actv_builder,
                pre_actv=pre_actv,
            )
            for _ in range(num_blocks)
        ]

        layers = [nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        layers += [
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * N_TILES, 1024),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Brain – state encoder
# ---------------------------------------------------------------------------

class Brain(nn.Module):
    """
    ResNet-based state encoder for 3P mahjong.

    Input:  ``(B, OBS_CHANNELS_3P, 34)`` observation tensor
    Output: ``(B, 1024)`` latent feature vector (version 2/3/4)

    The ``version`` parameter selects the architecture variant:
        1 – legacy (mu/logsig VAE-style latent, not recommended for new training)
        2 – ResNet, no BatchNorm
        3 – ResNet + BatchNorm (eps=1e-5)
        4 – ResNet + BatchNorm (eps=1e-3) **recommended for new training**
    """

    def __init__(
        self,
        *,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        is_oracle: bool = False,
        version: int = DEFAULT_VERSION,
    ):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = OBS_CHANNELS_3P

        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01)
        actv_builder = partial(nn.Mish, inplace=True)
        pre_actv = True

        match version:
            case 1:
                actv_builder = partial(nn.ReLU, inplace=True)
                pre_actv = False
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                )
                self.mu_head = nn.Linear(512, 512)
                self.logsig_head = nn.Linear(512, 512)
            case 2:
                pass
            case 3:
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-5)
            case 4:
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
            case _:
                raise ValueError(f"Unexpected Brain version {version}")

        self.encoder = ResNet(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            actv_builder=actv_builder,
            pre_actv=pre_actv,
        )
        self.actv = actv_builder()
        self._freeze_bn = False

    def forward(
        self, obs: Tensor, invisible_obs: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        phi = self.encoder(obs)
        match self.version:
            case 1:
                latent_out = self.latent_net(phi)
                return self.mu_head(latent_out), self.logsig_head(latent_out)
            case 2 | 3 | 4:
                return self.actv(phi)
            case _:
                raise ValueError(f"Unexpected Brain version {self.version}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
        return self

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)


# ---------------------------------------------------------------------------
# DQN – policy + value heads
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """
    Dueling-DQN head for 3P mahjong.

    Input:  ``(B, 1024)`` latent features from Brain + bool mask ``(B, 79)``
    Output: ``(B, ACTION_SPACE_3P)`` Q-values (masked, illegal = -inf)

    Version variants:
        1 – v_head + a_head each a single Linear (512 units)
        2 – v_head + a_head each a two-layer MLP (512 hidden units)
        3 – v_head + a_head each a two-layer MLP (256 hidden units)
        4 – single Linear outputting (1 + ACTION_SPACE_3P); faster
    """

    def __init__(self, *, version: int = DEFAULT_VERSION):
        super().__init__()
        self.version = version

        match version:
            case 1:
                self.v_head = nn.Linear(512, 1)
                self.a_head = nn.Linear(512, ACTION_SPACE_3P)
            case 2 | 3:
                hidden = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden, 1),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden, ACTION_SPACE_3P),
                )
            case 4:
                self.net = nn.Linear(1024, 1 + ACTION_SPACE_3P)
                nn.init.constant_(self.net.bias, 0)
            case _:
                raise ValueError(f"Unexpected DQN version {version}")

    def forward(self, phi: Tensor, mask: Tensor) -> Tensor:
        if self.version == 4:
            v, a = self.net(phi).split((1, ACTION_SPACE_3P), dim=-1)
        else:
            v = self.v_head(phi)
            a = self.a_head(phi)

        a_sum = a.masked_fill(~mask, 0.0).sum(-1, keepdim=True)
        mask_sum = mask.sum(-1, keepdim=True).clamp(min=1)
        a_mean = a_sum / mask_sum
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_model(
    version: int = DEFAULT_VERSION,
    conv_channels: int = DEFAULT_CONV_CHANNELS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    device: Optional[torch.device] = None,
) -> Tuple[Brain, DQN]:
    """
    Instantiate Brain + DQN and move to *device*.

    Returns
    -------
    (mortal, dqn)  – both in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
    dqn = DQN(version=version)
    mortal.to(device)
    dqn.to(device)
    return mortal, dqn


def save_checkpoint(
    mortal: Brain,
    dqn: DQN,
    *,
    steps: int = 0,
    path: Union[str, Path] = "mortal_3p.pth",
    extra: Optional[dict] = None,
):
    """
    Save a Mortal-compatible checkpoint.

    The saved file can be loaded by the ``mjai_bot/mortal3p/model.py`` loader
    in OIerty/Akagi without modification.

    Parameters
    ----------
    mortal : Brain instance
    dqn    : DQN instance
    steps  : Training step count (stored in checkpoint for resumption)
    path   : Destination file path (default: ``mortal_3p.pth``)
    extra  : Optional dict of additional keys to merge into the checkpoint
    """
    config = {
        "control": {
            "version": mortal.version,
        },
        "resnet": {
            "conv_channels": _get_conv_channels(mortal),
            "num_blocks": _get_num_blocks(mortal),
        },
        "is_3p": True,
    }
    state = {
        "mortal": mortal.state_dict(),
        "current_dqn": dqn.state_dict(),
        "steps": steps,
        "timestamp": datetime.now().timestamp(),
        "config": config,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[Brain, DQN]:
    """
    Load Brain + DQN from a Mortal-compatible checkpoint.

    Returns
    -------
    (mortal, dqn)  – both in eval mode on *device*
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(path, map_location=device, weights_only=True)
    cfg = state.get("config", {})
    version = cfg.get("control", {}).get("version", DEFAULT_VERSION)
    conv_channels = cfg.get("resnet", {}).get("conv_channels", DEFAULT_CONV_CHANNELS)
    num_blocks = cfg.get("resnet", {}).get("num_blocks", DEFAULT_NUM_BLOCKS)

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
    dqn = DQN(version=version)
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    mortal.to(device).eval()
    dqn.to(device).eval()
    return mortal, dqn


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_conv_channels(mortal: Brain) -> int:
    for mod in mortal.encoder.net.modules():
        if isinstance(mod, nn.Conv1d):
            return mod.out_channels
    return DEFAULT_CONV_CHANNELS


def _get_num_blocks(mortal: Brain) -> int:
    count = 0
    for mod in mortal.encoder.net.modules():
        if isinstance(mod, ResBlock):
            count += 1
    return count
