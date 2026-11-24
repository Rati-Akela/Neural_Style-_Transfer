"""PyTorch definition of the AdaIN style-transfer network (Huang & Belongie, 2017).

``VGGEncoder`` is a frozen, pre-trained VGG-19 truncated at ``relu4_1``;
``Decoder`` mirrors it and is the only part that is trained. Layer indices and
state-dict keys are identical to the original implementation, so existing
checkpoints (``decoder_final.pth``) load unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

# (start, end) slices of ``VGGEncoder.vgg`` that end at relu1_1 ... relu4_1.
_LEVELS = ((0, 4), (4, 11), (11, 18), (18, 31))
_N_LAYERS = 31


def _conv(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(c_in, c_out, (3, 3)), nn.ReLU()]


def _pool() -> nn.Module:
    return nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)


def _up() -> nn.Module:
    return nn.Upsample(scale_factor=2, mode="nearest")


def _vgg_to_relu4_1() -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(3, 3, (1, 1))]  # absorbs input normalisation
    layers += _conv(3, 64) + _conv(64, 64) + [_pool()]
    layers += _conv(64, 128) + _conv(128, 128) + [_pool()]
    layers += _conv(128, 256) + _conv(256, 256) + _conv(256, 256) + _conv(256, 256) + [_pool()]
    layers += _conv(256, 512)
    assert len(layers) == _N_LAYERS
    return nn.Sequential(*layers)


class VGGEncoder(nn.Module):
    """Frozen VGG-19 encoder. Expects RGB images in [0, 1], shape (N, 3, H, W)."""

    def __init__(self, weights: str | Path | None = None):
        super().__init__()
        self.vgg = _vgg_to_relu4_1()
        if weights is not None:
            self.load_weights(weights)
        for p in self.parameters():
            p.requires_grad = False

    def load_weights(self, path: str | Path) -> None:
        """Load either the full 'vgg_normalised.pth' or a trimmed relu4_1 checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        keep = {k: v for k, v in state.items() if int(k.split(".")[0]) < _N_LAYERS}
        self.vgg.load_state_dict(keep)

    def forward(self, x: torch.Tensor, all_levels: bool = False):
        """relu4_1 features, or the list of relu1_1..relu4_1 features when ``all_levels``."""
        outs, h = [], x
        for start, end in _LEVELS:
            for layer in self.vgg[start:end]:
                h = layer(h)
            outs.append(h)
        return outs if all_levels else outs[-1]


class Decoder(nn.Module):
    """Inverts relu4_1 features back to an RGB image (values roughly in [0, 1])."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *_conv(512, 256), _up(),
            *_conv(256, 256), *_conv(256, 256), *_conv(256, 256),
            *_conv(256, 128), _up(),
            *_conv(128, 128), *_conv(128, 64), _up(),
            *_conv(64, 64),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_weights(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


# ---- AdaIN (torch version, used by training and for parity tests) -------------------
def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    n, c = feat.shape[:2]
    flat = feat.reshape(n, c, -1)
    mean = flat.mean(dim=2).reshape(n, c, 1, 1)
    std = (flat.var(dim=2, unbiased=False) + eps).sqrt().reshape(n, c, 1, 1)
    return mean, std


def adaptive_instance_normalization(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    c_mean, c_std = calc_mean_std(content)
    s_mean, s_std = calc_mean_std(style)
    return (content - c_mean) / c_std * s_std + s_mean
