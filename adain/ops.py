"""Torch-free building blocks: AdaIN in NumPy plus image pre/post-processing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageOps

EPS = 1e-5
MULTIPLE = 8  # the network downsamples 3x, so sizes must be multiples of 8 to round-trip
MIN_SIDE = 32  # smaller images collapse to 1x1 feature maps, which reflection padding rejects


# --------------------------------------------------------------------------- AdaIN
def calc_mean_std(feat: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample, per-channel mean and std over the spatial dims of (N, C, H, W)."""
    if feat.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) features, got shape {feat.shape}")
    n, c = feat.shape[:2]
    flat = feat.reshape(n, c, -1)
    mean = flat.mean(axis=2).reshape(n, c, 1, 1)
    std = np.sqrt(flat.var(axis=2) + eps).reshape(n, c, 1, 1)
    return mean, std


def adain(content: np.ndarray, style: np.ndarray) -> np.ndarray:
    """Re-normalise content features so their channel statistics match the style's."""
    c_mean, c_std = calc_mean_std(content)
    s_mean, s_std = calc_mean_std(style)
    return (content - c_mean) / c_std * s_std + s_mean


def normalise_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if n < 1:
        raise ValueError("at least one style image is required")
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"got {len(weights)} weights for {n} styles")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = float(sum(weights))
    return [w / total for w in weights]


def blend_adain(
    content: np.ndarray,
    styles: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted mix of AdaIN outputs, one per style (style interpolation, paper section 6.3)."""
    w = normalise_weights(len(styles), weights)
    return sum(wi * adain(content, s) for wi, s in zip(w, styles, strict=True))


def mix(stylized: np.ndarray, content: np.ndarray, alpha: float) -> np.ndarray:
    """Style-strength control: alpha=0 reproduces the content, alpha=1 is full stylisation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * stylized + (1.0 - alpha) * content


# ------------------------------------------------------------------------ images
def snap(n: int, multiple: int = MULTIPLE, minimum: int = MIN_SIDE) -> int:
    return max(minimum, round(n / multiple) * multiple)


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale so the longer edge is ``max_side`` (aspect kept), then snap to multiples of 8."""
    scale = max_side / max(width, height)
    return snap(round(width * scale)), snap(round(height * scale))


def prepare_image(img: Image.Image, max_side: int) -> Image.Image:
    """Honour EXIF rotation (phone photos), force RGB, resize for the network."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white, not black
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("RGB")
    return img.resize(fit_size(*img.size, max_side), Image.Resampling.LANCZOS)


def to_array(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 (1, 3, H, W) in [0, 1]."""
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def to_image(arr: np.ndarray) -> Image.Image:
    """float (1, 3, H, W) -> PIL RGB, clipping out-of-range values."""
    hwc = np.clip(arr[0].transpose(1, 2, 0), 0.0, 1.0)
    return Image.fromarray((hwc * 255.0 + 0.5).astype(np.uint8))


def preserve_color(stylized: Image.Image, content: Image.Image) -> Image.Image:
    """Keep the stylised luminance but restore the content photo's colours (Gatys et al., 2017)."""
    content = content.resize(stylized.size, Image.Resampling.LANCZOS)
    y, _, _ = stylized.convert("YCbCr").split()
    _, cb, cr = content.convert("YCbCr").split()
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
