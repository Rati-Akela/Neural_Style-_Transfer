"""ONNX Runtime inference for AdaIN style transfer - no PyTorch needed to serve.

The network is split into two ONNX graphs, ``encoder.onnx`` (image -> relu4_1
features) and ``decoder.onnx`` (features -> image). AdaIN itself is a few lines
of NumPy (``adain.ops``), which makes it cheap to re-run with a different
style strength or blend without touching the encoder again.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import ops

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Encoded:
    """An image resized for the network, together with its relu4_1 features."""

    image: Image.Image
    features: np.ndarray


class StyleTransfer:
    def __init__(
        self,
        models_dir: str | Path = MODELS_DIR,
        providers: Sequence[str] | None = None,
        cache_size: int = 16,
        low_memory: bool = True,
    ):
        models_dir = Path(models_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        if low_memory:
            # Skip ONNX Runtime's memory arena: ~40% lower peak RAM for <10% extra latency
            # (measured at 512-768 px), which matters on small free hosting tiers.
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
        providers = list(providers or ort.get_available_providers())
        # Prefer an accelerator when present, always keep CPU as the fallback.
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        self._encoder = ort.InferenceSession(str(models_dir / "encoder.onnx"), opts, providers=providers)
        self._decoder = ort.InferenceSession(str(models_dir / "decoder.onnx"), opts, providers=providers)
        self._cache: OrderedDict[tuple, Encoded] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()  # Streamlit serves sessions from several threads

    # ------------------------------------------------------------------ network
    def _encode(self, arr: np.ndarray) -> np.ndarray:
        return self._encoder.run(None, {"image": arr})[0]

    def _decode(self, feats: np.ndarray) -> np.ndarray:
        return self._decoder.run(None, {"features": feats.astype(np.float32)})[0]

    # -------------------------------------------------------------------- API
    def encode_image(self, img: Image.Image, max_side: int = 512) -> Encoded:
        """Resize + encode an image. Results are cached, so re-using a style is free."""
        prepared = ops.prepare_image(img, max_side)
        key = (
            hashlib.blake2b(prepared.tobytes(), digest_size=16).digest(),
            prepared.size,
        )
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        encoded = Encoded(prepared, self._encode(ops.to_array(prepared)))
        with self._lock:
            self._cache[key] = encoded
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return encoded

    def render(
        self,
        content: Encoded,
        styles: Sequence[Encoded],
        *,
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        keep_colors: bool = False,
    ) -> Image.Image:
        """Stylise already-encoded images. Cheap: only AdaIN + one decoder pass."""
        stylized = ops.blend_adain(content.features, [s.features for s in styles], weights)
        out = ops.to_image(self._decode(ops.mix(stylized, content.features, alpha)))
        if out.size != content.image.size:  # decoder output is always a multiple of 8
            out = out.resize(content.image.size, Image.Resampling.LANCZOS)
        return ops.preserve_color(out, content.image) if keep_colors else out

    def stylize(
        self,
        content: Image.Image,
        styles: Image.Image | Sequence[Image.Image],
        *,
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        size: int = 512,
        style_scale: float = 1.0,
        keep_colors: bool = False,
    ) -> Image.Image:
        """One-call convenience: content photo + style image(s) -> stylised PIL image.

        ``size`` is the longer edge of the output in pixels. ``style_scale`` resizes the
        style image relative to that: smaller values give finer strokes, larger give
        bolder, coarser ones.
        """
        if isinstance(styles, Image.Image):
            styles = [styles]
        c = self.encode_image(content, size)
        s = [self.encode_image(img, max(32, round(size * style_scale))) for img in styles]
        return self.render(c, s, weights=weights, alpha=alpha, keep_colors=keep_colors)
