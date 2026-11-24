from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STYLES = ROOT / "examples" / "styles"
CONTENT = ROOT / "examples" / "content"


@pytest.fixture(scope="session")
def engine():
    from adain.engine import StyleTransfer

    return StyleTransfer()


@pytest.fixture(scope="session")
def cat() -> Image.Image:
    return Image.open(CONTENT / "cat.jpg").convert("RGB")


@pytest.fixture(scope="session")
def style_a() -> Image.Image:
    return Image.open(STYLES / "la_muse.jpg").convert("RGB")


@pytest.fixture(scope="session")
def style_b() -> Image.Image:
    return Image.open(STYLES / "mondrian.jpg").convert("RGB")


def random_image(w: int, h: int, seed: int = 0, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3 if mode == "RGB" else 4), dtype=np.uint8)
    return Image.fromarray(arr, mode)
