"""End-to-end tests of the ONNX inference engine with the real trained model."""

import numpy as np
import pytest
from PIL import Image

from adain import ops

from .conftest import random_image


def arr(img):
    return np.asarray(img, dtype=np.float32)


def test_output_is_valid_image_with_requested_size(engine, cat, style_a):
    out = engine.stylize(cat, style_a, size=256)
    assert out.mode == "RGB"
    assert max(out.size) == 256 and out.size[0] % 8 == out.size[1] % 8 == 0
    assert abs(out.width / out.height - cat.width / cat.height) < 0.05


@pytest.mark.parametrize("size", [128, (97, 211), (211, 97), (33, 33), (2000, 40)])
def test_handles_odd_shapes(engine, style_a, size):
    img = random_image(*(size if isinstance(size, tuple) else (size, size)))
    out = engine.stylize(img, style_a, size=128)
    assert out.mode == "RGB" and min(out.size) >= ops.MIN_SIDE


def test_accepts_grayscale_rgba_and_palette_inputs(engine, style_a):
    for img in (Image.new("L", (80, 60), 90), random_image(80, 60, mode="RGBA"), Image.new("P", (80, 60))):
        assert engine.stylize(img, style_a, size=96).mode == "RGB"


def test_different_styles_give_different_results(engine, cat, style_a, style_b):
    a = arr(engine.stylize(cat, style_a, size=192))
    b = arr(engine.stylize(cat, style_b, size=192))
    assert np.abs(a - b).mean() > 10


def test_is_deterministic(engine, cat, style_a):
    a = arr(engine.stylize(cat, style_a, size=192))
    b = arr(engine.stylize(cat, style_a, size=192))
    np.testing.assert_array_equal(a, b)


def test_alpha_interpolates_between_content_and_style(engine, cat, style_a):
    """Distance to the plain reconstruction must grow with style strength."""
    c = engine.encode_image(cat, 192)
    s = [engine.encode_image(style_a, 192)]
    base = arr(engine.render(c, s, alpha=0.0))
    dists = [np.abs(arr(engine.render(c, s, alpha=a)) - base).mean() for a in (0.25, 0.5, 0.75, 1.0)]
    assert dists == sorted(dists) and dists[0] > 0.5


def test_alpha_zero_reconstructs_the_photo(engine, cat, style_a):
    """The decoder is a lossy inverse of relu4_1 (~19 dB), but far closer to the photo than a stylisation."""
    c = engine.encode_image(cat, 256)
    s = [engine.encode_image(style_a, 256)]
    photo = arr(c.image)
    rebuilt = np.abs(arr(engine.render(c, s, alpha=0.0)) - photo).mean()
    stylized = np.abs(arr(engine.render(c, s, alpha=1.0)) - photo).mean()
    assert rebuilt < 0.6 * stylized


def test_style_blend_extremes_match_single_styles(engine, cat, style_a, style_b):
    c = engine.encode_image(cat, 160)
    a, b = engine.encode_image(style_a, 160), engine.encode_image(style_b, 160)
    np.testing.assert_array_equal(arr(engine.render(c, [a, b], weights=[1, 0])), arr(engine.render(c, [a])))
    np.testing.assert_array_equal(arr(engine.render(c, [a, b], weights=[0, 1])), arr(engine.render(c, [b])))
    mid = arr(engine.render(c, [a, b], weights=[1, 1]))
    assert np.abs(mid - arr(engine.render(c, [a]))).mean() > 1


def test_keep_colors_restores_photo_chroma(engine, cat, style_a):
    c = engine.encode_image(cat, 192)
    s = [engine.encode_image(style_a, 192)]
    plain, kept = engine.render(c, s), engine.render(c, s, keep_colors=True)
    chroma = lambda im: np.asarray(im.convert("YCbCr"), dtype=float)[..., 1:]  # noqa: E731
    target = chroma(c.image)
    assert np.abs(chroma(kept) - target).mean() < np.abs(chroma(plain) - target).mean()


def test_encoding_cache_reuses_results(engine, cat):
    engine._cache.clear()
    first = engine.encode_image(cat, 128)
    assert engine.encode_image(cat, 128) is first
    assert engine.encode_image(cat, 160) is not first  # different size = different key


def test_cache_is_bounded(engine):
    engine._cache.clear()
    for i in range(engine._cache_size + 5):
        engine.encode_image(random_image(48, 48, seed=i), 48)
    assert len(engine._cache) == engine._cache_size


def test_style_scale_changes_the_texture(engine, cat, style_a):
    fine = arr(engine.stylize(cat, style_a, size=192, style_scale=0.5))
    bold = arr(engine.stylize(cat, style_a, size=192, style_scale=1.5))
    assert np.abs(fine - bold).mean() > 5


def test_invalid_arguments_raise(engine, cat, style_a):
    with pytest.raises(ValueError):
        engine.stylize(cat, style_a, size=96, alpha=1.5)
    with pytest.raises(ValueError):
        engine.stylize(cat, [style_a], weights=[1, 2], size=96)
