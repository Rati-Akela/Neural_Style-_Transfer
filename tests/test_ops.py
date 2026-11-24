import numpy as np
import pytest
from PIL import Image

from adain import ops

RNG = np.random.default_rng(0)


def feats(n=1, c=8, h=6, w=7, scale=1.0, shift=0.0):
    return (RNG.normal(size=(n, c, h, w)) * scale + shift).astype(np.float32)


def test_calc_mean_std_matches_numpy():
    x = feats(2, 4)
    mean, std = ops.calc_mean_std(x)
    assert mean.shape == std.shape == (2, 4, 1, 1)
    np.testing.assert_allclose(mean[:, :, 0, 0], x.mean(axis=(2, 3)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std[:, :, 0, 0], np.sqrt(x.var(axis=(2, 3)) + ops.EPS), rtol=1e-5)


def test_calc_mean_std_rejects_wrong_rank():
    with pytest.raises(ValueError):
        ops.calc_mean_std(np.zeros((3, 4, 5)))


def test_adain_gives_content_the_style_statistics():
    content, style = feats(scale=1.0), feats(scale=3.0, shift=5.0)
    out = ops.adain(content, style)
    c_mean, c_std = ops.calc_mean_std(out)
    s_mean, s_std = ops.calc_mean_std(style)
    np.testing.assert_allclose(c_mean, s_mean, atol=1e-4)
    np.testing.assert_allclose(c_std, s_std, rtol=1e-3)


def test_adain_ignores_content_scale_and_shift():
    content, style = feats(), feats(scale=2.0, shift=1.0)
    np.testing.assert_allclose(ops.adain(content * 7 + 3, style), ops.adain(content, style), atol=1e-3)


def test_adain_is_identity_when_style_is_content():
    x = feats()
    np.testing.assert_allclose(ops.adain(x, x), x, atol=1e-3)


def test_style_may_have_a_different_spatial_size():
    assert ops.adain(feats(h=6, w=7), feats(h=11, w=3)).shape == (1, 8, 6, 7)


def test_blend_with_one_style_equals_adain():
    c, s = feats(), feats(scale=2)
    np.testing.assert_allclose(ops.blend_adain(c, [s]), ops.adain(c, s), atol=1e-6)


def test_blend_weights_select_the_style():
    c, a, b = feats(), feats(scale=2), feats(scale=3, shift=2)
    np.testing.assert_allclose(ops.blend_adain(c, [a, b], [1, 0]), ops.adain(c, a), atol=1e-6)
    np.testing.assert_allclose(ops.blend_adain(c, [a, b], [0, 1]), ops.adain(c, b), atol=1e-6)
    half = ops.blend_adain(c, [a, b], [3, 3])  # weights are normalised
    np.testing.assert_allclose(half, 0.5 * ops.adain(c, a) + 0.5 * ops.adain(c, b), atol=1e-6)


@pytest.mark.parametrize("n,w", [(2, [1]), (2, [-1, 2]), (2, [0, 0]), (0, None)])
def test_invalid_blend_weights(n, w):
    with pytest.raises(ValueError):
        ops.normalise_weights(n, w)


def test_mix_endpoints_and_validation():
    s, c = feats(), feats()
    np.testing.assert_array_equal(ops.mix(s, c, 1.0), s)
    np.testing.assert_array_equal(ops.mix(s, c, 0.0), c)
    np.testing.assert_allclose(ops.mix(s, c, 0.25), 0.25 * s + 0.75 * c, atol=1e-6)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            ops.mix(s, c, bad)


@pytest.mark.parametrize("w,h", [(500, 333), (333, 500), (1, 1), (4000, 10), (512, 512)])
def test_fit_size_is_multiple_of_8_and_bounded(w, h):
    nw, nh = ops.fit_size(w, h, 512)
    assert nw % 8 == 0 and nh % 8 == 0
    assert min(nw, nh) >= ops.MIN_SIDE and max(nw, nh) <= 512 + 8


def test_fit_size_keeps_aspect_ratio():
    nw, nh = ops.fit_size(1200, 800, 512)
    assert abs(nw / nh - 1.5) < 0.05


def test_prepare_image_flattens_transparency_onto_white():
    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # fully transparent
    out = ops.prepare_image(rgba, 64)
    assert out.mode == "RGB" and out.getpixel((10, 10)) == (255, 255, 255)


def test_prepare_image_handles_grayscale_and_palette():
    assert ops.prepare_image(Image.new("L", (50, 40), 128), 64).mode == "RGB"
    assert ops.prepare_image(Image.new("P", (50, 40)), 64).mode == "RGB"


def test_array_roundtrip():
    img = Image.fromarray(RNG.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(ops.to_image(ops.to_array(img))), np.asarray(img))
    assert ops.to_array(img).shape == (1, 3, 24, 32)


def test_to_image_clips_out_of_range():
    arr = np.full((1, 3, 8, 8), 2.0, dtype=np.float32)
    arr[0, 1] = -1.0
    px = ops.to_image(arr).getpixel((0, 0))
    assert px == (255, 0, 255)


def test_preserve_color_keeps_luma_from_style_and_chroma_from_content():
    stylized = Image.fromarray(RNG.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    content = Image.new("RGB", (32, 32), (200, 40, 40))
    out = ops.preserve_color(stylized, content)
    y_out = np.asarray(out.convert("YCbCr"))[..., 0].astype(int)
    y_sty = np.asarray(stylized.convert("YCbCr"))[..., 0].astype(int)
    cb_out = np.asarray(out.convert("YCbCr"))[..., 1].astype(int)
    cb_con = np.asarray(content.convert("YCbCr"))[..., 1].astype(int)
    assert np.abs(y_out - y_sty).mean() < 6
    assert np.abs(cb_out - cb_con).mean() < 12
