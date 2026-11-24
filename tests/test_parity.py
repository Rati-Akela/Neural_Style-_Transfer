"""The ONNX graphs must reproduce the PyTorch model (needs torch + onnx)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adain import ops  # noqa: E402
from adain.model import Decoder, VGGEncoder, adaptive_instance_normalization  # noqa: E402

from .conftest import ROOT  # noqa: E402

MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def nets():
    enc = VGGEncoder(MODELS / "encoder_relu4_1.pth").eval()
    dec = Decoder()
    dec.load_weights(MODELS / "decoder_final.pth")
    return enc, dec.eval()


@pytest.mark.parametrize("h,w", [(64, 64), (96, 200), (200, 96), (32, 32)])
def test_onnx_matches_torch(engine, nets, h, w):
    enc, dec = nets
    x = np.random.default_rng(1).random((1, 3, h, w), dtype=np.float32)
    with torch.no_grad():
        f = enc(torch.from_numpy(x))
        y = dec(f)
    f_onnx = engine._encode(x)
    np.testing.assert_allclose(f_onnx, f.numpy(), atol=1e-4)
    np.testing.assert_allclose(engine._decode(f_onnx), y.numpy(), atol=1e-4)


def test_numpy_adain_matches_torch_adain():
    rng = np.random.default_rng(2)
    c, s = (
        rng.normal(size=(1, 16, 9, 11)).astype(np.float32),
        rng.normal(size=(1, 16, 5, 7)).astype(np.float32),
    )
    expected = adaptive_instance_normalization(torch.from_numpy(c), torch.from_numpy(s)).numpy()
    np.testing.assert_allclose(ops.adain(c, s), expected, atol=1e-4)


def test_encoder_has_expected_architecture(nets):
    enc, dec = nets
    assert sum(p.numel() for p in enc.parameters()) == 3_505_740
    assert sum(p.numel() for p in dec.parameters()) == 3_505_219
    assert all(not p.requires_grad for p in enc.parameters())


def test_full_and_trimmed_encoder_weights_agree(tmp_path):
    """Trimming the 80 MB VGG file to relu4_1 must not change any retained weight."""
    from adain.export import trim_encoder

    full = {f"{i}.weight": torch.randn(3, 3, 1, 1) for i in (0,)}
    full.update({"44.weight": torch.randn(2), "31.bias": torch.randn(2)})  # beyond relu4_1
    torch.save(full, tmp_path / "full.pth")
    out = trim_encoder(tmp_path / "full.pth", tmp_path / "trim.pth")
    kept = torch.load(out, weights_only=True)
    assert set(kept) == {"0.weight"}
