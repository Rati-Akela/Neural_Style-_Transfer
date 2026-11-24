"""Model packaging utilities.

python -m adain.export trim  vgg_normalised.pth        # -> models/encoder_relu4_1.pth
python -m adain.export onnx                             # -> models/encoder.onnx, decoder.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import _N_LAYERS, Decoder, VGGEncoder

MODELS = Path(__file__).resolve().parent.parent / "models"


def trim_encoder(full_weights: Path, out: Path) -> Path:
    """Keep only the layers up to relu4_1 (~14 MB instead of ~80 MB)."""
    state = torch.load(full_weights, map_location="cpu", weights_only=True)
    trimmed = {k: v for k, v in state.items() if int(k.split(".")[0]) < _N_LAYERS}
    torch.save(trimmed, out)
    return out


def export_onnx(models_dir: Path = MODELS, opset: int = 17) -> tuple[Path, Path]:
    encoder = VGGEncoder(models_dir / "encoder_relu4_1.pth").eval()
    decoder = Decoder()
    decoder.load_weights(models_dir / "decoder_final.pth")
    decoder.eval()

    enc_path, dec_path = models_dir / "encoder.onnx", models_dir / "decoder.onnx"
    img = torch.rand(1, 3, 64, 64)
    feats = torch.rand(1, 512, 8, 8)
    kwargs = {
        "opset_version": opset,
        "dynamo": False,
    }  # TorchScript exporter: small, stable graphs
    torch.onnx.export(
        encoder, img, str(enc_path), input_names=["image"], output_names=["features"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "features": {2: "fh", 3: "fw"}}, **kwargs,
    )  # fmt: skip
    torch.onnx.export(
        decoder, feats, str(dec_path), input_names=["features"], output_names=["image"],
        dynamic_axes={"features": {2: "fh", 3: "fw"}, "image": {2: "h", 3: "w"}}, **kwargs,
    )  # fmt: skip
    return enc_path, dec_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trim", help="trim a full vgg_normalised.pth to relu4_1")
    t.add_argument("weights", type=Path)
    t.add_argument("--out", type=Path, default=MODELS / "encoder_relu4_1.pth")
    sub.add_parser("onnx", help="export encoder and decoder to ONNX")
    args = parser.parse_args()
    if args.cmd == "trim":
        print("wrote", trim_encoder(args.weights, args.out))
    else:
        print("wrote", *export_onnx())


if __name__ == "__main__":
    main()
