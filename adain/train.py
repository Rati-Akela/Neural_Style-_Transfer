"""Train the AdaIN decoder.

    python -m adain.train --content-dir data/content --style-dir data/style --epochs 10

Only the decoder learns; the VGG encoder stays frozen. The loss follows the paper:
content loss  = || E(D(t)) - t ||                       (t = AdaIN output, E = encoder)
style loss    = sum over relu1_1..relu4_1 of || mean/std of E(D(t)) vs. style ||
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .data import ImageFolderDataset, get_transform
from .model import Decoder, VGGEncoder, adaptive_instance_normalization, calc_mean_std

MODELS = Path(__file__).resolve().parent.parent / "models"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--content-dir", type=Path, default=Path("data/content"))
    p.add_argument("--style-dir", type=Path, default=Path("data/style"))
    p.add_argument(
        "--vgg",
        type=Path,
        default=MODELS / "encoder_relu4_1.pth",
        help="encoder weights",
    )
    p.add_argument("--experiment", default="experiment1", help="output goes to experiment/<name>")
    p.add_argument("--final-size", type=int, default=256, help="training crop size")
    p.add_argument("--content-size", type=int, default=512, help="short side before cropping")
    p.add_argument("--style-size", type=int, default=512)
    p.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="resize instead of random-crop",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-decay", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=5.0)
    p.add_argument("--log-interval", type=int, default=1, help="epochs between log lines")
    p.add_argument("--save-interval", type=int, default=2, help="epochs between checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--decoder-path", type=Path, default=None, help="checkpoint to resume from")
    p.add_argument("--optimizer-path", type=Path, default=None)
    return p.parse_args(argv)


def style_loss(gen_feats, style_feats) -> torch.Tensor:
    total = 0.0
    for g, s in zip(gen_feats, style_feats, strict=True):
        g_mean, g_std = calc_mean_std(g)
        s_mean, s_std = calc_mean_std(s)
        total = total + F.mse_loss(g_mean, s_mean) + F.mse_loss(g_std, s_std)
    return total


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.txt").write_text("".join(f"{k}: {v}\n" for k, v in vars(args).items()))

    def loader(folder, size):
        ds = ImageFolderDataset(folder, get_transform(size, args.crop, args.final_size))
        if len(ds) < args.batch_size:
            raise SystemExit(
                f"No full batches found in {folder}: {len(ds)} image(s) (.jpg/.jpeg/.png), "
                f"batch size {args.batch_size}."
            )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          pin_memory=device.type == "cuda")  # fmt: skip

    content_loader = loader(args.content_dir, args.content_size)
    style_loader = loader(args.style_dir, args.style_size)
    n_batches = min(len(content_loader), len(style_loader))
    print(f"{n_batches} batches per epoch")

    encoder = VGGEncoder(args.vgg).to(device).eval()
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0 / (1.0 + args.lr_decay * e))
    if args.resume:
        decoder.load_weights(args.decoder_path)
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))

    last = {"loss": float("nan"), "content": float("nan"), "style": float("nan")}
    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "content": 0.0, "style": 0.0}
        bar = tqdm(zip(content_loader, style_loader, strict=False), total=n_batches)
        for content, style in bar:
            content, style = content.to(device), style.to(device)
            with torch.no_grad():
                c_feat = encoder(content)
                s_feats = encoder(style, all_levels=True)
                target = adaptive_instance_normalization(c_feat, s_feats[-1])
            generated = decoder(target)
            g_feats = encoder(generated, all_levels=True)

            loss_c = F.mse_loss(g_feats[-1], target) * args.content_weight
            loss_s = style_loss(g_feats, s_feats) * args.style_weight
            loss = loss_c + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_description(
                f"loss {loss.item():.4f} (content {loss_c.item():.4f}, style {loss_s.item():.4f})"
            )
            totals["loss"] += loss.item()
            totals["content"] += loss_c.item()
            totals["style"] += loss_s.item()
        scheduler.step()
        last = {k: v / n_batches for k, v in totals.items()}

        if epoch % args.log_interval == 0:
            tqdm.write(f"epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in last.items()))
        if epoch % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_{epoch}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_{epoch}.pth")
            with torch.no_grad():
                save_image(torch.cat([content, style, generated]), save_dir / f"output_{epoch}.png",
                           nrow=args.batch_size)  # fmt: skip

    torch.save(decoder.state_dict(), save_dir / "decoder_final.pth")
    return last


if __name__ == "__main__":
    train(parse_args())
