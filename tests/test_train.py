"""A real (tiny) training run: decoder weights must change and the loss must be finite."""

import pytest

torch = pytest.importorskip("torch")

from adain import train as train_mod  # noqa: E402
from adain.model import Decoder  # noqa: E402

from .conftest import random_image  # noqa: E402


def test_training_runs_and_updates_the_decoder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for kind in ("content", "style"):
        d = tmp_path / kind
        d.mkdir()
        for i in range(4):
            random_image(80, 80, seed=i + (10 if kind == "style" else 0)).save(d / f"{i}.png")
    args = train_mod.parse_args([
        "--content-dir", str(tmp_path / "content"), "--style-dir", str(tmp_path / "style"),
        "--epochs", "2", "--batch-size", "2", "--final-size", "64", "--content-size", "72",
        "--style-size", "72", "--save-interval", "1", "--experiment", "t",
    ])  # fmt: skip
    losses = train_mod.train(args)
    assert all(v == v and v > 0 for v in losses.values())  # finite, positive
    out = tmp_path / "experiment" / "t"
    assert (out / "decoder_final.pth").exists() and (out / "decoder_1.pth").exists()
    assert (out / "args.txt").exists() and (out / "output_2.png").exists()
    fresh, trained = Decoder(), Decoder()
    trained.load_weights(out / "decoder_final.pth")
    first = next(iter(fresh.state_dict()))
    assert not torch.equal(fresh.state_dict()[first], trained.state_dict()[first])


def test_empty_folders_fail_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c").mkdir()
    (tmp_path / "s").mkdir()
    args = train_mod.parse_args(["--content-dir", str(tmp_path / "c"), "--style-dir", str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="No full batches"):
        train_mod.train(args)
