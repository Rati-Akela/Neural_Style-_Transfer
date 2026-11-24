"""Render the README images from the real model:  python scripts/make_gallery.py"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adain.engine import StyleTransfer  # noqa: E402

CELL, GAP, LABEL_H = 300, 8, 28
FONT = ImageFont.load_default(size=16)


def tile(img: Image.Image, label: str) -> Image.Image:
    """Image scaled to fit a CELL box, with its caption directly underneath."""
    img = img.copy()
    img.thumbnail((CELL, CELL))
    t = Image.new("RGB", (CELL, img.height + LABEL_H), "white")
    t.paste(img, ((CELL - img.width) // 2, 0))
    ImageDraw.Draw(t).text(
        (CELL // 2, img.height + LABEL_H // 2), label, fill=(40, 40, 60), font=FONT, anchor="mm"
    )
    return t


def grid(rows: list[list[Image.Image]]) -> Image.Image:
    """Rows are sized to their tallest tile, so landscape photos leave no dead space."""
    heights = [max(t.height for t in row) for row in rows]
    w = max(len(r) for r in rows) * (CELL + GAP) - GAP
    canvas = Image.new("RGB", (w, sum(heights) + GAP * (len(rows) - 1)), "white")
    y = 0
    for row, h in zip(rows, heights, strict=True):
        for x, t in enumerate(row):
            canvas.paste(t, (x * (CELL + GAP), y))
        y += h + GAP
    return canvas


def main() -> None:
    eng = StyleTransfer()
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    load = lambda p: Image.open(ROOT / p).convert("RGB")  # noqa: E731
    contents = {
        "Photo: cat": load("examples/content/cat.jpg"),
        "Photo: coffee": load("examples/content/coffee.jpg"),
    }
    styles = {
        "La Muse": load("examples/styles/la_muse.jpg"),
        "Mondrian": load("examples/styles/mondrian.jpg"),
        "Seated Nude": load("examples/styles/seated_nude.jpg"),
    }

    # 1. content x style gallery (first row shows the style images themselves)
    rows = [
        [tile(Image.new("RGB", (CELL, CELL), "white"), "")]
        + [tile(s, f"Style: {n}") for n, s in styles.items()]
    ]
    for cname, c in contents.items():
        rows.append([tile(c, cname)] + [tile(eng.stylize(c, s, size=CELL * 2), n) for n, s in styles.items()])
    grid(rows).save(out / "gallery.jpg", quality=90)

    # 2. style strength sweep
    cat, muse = contents["Photo: cat"], styles["La Muse"]
    c, s = eng.encode_image(cat, 512), [eng.encode_image(muse, 512)]
    grid([[tile(eng.render(c, s, alpha=a), f"strength {a:.2f}") for a in (0.0, 0.25, 0.5, 0.75, 1.0)]]).save(
        out / "strength.jpg", quality=90
    )

    # 3. blending two styles
    coffee = contents["Photo: coffee"]
    c = eng.encode_image(coffee, 512)
    a, b = eng.encode_image(styles["Mondrian"], 512), eng.encode_image(styles["La Muse"], 512)
    grid(
        [
            [
                tile(eng.render(c, [a, b], weights=[1 - m, m]), f"mix {m:.2f}")
                for m in (0.0, 0.25, 0.5, 0.75, 1.0)
            ]
        ]
    ).save(out / "blend.jpg", quality=90)

    # 4. colour preservation
    plain, kept = eng.render(c, [b]), eng.render(c, [b], keep_colors=True)
    grid([[tile(coffee, "Photo"), tile(plain, "Stylised"), tile(kept, "Keep colours")]]).save(
        out / "colors.jpg", quality=90
    )
    print("wrote", *sorted(p.name for p in out.glob("*.jpg")))


if __name__ == "__main__":
    main()
