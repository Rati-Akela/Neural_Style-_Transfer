"""Neural Style Transfer - Streamlit app.   Run with:  streamlit run app.py"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from adain.engine import StyleTransfer

ROOT = Path(__file__).parent
MAX_SIZE = int(os.environ.get("NST_MAX_SIZE", "640"))  # longest output edge; memory grows ~size^2
SIZES = [s for s in (256, 384, 512, 640, 768, 1024) if s <= max(MAX_SIZE, 256)]
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs

CONTENT_EXAMPLES = {
    "Cat": ROOT / "examples/content/cat.jpg",
    "Coffee": ROOT / "examples/content/coffee.jpg",
}
STYLE_EXAMPLES = {
    "La Muse (Picasso)": ROOT / "examples/styles/la_muse.jpg",
    "Mondrian": ROOT / "examples/styles/mondrian.jpg",
    "Seated Nude (Picasso)": ROOT / "examples/styles/seated_nude.jpg",
}

st.set_page_config(page_title="Neural Style Transfer", page_icon=":material/palette:", layout="wide")
st.markdown(
    "<style>#MainMenu, footer {visibility: hidden;} .block-container {padding-top: 2rem; max-width: 1150px;}"
    "div.stButton > button {border-radius: 1.5rem; font-weight: 600;}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_engine() -> StyleTransfer:
    return StyleTransfer()


def pick_image(
    label: str, examples: dict[str, Path], key: str, default: int = 0
) -> tuple[Image.Image | None, str]:
    """Example dropdown or file upload. Returns (image, display name)."""
    source = st.radio(label, ["Example", "Upload"], horizontal=True, key=f"{key}_src")
    if source == "Example":
        name = st.selectbox(
            "Example", list(examples), index=default, key=f"{key}_ex", label_visibility="collapsed"
        )
        return Image.open(examples[name]), name
    file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up",
                            label_visibility="collapsed")  # fmt: skip
    if file is None:
        return None, ""
    try:
        img = Image.open(file)
        img.load()
        return img, file.name
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("That file couldn't be read as an image.")
        return None, ""


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    size = st.select_slider(
        "Output size (longest edge, px)",
        options=SIZES,
        value=min(512, SIZES[-1]),
        help="Bigger is sharper but slower and uses more memory.",
    )
    alpha = st.slider(
        "Style strength",
        0.0,
        1.0,
        1.0,
        0.05,
        help="0 = your original photo, 1 = full stylisation. Values in between interpolate.",
    )
    style_scale = st.slider(
        "Brushstroke scale",
        0.5,
        1.5,
        1.0,
        0.25,
        help="Resizes the style image before encoding: smaller = finer detail, larger = bolder strokes.",
    )
    keep_colors = st.checkbox(
        "Keep original colours",
        help="Apply the style's texture but keep the photo's colours.",
    )
    blend = st.toggle("Blend two styles")
    with st.expander("How it works"):
        st.markdown(
            "**AdaIN** (adaptive instance normalisation) encodes both images with a fixed VGG-19, "
            "then re-scales the photo's feature statistics to match the style's:"
        )
        st.latex(r"\mathrm{AdaIN}(x, y) = \sigma(y)\,\frac{x - \mu(x)}{\sigma(x)} + \mu(y)")
        st.markdown(
            "A trained decoder turns the result back into an image, in one forward pass. "
            "Based on [Huang & Belongie, 2017](https://arxiv.org/abs/1703.06868)."
        )

# ---------------------------------------------------------------------------- main
st.title("Neural Style Transfer")
st.caption("Repaint any photo in the style of any artwork - one forward pass of an AdaIN network, "
           "running on ONNX Runtime (no GPU needed).")  # fmt: skip

col_content, col_style = st.columns(2)
with col_content:
    st.subheader("1. Photo")
    content_img, content_name = pick_image("Source", CONTENT_EXAMPLES, "content")
    if content_img:
        st.image(content_img, width="stretch")
with col_style:
    st.subheader("2. Style")
    style_img, style_name = pick_image("Source", STYLE_EXAMPLES, "style")
    if style_img:
        st.image(style_img, width="stretch")

style_b_img, mix = None, 0.5
if blend:
    st.subheader("3. Second style")
    cb1, cb2 = st.columns(2)
    with cb1:
        style_b_img, _ = pick_image("Source", STYLE_EXAMPLES, "style_b", default=1)
        if style_b_img:
            st.image(style_b_img, width="stretch")
    with cb2:
        mix = st.slider("Mix (style 1  <->  style 2)", 0.0, 1.0, 0.5, 0.05)

ready = content_img is not None and style_img is not None and (not blend or style_b_img is not None)
go = st.button("Stylize", type="primary", disabled=not ready)

if go and ready:
    engine = get_engine()
    start = time.perf_counter()
    with st.spinner("Painting..."):
        content = engine.encode_image(content_img, size)
        styles = [engine.encode_image(style_img, max(32, round(size * style_scale)))]
        weights = None
        if blend and style_b_img is not None:
            styles.append(engine.encode_image(style_b_img, max(32, round(size * style_scale))))
            weights = [1.0 - mix, mix]
        result = engine.render(content, styles, weights=weights, alpha=alpha, keep_colors=keep_colors)
    st.session_state["result"] = {
        "image": result, "original": content.image, "seconds": time.perf_counter() - start,
        "content": content, "styles": styles, "weights": weights, "keep_colors": keep_colors,
        "name": f"{Path(content_name).stem or 'photo'}_stylized.png",
    }  # fmt: skip

res = st.session_state.get("result")
if res:
    st.divider()
    st.subheader("Result")
    left, right = st.columns(2)
    left.image(res["original"], caption="Original", width="stretch")
    caption = f"Stylized ({res['seconds']:.1f}s, {res['image'].width}x{res['image'].height})"
    right.image(res["image"], caption=caption, width="stretch")
    st.download_button("Download PNG", png_bytes(res["image"]), res["name"], "image/png")

    with st.expander("Style-strength sweep"):
        st.caption("Same photo and style at five strengths - AdaIN interpolates smoothly in feature space.")
        if st.button("Render sweep"):
            engine = get_engine()
            cols = st.columns(5)
            for col, a in zip(cols, (0.0, 0.25, 0.5, 0.75, 1.0), strict=True):
                img = engine.render(res["content"], res["styles"], weights=res["weights"], alpha=a,
                                    keep_colors=res["keep_colors"])  # fmt: skip
                col.image(img, caption=f"strength {a:.2f}", width="stretch")

st.caption(
    "Model: AdaIN (Huang & Belongie, 2017). VGG-19 encoder with normalised weights from "
    "naoto0804/pytorch-AdaIN; decoder trained with adain/train.py."
)
