"""Headless UI tests with Streamlit's AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def at():
    return AppTest.from_file(APP, default_timeout=90).run()


def click(at, label):
    return next(b for b in at.button if b.label == label).click().run()


def test_first_load_shows_examples_and_no_result(at):
    assert not at.exception
    assert at.title[0].value == "Neural Style Transfer"
    assert "result" not in at.session_state
    assert len(at.image) == 2  # one example photo, one example style


def test_stylize_example_produces_a_result(at):
    at = click(at, "Stylize")
    assert not at.exception
    res = at.session_state["result"]
    assert res["image"].size == res["original"].size
    assert res["seconds"] > 0 and res["name"] == "Cat_stylized.png"


def test_settings_change_the_output(at):
    base = click(at, "Stylize").session_state["result"]["image"]
    at.slider[0].set_value(0.0)  # style strength 0 -> plain reconstruction
    weak = click(at, "Stylize").session_state["result"]["image"]
    assert list(base.tobytes()) != list(weak.tobytes())


def test_blend_mode_requires_second_style_and_runs(at):
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert len(at.image) == 3  # photo + two styles
    at = click(at, "Stylize")
    assert not at.exception and "result" in at.session_state


def test_sweep_renders_five_strengths(at):
    at = click(at, "Stylize")
    before = len(at.image)
    at = click(at, "Render sweep")
    assert not at.exception and len(at.image) == before + 5


def test_blend_defaults_to_two_different_styles(at):
    at.toggle[0].set_value(True).run()
    selects = {s.key: s.value for s in at.selectbox}
    assert selects["style_ex"] != selects["style_b_ex"]
