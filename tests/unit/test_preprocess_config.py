from __future__ import annotations

import pytest

from pipeline.config import load_config

pytestmark = pytest.mark.unit


def test_preprocess_defaults():
    cfg = load_config({})
    assert cfg.preprocess_poll_seconds == 15.0
    assert cfg.preprocess_once is False


@pytest.mark.parametrize("raw,expected", [("0", False), ("1", True), (" 1 ", True)])
def test_preprocess_once_values(raw, expected):
    assert load_config({"PREPROCESS_ONCE": raw}).preprocess_once is expected


def test_preprocess_poll_override():
    assert load_config({"PREPROCESS_POLL_SECONDS": "0.2"}).preprocess_poll_seconds == 0.2


@pytest.mark.parametrize("var,bad", [
    ("PREPROCESS_POLL_SECONDS", "0"),
    ("PREPROCESS_POLL_SECONDS", "-1"),
    ("PREPROCESS_POLL_SECONDS", "nan"),
    ("PREPROCESS_POLL_SECONDS", "abc"),
    ("PREPROCESS_ONCE", "2"),
    ("PREPROCESS_ONCE", "yes"),
    ("PREPROCESS_ONCE", "true"),
    ("PREPROCESS_ONCE", ""),
])
def test_bad_values_raise_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})
