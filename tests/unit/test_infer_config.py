from __future__ import annotations

import pytest

from pipeline.config import load_config

pytestmark = pytest.mark.unit


def test_infer_defaults_and_overrides():
    cfg = load_config({})
    assert (cfg.infer_poll_seconds, cfg.infer_once) == (15.0, False)
    cfg = load_config({"INFER_POLL_SECONDS": "0.2", "INFER_ONCE": "1"})
    assert (cfg.infer_poll_seconds, cfg.infer_once) == (0.2, True)


@pytest.mark.parametrize("var,bad", [
    ("INFER_POLL_SECONDS", "0"),
    ("INFER_POLL_SECONDS", "-1"),
    ("INFER_POLL_SECONDS", "nan"),
    ("INFER_POLL_SECONDS", "abc"),
    ("INFER_ONCE", "2"),
    ("INFER_ONCE", "yes"),
])
def test_bad_values_raise_naming_variable(var, bad):
    with pytest.raises(ValueError, match=var):
        load_config({var: bad})
