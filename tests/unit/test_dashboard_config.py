from __future__ import annotations

import pytest

from pipeline.config import load_config, main

pytestmark = pytest.mark.unit


def test_dashboard_defaults():
    cfg = load_config({})
    assert (cfg.dashboard_window_minutes, cfg.dashboard_bucket_minutes,
            cfg.dashboard_refresh_seconds, cfg.dashboard_port) == (60, 1, 15.0, 8501)


def test_dashboard_overrides():
    cfg = load_config({"DASHBOARD_WINDOW_MINUTES": "1440", "DASHBOARD_BUCKET_MINUTES": "15",
                       "DASHBOARD_REFRESH_SECONDS": "2.5", "DASHBOARD_PORT": "8600"})
    assert (cfg.dashboard_window_minutes, cfg.dashboard_bucket_minutes,
            cfg.dashboard_refresh_seconds, cfg.dashboard_port) == (1440, 15, 2.5, 8600)


@pytest.mark.parametrize("env,var", [
    ({"DASHBOARD_WINDOW_MINUTES": "0"}, "DASHBOARD_WINDOW_MINUTES"),
    ({"DASHBOARD_WINDOW_MINUTES": "abc"}, "DASHBOARD_WINDOW_MINUTES"),
    ({"DASHBOARD_BUCKET_MINUTES": "7"}, "DASHBOARD_BUCKET_MINUTES"),  # doesn't divide 60
    ({"DASHBOARD_BUCKET_MINUTES": "0"}, "DASHBOARD_BUCKET_MINUTES"),
    ({"DASHBOARD_WINDOW_MINUTES": "1440", "DASHBOARD_BUCKET_MINUTES": "1"},
     "DASHBOARD_BUCKET_MINUTES"),  # 1440 buckets > 288
    ({"DASHBOARD_REFRESH_SECONDS": "0"}, "DASHBOARD_REFRESH_SECONDS"),
    ({"DASHBOARD_REFRESH_SECONDS": "nan"}, "DASHBOARD_REFRESH_SECONDS"),
    ({"DASHBOARD_PORT": "80"}, "DASHBOARD_PORT"),
    ({"DASHBOARD_PORT": "70000"}, "DASHBOARD_PORT"),
    ({"DASHBOARD_PORT": "abc"}, "DASHBOARD_PORT"),
])
def test_bad_values_raise_naming_variable(env, var):
    with pytest.raises(ValueError, match=var):
        load_config(env)


def test_config_get_prints_one_value(monkeypatch, capsys):
    monkeypatch.setenv("DASHBOARD_PORT", "8600")
    assert main(["get", "dashboard_port"]) == 0
    assert capsys.readouterr().out == "8600\n"
    assert main(["get", "nope"]) == 2
