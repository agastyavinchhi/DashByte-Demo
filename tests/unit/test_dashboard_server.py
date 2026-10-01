from __future__ import annotations

import socket

import pytest

from pipeline import dashboard_server

pytestmark = pytest.mark.unit


def test_streamlit_argv_keeps_everything_local():
    argv = dashboard_server.streamlit_argv(8600)
    assert argv[:3] == ["streamlit", "run", str(dashboard_server.APP)]
    flags = dict(zip(argv[3::2], argv[4::2]))
    assert flags == {
        "--server.headless": "true", "--server.port": "8600", "--server.address": "localhost",
        "--server.fileWatcherType": "none", "--browser.gatherUsageStats": "false",
    }


def test_busy_port_is_refused_before_streamlit_starts(tmp_data_dir, monkeypatch, capsys):
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        sock.listen()
        port = sock.getsockname()[1]
        monkeypatch.setenv("DASHBOARD_PORT", str(port))
        monkeypatch.setattr(dashboard_server, "streamlit_argv",
                            lambda p: pytest.fail("Streamlit must not start"))
        assert dashboard_server.main() == 1
    assert f"port {port} is already in use" in capsys.readouterr().err


def test_bad_port_is_a_config_error(monkeypatch, capsys):
    monkeypatch.setenv("DASHBOARD_PORT", "80")
    assert dashboard_server.main() == 1
    assert "DASHBOARD_PORT" in capsys.readouterr().err
