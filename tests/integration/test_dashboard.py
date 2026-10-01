"""Model Pulse against real pipeline output. Stages run as subprocesses; the page renders headlessly."""
from __future__ import annotations

import asyncio
import csv
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd
import pytest

from pipeline import pulse
from pipeline.config import PROJECT_ROOT, load_config
from tests.conftest import clean_env

pytestmark = pytest.mark.integration

DASHBOARD = str(PROJECT_ROOT / "pipeline" / "dashboard.py")
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _stage(module, data, **extra):
    env = clean_env(PYTHONPATH=str(PROJECT_ROOT), DASHBITE_DATA_DIR=str(data), **extra)
    result = subprocess.run([sys.executable, "-u", "-m", module], cwd=data.parent, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "Traceback" not in result.stderr, result.stderr
    return result


def _feed(data, batches, seed=7, **extra):
    extra.setdefault("SIM_CLOCK_SPEED", "1")  # the page reads per real minute; the default is 300x
    _stage("pipeline.simulator", data, SIM_MAX_BATCHES=str(batches), SIM_INTERVAL_SECONDS="0.1",
           SIM_SEED=str(seed), **extra)
    _stage("pipeline.preprocess", data, PREPROCESS_ONCE="1")


def _pipeline(data, batches=30, **extra):
    _feed(data, batches, **extra)
    _stage("pipeline.train", data, TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="100")
    _stage("pipeline.infer", data, INFER_ONCE="1")


def _app(data, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for name in os.environ:
        if name in clean_env.__globals__["ENV_VARS"]:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHBITE_DATA_DIR", str(data))
    return AppTest.from_file(DASHBOARD, default_timeout=60)


def _rows(folder, prefix, column=None):
    values = []
    for path in sorted(folder.glob(f"{prefix}*.csv")):
        with open(path, newline="") as fh:
            values += [row[column] if column else row for row in csv.DictReader(fh)]
    return values


def test_empty_data_renders_empty_states(tmp_path, monkeypatch):
    at = _app(tmp_path / "data", monkeypatch).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["—", "—", "—"]
    assert [i.value for i in at.info] == [
        "No orders yet — start the pipeline with make run",
        "No predictions yet — waiting for the first model",
        "No rows dropped in the last 60 minutes",
    ]
    assert len(at.get("arrow_vega_lite_chart")) == 0
    assert not (tmp_path / "data").exists()  # the page never creates data/


def test_live_data_numbers_match_the_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _pipeline(data)
    at = _app(data, monkeypatch).run()
    assert not at.exception and not at.warning  # real-time clock: no fast-clock warning
    assert all(m.value != "—" for m in at.metric)

    cfg = load_config({"DASHBITE_DATA_DIR": str(data)})
    view = pulse.build_view(pulse.load_artifacts(cfg, pulse.Cache()), cfg,
                            now=pd.Timestamp.now(tz="UTC"))
    feature_rows = len(_rows(data / "features", "features_"))
    assert int(view.volume.sum()) == feature_rows  # 3 real seconds of orders, all in the window
    assert at.metric[0].value == f"{feature_rows:,}"

    flags = [int(v) for v in _rows(data / "predictions", "predictions_", "predicted_late")]
    assert view.numbers["flag_rate"] == pytest.approx(sum(flags) / len(flags) * 100)
    assert at.metric[2].value == f"{sum(flags) / len(flags) * 100:.0f}%"


def test_fast_clock_feed_shows_one_warning(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _feed(data, 10, SIM_CLOCK_SPEED="3600")  # 10 batches x 6 simulated minutes: an hour ahead
    at = _app(data, monkeypatch).run()
    assert not at.exception
    assert len(at.warning) == 1 and "fast simulated clock" in at.warning[0].value


def test_refresh_picks_up_new_files(tmp_path, monkeypatch, capfd):
    data = tmp_path / "data"
    _feed(data, 5)
    at = _app(data, monkeypatch).run()
    before = int(at.metric[0].value.replace(",", ""))
    capfd.readouterr()

    known = {p.name for p in (data / "features").glob("features_*.csv")}
    _feed(data, 1, seed=8)
    (new,) = [p for p in (data / "features").glob("features_*.csv") if p.name not in known]
    added = len(_rows(data / "features", new.name[:-4]))

    at.run()
    assert int(at.metric[0].value.replace(",", "")) == before + added
    # One new batch is one features file plus its rejects file.
    assert "(+2 new)" in capfd.readouterr().out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _open_session(port: int) -> None:
    """Connect like a browser tab would, so Streamlit runs the page script once."""
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from tornado.websocket import websocket_connect

    async def connect():
        ws = await websocket_connect(f"ws://localhost:{port}/_stcore/stream",
                                     subprotocols=["streamlit"])
        message = BackMsg()
        message.rerun_script.query_string = ""
        await ws.write_message(message.SerializeToString(), binary=True)
        await asyncio.wait_for(ws.read_message(), 15)
        await asyncio.sleep(2)
        ws.close()

    asyncio.run(connect())


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")
def test_make_dashboard_wiring(tmp_path):
    data = tmp_path / "data"
    _feed(data, 3)
    port = _free_port()
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    proc = subprocess.Popen(
        ["make", "-f", str(PROJECT_ROOT / "Makefile"), "dashboard"], cwd=tmp_path,
        env=clean_env(DASHBITE_DATA_DIR=str(data), DASHBOARD_PORT=str(port)),
        stdout=open(out, "w"), stderr=open(err, "w"), start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, err.read_text()
            try:
                if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health",
                                          timeout=1).read() == b"ok":
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, "dashboard not healthy within 30s"
            time.sleep(0.25)

        _open_session(port)
        deadline = time.monotonic() + 10
        while "[dashboard " not in out.read_text():
            assert time.monotonic() < deadline, "the page script never ran"
            time.sleep(0.2)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)

    stdout = out.read_text()
    assert f"http://localhost:{port}" in stdout
    assert "External URL" not in stdout  # no public-IP lookup leaves the machine
    assert "read " in stdout and "3 feature" in stdout
    assert "Traceback" not in err.read_text()

    bad = subprocess.run(["make", "-f", str(PROJECT_ROOT / "Makefile"), "dashboard"],
                         cwd=tmp_path, env=clean_env(DASHBOARD_PORT="80"),
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and "DASHBOARD_PORT" in bad.stderr
