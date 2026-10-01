"""Simulator -> preprocess -> train as separate real processes, handing off through data/."""
from __future__ import annotations

import csv
import json
import shutil
import signal
import subprocess
import sys
import time

import pytest

from pipeline.config import PROJECT_ROOT
from tests.conftest import clean_env

pytestmark = pytest.mark.integration

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _env(data_dir, **extra):
    return clean_env(PYTHONPATH=str(PROJECT_ROOT), DASHBITE_DATA_DIR=str(data_dir), **extra)


def _run(module, data_dir, cwd, **extra):
    return subprocess.run(
        [sys.executable, "-u", "-m", module],
        cwd=cwd, env=_env(data_dir, **extra), capture_output=True, text=True, timeout=120,
    )


def _simulate_and_preprocess(data, cwd, batches, seed):
    sim = _run("pipeline.simulator", data, cwd, SIM_MAX_BATCHES=str(batches), BATCH_SIZE="20",
               SIM_SEED=str(seed), SIM_CLOCK_SPEED="3600", SIM_INTERVAL_SECONDS="0.1")
    assert sim.returncode == 0, sim.stderr
    pre = _run("pipeline.preprocess", data, cwd, PREPROCESS_ONCE="1")
    assert pre.returncode == 0, pre.stderr


def _feature_rows(data):
    total = 0
    for path in (data / "features").glob("features_*.csv"):
        with open(path, newline="") as fh:
            total += sum(1 for _ in csv.DictReader(fh))
    return total


def test_pipeline_publishes_first_model(tmp_path):
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=15, seed=7)

    result = _run("pipeline.train", data, tmp_path, TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="100")
    assert result.returncode == 0, result.stderr
    assert "published model_v0001" in result.stdout
    assert "Traceback" not in result.stderr and "Warning" not in result.stderr
    assert sorted(p.name for p in (data / "models").iterdir()) == [
        "model_v0001.joblib", "model_v0001.json",
    ]
    sidecar = json.loads((data / "models" / "model_v0001.json").read_text())
    assert sidecar["rows_total"] == _feature_rows(data)


def test_threshold_not_met_writes_nothing(tmp_path):
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=15, seed=7)
    result = _run("pipeline.train", data, tmp_path, TRAIN_ONCE="1",
                  TRAIN_EVERY_N_EVENTS="100000")
    assert result.returncode == 0, result.stderr
    assert "waiting:" in result.stdout
    assert list((data / "models").iterdir()) == []


def test_second_version_leaves_first_untouched(tmp_path):
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=15, seed=7)
    assert _run("pipeline.train", data, tmp_path, TRAIN_ONCE="1",
                TRAIN_EVERY_N_EVENTS="100").returncode == 0
    v1 = {p.name: p.read_bytes() for p in (data / "models").iterdir()}
    rows_at_v1 = json.loads(v1["model_v0001.json"])["rows_total"]

    _simulate_and_preprocess(data, tmp_path, batches=10, seed=8)
    result = _run("pipeline.train", data, tmp_path, TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="100")
    assert result.returncode == 0, result.stderr
    assert "published model_v0002" in result.stdout

    for name, content in v1.items():
        assert (data / "models" / name).read_bytes() == content
    v2 = json.loads((data / "models" / "model_v0002.json").read_text())
    assert v2["rows_new"] == _feature_rows(data) - rows_at_v1
    assert v2["rows_total"] == _feature_rows(data)


def test_live_loop_publishes_and_stops_on_sigterm(tmp_path):
    source, data = tmp_path / "source", tmp_path / "data"
    _simulate_and_preprocess(source, tmp_path, batches=10, seed=7)

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pipeline.train"], cwd=tmp_path,
        env=_env(data, TRAIN_POLL_SECONDS="0.2", TRAIN_EVERY_N_EVENTS="50"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not (data / "features").is_dir():
            assert time.monotonic() < deadline and proc.poll() is None, "train didn't start"
            time.sleep(0.05)
        # Drop in feature files while train is watching.
        for path in sorted((source / "features").glob("features_*.csv")):
            shutil.copy(path, data / "features" / path.name)

        sidecar = data / "models" / "model_v0001.json"
        deadline = time.monotonic() + 5
        while not sidecar.exists():
            assert time.monotonic() < deadline, "no model within 5s"
            time.sleep(0.05)

        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "stopped after publishing 1 model (latest v0001)" in out
    assert "Traceback" not in err
    assert not list(data.rglob("*.tmp"))


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")
def test_make_train_wiring(tmp_path):
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=10, seed=7)

    def make(**extra):
        return subprocess.run(
            ["make", "-f", str(PROJECT_ROOT / "Makefile"), "train"],
            cwd=tmp_path, env=clean_env(DASHBITE_DATA_DIR=str(data), **extra),
            capture_output=True, text=True, timeout=120,
        )

    ok = make(TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="100")
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[0].startswith("[train] watching ")
    assert (data / "models" / "model_v0001.json").exists()

    bad = make(TRAIN_POLL_SECONDS="0")
    assert bad.returncode != 0
    assert "TRAIN_POLL_SECONDS" in bad.stderr


def _snapshot(data, exclude):
    return {p.relative_to(data): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in data.rglob("*") if p.is_file() and exclude not in p.relative_to(data).parts}


def test_train_writes_only_to_models(tmp_path):
    # Hard isolation, observed on disk: train publishes into data/models and
    # leaves every other file (raw, features, quality, predictions) untouched.
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=10, seed=3)
    (data / "predictions" / "sentinel.csv").write_text("owned by inference\n")
    before = _snapshot(data, "models")

    result = _run("pipeline.train", data, tmp_path, TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="50")
    assert result.returncode == 0, result.stderr
    assert "published model_v0001" in result.stdout
    assert _snapshot(data, "models") == before
    assert sorted(p.name for p in (data / "models").iterdir()) == [
        "model_v0001.joblib", "model_v0001.json",
    ]


def test_corrupt_sidecar_still_stops_cleanly_on_sigterm(tmp_path):
    # Regression: a damaged sidecar made the stop summary crash with a traceback (exit 1).
    data = tmp_path / "data"
    _simulate_and_preprocess(data, tmp_path, batches=10, seed=3)
    (data / "models").mkdir(exist_ok=True)
    (data / "models" / "model_v0001.json").write_text('{"version": 1, "rows_to')

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pipeline.train"], cwd=tmp_path,
        env=_env(data, TRAIN_POLL_SECONDS="0.2", TRAIN_EVERY_N_EVENTS="50"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not (data / "models" / "model_v0002.json").exists():
            assert proc.poll() is None, proc.communicate()
            assert time.monotonic() < deadline, "v0002 never appeared"
            time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "Traceback" not in err
    assert "ignoring model_v0001.json: unreadable sidecar" in out
    assert "stopped after publishing 1 model (latest v0002)" in out
