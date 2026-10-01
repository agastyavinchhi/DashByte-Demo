"""Inference as its own process, consuming only artifacts on disk. Every stage is a subprocess."""
from __future__ import annotations

import csv
import os
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


def _env(data, **extra):
    return clean_env(PYTHONPATH=str(PROJECT_ROOT), DASHBITE_DATA_DIR=str(data), **extra)


def _run(module, data, cwd, **extra):
    result = subprocess.run([sys.executable, "-u", "-m", module], cwd=cwd,
                            env=_env(data, **extra), capture_output=True, text=True, timeout=120)
    assert "Traceback" not in result.stderr, result.stderr
    return result


def _feed(data, cwd, batches, seed):
    assert _run("pipeline.simulator", data, cwd, SIM_MAX_BATCHES=str(batches), SIM_SEED=str(seed),
                SIM_CLOCK_SPEED="3600", SIM_INTERVAL_SECONDS="0.1").returncode == 0
    assert _run("pipeline.preprocess", data, cwd, PREPROCESS_ONCE="1").returncode == 0


def _train_once(data, cwd):
    result = _run("pipeline.train", data, cwd, TRAIN_ONCE="1", TRAIN_EVERY_N_EVENTS="100")
    assert result.returncode == 0 and "published model_v" in result.stdout, result.stdout
    return result


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _key(path, prefix):
    return path.name[len(prefix):]


def _models_snapshot(data):
    return {p.name: (p.read_bytes(), os.stat(p).st_mtime_ns) for p in (data / "models").iterdir()}


def _start_infer(data, cwd):
    return subprocess.Popen([sys.executable, "-u", "-m", "pipeline.infer"], cwd=cwd,
                            env=_env(data, INFER_POLL_SECONDS="0.2"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_until(condition, timeout, what):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.05)


def _stop(proc):
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=15)
    assert proc.returncode == 0, err
    assert "Traceback" not in err
    return out


def test_full_chain(tmp_path):
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=15, seed=7)
    _train_once(data, tmp_path)

    result = _run("pipeline.infer", data, tmp_path, INFER_ONCE="1")
    assert result.returncode == 0, result.stderr
    assert "found model_v0001" in result.stdout

    features = sorted((data / "features").glob("features_*.csv"))
    predictions = sorted((data / "predictions").glob("predictions_*.csv"))
    assert [_key(p, "features_") for p in features] == [_key(p, "predictions_") for p in predictions]
    for f, p in zip(features, predictions):
        rows = _read(p)
        assert [r["order_id"] for r in rows] == [r["order_id"] for r in _read(f)]
        assert {r["checkpoint_id"] for r in rows} <= {"model_v0001"}
    assert result.stdout.count("scored with model_v0001") == len(features)


def test_waits_cleanly_then_starts_when_train_publishes(tmp_path):
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=10, seed=7)
    proc = _start_infer(data, tmp_path)
    try:
        time.sleep(1.5)
        assert proc.poll() is None, "infer exited while waiting"
        assert list((data / "predictions").iterdir()) == []

        _train_once(data, tmp_path)  # a separate process that exits
        _wait_until(lambda: len(list((data / "predictions").glob("predictions_*.csv"))) == 10,
                    5, "predictions didn't appear within 5s")
        out = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert out.count("waiting for a checkpoint") == 1
    assert "found model_v0001" in out
    assert "stopped after scoring 10 batches" in out


def test_training_does_not_need_to_be_running(tmp_path):
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=10, seed=7)
    _train_once(data, tmp_path)
    # No train process is left: train ran once and exited.
    pgrep = subprocess.run(["pgrep", "-f", f"pipeline.train.*{data}"], capture_output=True)
    assert pgrep.returncode != 0
    models = _models_snapshot(data)

    before = {p.name for p in (data / "features").glob("features_*.csv")}
    _feed(data, tmp_path, batches=3, seed=99)
    new = sorted({p.name for p in (data / "features").glob("features_*.csv")} - before)
    assert len(new) == 3

    result = _run("pipeline.infer", data, tmp_path, INFER_ONCE="1")
    assert result.returncode == 0, result.stderr
    for name in new:
        rows = _read(data / "predictions" / f"predictions_{name[len('features_'):]}")
        assert {r["checkpoint_id"] for r in rows} <= {"model_v0001"}
    assert _models_snapshot(data) == models


def test_newer_checkpoint_is_picked_up_while_running(tmp_path):
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=10, seed=7)
    _train_once(data, tmp_path)
    proc = _start_infer(data, tmp_path)
    try:
        _wait_until(lambda: len(list((data / "predictions").iterdir())) == 10, 10,
                    "first batch of predictions didn't appear")
        early = {p.name: p.read_bytes() for p in (data / "predictions").iterdir()}

        _feed(data, tmp_path, batches=8, seed=8)  # new rows, then v0002, then more rows
        _train_once(data, tmp_path)
        # Identify the last batches by arrival, not by name: each simulator run
        # restarts its simulated clock, so names don't sort across runs.
        before_last = {p.name for p in (data / "features").glob("features_*.csv")}
        _feed(data, tmp_path, batches=3, seed=9)
        last = {p.name for p in (data / "features").glob("features_*.csv")} - before_last
        _wait_until(lambda: len(list((data / "predictions").iterdir())) == 21, 10,
                    "later predictions didn't appear")
        out = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert "loaded model_v0002" in out and "replacing model_v0001" in out
    for name, content in early.items():  # never re-scored
        assert (data / "predictions" / name).read_bytes() == content
    assert len(last) == 3
    for name in last:
        rows = _read(data / "predictions" / f"predictions_{name[len('features_'):]}")
        assert {r["checkpoint_id"] for r in rows} <= {"model_v0002"}


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="run `make install` first")
def test_make_infer_wiring(tmp_path):
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=10, seed=7)  # ~190 clean rows, over train's 100
    _train_once(data, tmp_path)

    def make(**extra):
        return subprocess.run(["make", "-f", str(PROJECT_ROOT / "Makefile"), "infer"],
                              cwd=tmp_path, env=clean_env(DASHBITE_DATA_DIR=str(data), **extra),
                              capture_output=True, text=True, timeout=120)

    ok = make(INFER_ONCE="1")
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[0].startswith("[infer] watching ")
    assert len(list((data / "predictions").glob("predictions_*.csv"))) == 10

    bad = make(INFER_POLL_SECONDS="0")
    assert bad.returncode != 0 and "INFER_POLL_SECONDS" in bad.stderr


def test_models_reset_while_infer_runs_switches_to_the_new_checkpoint(tmp_path):
    # Regression: infer kept scoring with its in-memory model after data/models
    # was emptied, labelling rows with a version name that now meant a new file.
    data = tmp_path / "data"
    _feed(data, tmp_path, batches=6, seed=7)
    predictions = lambda: len(list((data / "predictions").glob("predictions_*.csv")))  # noqa: E731
    proc = _start_infer(data, tmp_path)
    try:
        _train_once(data, tmp_path)
        _wait_until(lambda: predictions() == 6, 10, "first predictions didn't appear")
        old_sidecar = (data / "models" / "model_v0001.json").read_text()
        for path in (data / "models").iterdir():
            path.unlink()
        time.sleep(1.1)  # so the new sidecar's trained_at (whole seconds) differs
        _train_once(data, tmp_path)  # publishes a *new* model_v0001
        _feed(data, tmp_path, batches=1, seed=8)
        _wait_until(lambda: predictions() == 7, 10, "post-reset batch wasn't scored")
        out = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert (data / "models" / "model_v0001.json").read_text() != old_sidecar
    assert "model_v0001 is no longer the file on disk" in out
    assert out.count("found model_v0001") == 2
