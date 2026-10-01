from __future__ import annotations

import csv
import json
import os
import random

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline import infer
from pipeline.config import load_config
from pipeline.infer import (
    PREDICTION_COLUMNS,
    load_checkpoint,
    newest_checkpoint,
    pending_files,
    score_file,
)

pytestmark = pytest.mark.unit

FEATURE_HEADER = ("order_id", "timestamp", "hour", "is_peak", "distance_km", "prep_minutes",
                  "order_value", "was_late")


def fitted_model(columns=("distance_km", "prep_minutes"), seed=0):
    """A small real LogisticRegression; tests build their own, never via pipeline.train."""
    rng = np.random.default_rng(seed)
    X = rng.uniform([0.5, 5, 0][:len(columns)], [12, 30, 1][:len(columns)],
                    size=(300, len(columns)))
    y = (X[:, 0] * 3 + X[:, 1] > 30).astype(int)
    with np.errstate(all="ignore"):
        return LogisticRegression().fit(X, y)


def publish_ckpt(models, version, columns=("distance_km", "prep_minutes"), rows_total=100,
                 sidecar=None, model=None, write_joblib=True):
    models.mkdir(parents=True, exist_ok=True)
    stem = f"model_v{version:04d}"
    if write_joblib:
        joblib.dump({"model": model or fitted_model(columns), "feature_columns": list(columns),
                     "version": version}, models / f"{stem}.joblib")
    data = {"version": version, "model_file": f"{stem}.joblib",
            "feature_columns": list(columns), "rows_total": rows_total, "accuracy": 0.8}
    data.update(sidecar or {})
    (models / f"{stem}.json").write_text(json.dumps(data))
    return models / f"{stem}.json"


def write_features(data_dir, seq, n=5, seed=1, header=FEATURE_HEADER, rows=None):
    folder = data_dir / "features"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"features_20260928T{seq:06d}Z_{seq:04d}.csv"
    rng = random.Random(seed + seq)
    if rows is None:
        rows = [{"order_id": f"ORD-{seq:04d}{i:06d}", "timestamp": "2026-09-28T14:00:00Z",
                 "hour": "14", "is_peak": "0", "distance_km": f"{rng.uniform(0.5, 12):.1f}",
                 "prep_minutes": str(rng.randint(5, 30)), "order_value": "20.00",
                 "was_late": "0"} for i in range(n)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return tuple(reader.fieldnames), list(reader)


@pytest.fixture
def cfg(tmp_data_dir):
    return load_config()


@pytest.fixture
def models(tmp_data_dir):
    return tmp_data_dir / "models"


# ---------- newest_checkpoint ----------

def test_newest_is_none_for_empty_dir(cfg):
    assert newest_checkpoint(cfg) is None


def test_newest_is_highest_version_with_a_sidecar(cfg, models):
    publish_ckpt(models, 1)
    publish_ckpt(models, 3)
    publish_ckpt(models, 2)
    assert newest_checkpoint(cfg).version == 3


def test_newest_ignores_orphan_joblib_bad_json_and_tmp(cfg, models):
    publish_ckpt(models, 1)
    joblib.dump({"model": fitted_model()}, models / "model_v0002.joblib")  # no sidecar
    (models / "model_v0003.json").write_text("{not json")  # damaged sidecar
    (models / ".model_v0004.json.tmp").write_text(json.dumps({"rows_total": 1}))
    assert newest_checkpoint(cfg).version == 1


def test_newest_is_by_version_not_modification_time(cfg, models):
    old = publish_ckpt(models, 1)
    publish_ckpt(models, 2)
    later = os.stat(old).st_mtime + 1000
    os.utime(old, (later, later))
    assert newest_checkpoint(cfg).version == 2


# ---------- load_checkpoint ----------

def test_load_uses_sidecar_feature_columns(cfg, models):
    publish_ckpt(models, 1, columns=("prep_minutes", "distance_km"))
    loaded = load_checkpoint(newest_checkpoint(cfg))
    assert loaded.feature_columns == ("prep_minutes", "distance_km")
    assert loaded.checkpoint_id == "model_v0001"


@pytest.mark.parametrize("breakage,reason", [
    ("corrupt_joblib", "unreadable"),
    ("missing_joblib", "missing"),
    ("column_mismatch", "feature_columns differ"),
    ("no_columns", "no usable feature_columns"),
    ("foreign_model_file", "expected model_v0001.joblib"),
])
def test_load_failures_raise_cannot_load(cfg, models, breakage, reason):
    if breakage == "corrupt_joblib":
        publish_ckpt(models, 1)
        (models / "model_v0001.joblib").write_bytes(b"not a pickle")
    elif breakage == "missing_joblib":
        publish_ckpt(models, 1, write_joblib=False)
    elif breakage == "column_mismatch":
        publish_ckpt(models, 1, sidecar={"feature_columns": ["distance_km", "is_peak"]})
    elif breakage == "no_columns":
        publish_ckpt(models, 1, sidecar={"feature_columns": []})
    else:
        publish_ckpt(models, 1, sidecar={"model_file": "../../elsewhere.joblib"})
    with pytest.raises(infer.CannotLoad, match=reason):
        load_checkpoint(newest_checkpoint(cfg))


def _polls(monkeypatch, n, between=None):
    """Make run() do n polls; between(i) runs after poll i."""
    calls = []

    def fake_sleep(_seconds):
        calls.append(1)
        if between:
            between(len(calls))
        if len(calls) == n:
            raise KeyboardInterrupt

    monkeypatch.setattr(infer.time, "sleep", fake_sleep)


def test_broken_newer_version_keeps_current_model_and_is_not_retried(
        cfg, models, tmp_data_dir, capsys, monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)

    def between(i):
        if i == 1:
            publish_ckpt(models, 2)
            (models / "model_v0002.joblib").write_bytes(b"garbage")
            write_features(tmp_data_dir, 2)

    _polls(monkeypatch, 3, between)
    assert infer.run(cfg, poll_seconds=0.01)[2] == "model_v0001"
    out = capsys.readouterr().out
    assert out.count("cannot load model_v0002: model_v0002.joblib is unreadable") == 1
    _, rows = read(tmp_data_dir / "predictions" / "predictions_20260928T000002Z_0002.csv")
    assert {r["checkpoint_id"] for r in rows} == {"model_v0001"}


def test_fresh_start_falls_back_to_lower_version_that_loads(cfg, models, tmp_data_dir, capsys):
    publish_ckpt(models, 1)
    publish_ckpt(models, 2)
    (models / "model_v0002.joblib").write_bytes(b"garbage")
    write_features(tmp_data_dir, 1)
    assert infer.run(cfg, poll_seconds=0.01, once=True) == (1, 5, "model_v0001")
    out = capsys.readouterr().out
    assert "cannot load model_v0002" in out and "found model_v0001" in out
    assert "waiting for a checkpoint" not in out


def test_nothing_loadable_waits_instead_of_failing(cfg, models, tmp_data_dir, capsys):
    publish_ckpt(models, 1)
    (models / "model_v0001.joblib").write_bytes(b"garbage")
    write_features(tmp_data_dir, 1)
    assert infer.run(cfg, poll_seconds=0.01, once=True) == (0, 0, None)
    assert "waiting for a checkpoint: none of the published models could be loaded" in (
        capsys.readouterr().out)
    assert not list((tmp_data_dir / "predictions").iterdir())


# ---------- score_file ----------

def test_score_file_output_contract(cfg, models, tmp_data_dir):
    publish_ckpt(models, 3)
    loaded = load_checkpoint(newest_checkpoint(cfg))
    features = write_features(tmp_data_dir, 7, n=40)
    result = score_file(features, loaded, cfg)

    assert result.predictions_path.name == "predictions_20260928T000007Z_0007.csv"
    header, rows = read(result.predictions_path)
    assert header == PREDICTION_COLUMNS
    _, inputs = read(features)
    assert [r["order_id"] for r in rows] == [r["order_id"] for r in inputs]
    for row in rows:
        probability = float(row["late_probability"])
        assert 0 <= probability <= 1
        assert len(row["late_probability"].split(".")[1]) == 3
        assert row["predicted_late"] == ("1" if probability >= 0.5 else "0")
        assert row["checkpoint_id"] == "model_v0003"
    assert (result.orders, result.predicted_late) == (40, sum(r["predicted_late"] == "1"
                                                              for r in rows))
    assert {r["predicted_late"] for r in rows} == {"0", "1"}


def test_score_file_matches_model_probabilities(cfg, models, tmp_data_dir):
    publish_ckpt(models, 1)
    loaded = load_checkpoint(newest_checkpoint(cfg))
    features = write_features(tmp_data_dir, 1, n=10)
    _, inputs = read(features)
    X = np.array([[float(r["distance_km"]), float(r["prep_minutes"])] for r in inputs])
    with np.errstate(all="ignore"):
        expected = loaded.model.predict_proba(X)[:, 1]
    _, rows = read(score_file(features, loaded, cfg).predictions_path)
    assert [r["late_probability"] for r in rows] == [f"{p:.3f}" for p in expected]


def test_scores_with_whatever_columns_the_checkpoint_lists(cfg, models, tmp_data_dir):
    columns = ("distance_km", "prep_minutes", "is_peak")
    publish_ckpt(models, 1, columns=columns)
    loaded = load_checkpoint(newest_checkpoint(cfg))
    assert loaded.feature_columns == columns
    _, rows = read(score_file(write_features(tmp_data_dir, 1, n=8), loaded, cfg).predictions_path)
    assert len(rows) == 8


def test_header_only_input_gives_header_only_output(cfg, models, tmp_data_dir):
    publish_ckpt(models, 1)
    loaded = load_checkpoint(newest_checkpoint(cfg))
    result = score_file(write_features(tmp_data_dir, 1, rows=[]), loaded, cfg)
    assert read(result.predictions_path) == (PREDICTION_COLUMNS, [])
    assert pending_files(cfg) == []  # never retried


@pytest.mark.parametrize("problem", ["missing_column", "non_numeric", "nan"])
def test_unusable_features_file_is_skipped_once(cfg, models, tmp_data_dir, capsys,
                                                monkeypatch, problem):
    publish_ckpt(models, 1)
    if problem == "missing_column":
        bad = write_features(tmp_data_dir, 1,
                             header=tuple(c for c in FEATURE_HEADER if c != "prep_minutes"))
        reason = "missing column prep_minutes"
    else:
        value = "far" if problem == "non_numeric" else "nan"
        bad = write_features(tmp_data_dir, 1, rows=[{**{c: "1" for c in FEATURE_HEADER},
                                                     "order_id": "ORD-1", "distance_km": value}])
        reason = "non-numeric distance_km on line 2"
    good = write_features(tmp_data_dir, 2)

    _polls(monkeypatch, 3)
    assert infer.run(cfg, poll_seconds=0.01)[0] == 1
    assert capsys.readouterr().out.count(f"skipped {bad.name}: {reason}") == 1
    assert not infer.predictions_path_for(bad, cfg).exists()
    assert infer.predictions_path_for(good, cfg).exists()


def test_failed_write_leaves_no_tmp(cfg, models, tmp_data_dir, monkeypatch):
    publish_ckpt(models, 1)
    loaded = load_checkpoint(newest_checkpoint(cfg))
    features = write_features(tmp_data_dir, 1)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(infer.os, "replace", fail)
    with pytest.raises(OSError):
        score_file(features, loaded, cfg)
    assert not list(tmp_data_dir.rglob("*.tmp"))
    assert pending_files(cfg) == [features]


# ---------- pending_files ----------

def test_pending_files_lists_unscored_sorted_and_ignores_tmp(cfg, models, tmp_data_dir):
    later = write_features(tmp_data_dir, 2)
    earlier = write_features(tmp_data_dir, 1)
    (tmp_data_dir / "features" / ".features_20260928T000003Z_0003.csv.tmp").write_text("x")
    assert pending_files(cfg) == [earlier, later]
    publish_ckpt(models, 1)
    score_file(earlier, load_checkpoint(newest_checkpoint(cfg)), cfg)
    assert pending_files(cfg) == [later]


def test_scoring_twice_is_a_no_op(cfg, models, tmp_data_dir):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)
    assert infer.run(cfg, poll_seconds=0.01, once=True)[0] == 1
    out = tmp_data_dir / "predictions" / "predictions_20260928T000001Z_0001.csv"
    mtime = out.stat().st_mtime_ns
    assert infer.run(cfg, poll_seconds=0.01, once=True)[0] == 0
    assert out.stat().st_mtime_ns == mtime


# ---------- hot swap ----------

def test_hot_swap_to_newer_checkpoint(cfg, models, tmp_data_dir, capsys, monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)
    first = tmp_data_dir / "predictions" / "predictions_20260928T000001Z_0001.csv"
    snapshot = {}

    def between(i):
        if i == 1:
            snapshot["v1"] = first.read_bytes()
            publish_ckpt(models, 2, rows_total=250, model=fitted_model(seed=9))
            write_features(tmp_data_dir, 2)

    _polls(monkeypatch, 2, between)
    assert infer.run(cfg, poll_seconds=0.01) == (2, 10, "model_v0002")
    _, rows = read(tmp_data_dir / "predictions" / "predictions_20260928T000002Z_0002.csv")
    assert {r["checkpoint_id"] for r in rows} == {"model_v0002"}
    assert first.read_bytes() == snapshot["v1"]  # never re-scored
    assert ("loaded model_v0002 (accuracy 0.80, trained on 250 rows) — replacing model_v0001"
            in capsys.readouterr().out)


# ---------- waiting & loop ----------

def test_once_with_no_models_waits_and_writes_nothing(cfg, tmp_data_dir, capsys):
    write_features(tmp_data_dir, 1)
    assert infer.run(cfg, poll_seconds=0.01, once=True) == (0, 0, None)
    out = capsys.readouterr().out
    assert "waiting for a checkpoint: no model_v*.json in" in out
    assert "(run make train)" in out
    assert not list((tmp_data_dir / "predictions").iterdir())


def test_waiting_line_appears_once_across_polls(cfg, tmp_data_dir, capsys, monkeypatch):
    _polls(monkeypatch, 5)
    infer.run(cfg, poll_seconds=0.01)
    assert capsys.readouterr().out.count("waiting for a checkpoint") == 1


def test_first_checkpoint_is_found_while_waiting(cfg, models, tmp_data_dir, capsys, monkeypatch):
    write_features(tmp_data_dir, 1)
    _polls(monkeypatch, 3, lambda i: publish_ckpt(models, 1) if i == 1 else None)
    assert infer.run(cfg, poll_seconds=0.01) == (1, 5, "model_v0001")
    out = capsys.readouterr().out
    assert out.index("waiting for a checkpoint") < out.index("found model_v0001")


def test_unexpected_error_is_logged_once_and_loop_survives(cfg, models, tmp_data_dir, capsys,
                                                          monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)
    real = infer.pending_files
    calls = []

    def flaky(cfg):
        calls.append(1)
        if len(calls) <= 2:
            raise RuntimeError("disk hiccup")
        return real(cfg)

    monkeypatch.setattr(infer, "pending_files", flaky)
    _polls(monkeypatch, 4)
    assert infer.run(cfg, poll_seconds=0.01)[0] == 1
    assert capsys.readouterr().out.count("inference failed: RuntimeError: disk hiccup") == 1


def test_ctrl_c_stops_cleanly(cfg, models, tmp_data_dir, capsys, monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)
    _polls(monkeypatch, 1)
    assert infer.run(cfg, poll_seconds=5) == (1, 5, "model_v0001")
    assert ("[infer] stopped after scoring 1 batches (5 orders), last checkpoint model_v0001"
            in capsys.readouterr().out)
    assert not list(tmp_data_dir.rglob("*.tmp"))


# ---------- review fixes ----------

def fixed_model(coef, intercept=-7.0):
    """A LogisticRegression with chosen weights, so two 'different models' are truly different."""
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([coef], dtype=float)
    model.intercept_ = np.array([intercept])
    model.n_features_in_ = len(coef)
    return model


def test_reset_models_dir_while_running_reloads_from_disk(cfg, models, tmp_data_dir, capsys,
                                                          monkeypatch):
    # Regression: after data/models was reset and train started again at v0001,
    # infer kept its old in-memory v0002 and labelled new rows "model_v0002",
    # a name that by then belonged to a different model on disk.
    publish_ckpt(models, 1, model=fixed_model([0.6, 0.25]))
    publish_ckpt(models, 2, model=fixed_model([0.6, 0.25]))
    write_features(tmp_data_dir, 1)

    def between(i):
        if i == 1:
            for p in models.iterdir():
                p.unlink()
            publish_ckpt(models, 1, model=fixed_model([-0.6, -0.25]), rows_total=7)
            write_features(tmp_data_dir, 2)

    _polls(monkeypatch, 2, between)
    assert infer.run(cfg, poll_seconds=0.01)[2] == "model_v0001"
    out = capsys.readouterr().out
    assert "model_v0002 is no longer the file on disk" in out
    _, rows = read(tmp_data_dir / "predictions" / "predictions_20260928T000002Z_0002.csv")
    expected = load_checkpoint(newest_checkpoint(cfg))
    assert {r["checkpoint_id"] for r in rows} == {"model_v0001"}
    assert all(float(r["late_probability"]) < 0.01 for r in rows)  # the new model's answer
    assert expected.checkpoint.sidecar["rows_total"] == 7


def test_models_removed_while_running_waits_instead_of_scoring(cfg, models, tmp_data_dir,
                                                               capsys, monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)

    def between(i):
        if i == 1:
            for p in models.iterdir():
                p.unlink()
            write_features(tmp_data_dir, 2)

    _polls(monkeypatch, 3, between)
    assert infer.run(cfg, poll_seconds=0.01)[2] is None
    out = capsys.readouterr().out
    assert out.count("waiting for a checkpoint") == 1
    assert not (tmp_data_dir / "predictions" / "predictions_20260928T000002Z_0002.csv").exists()


def test_model_that_cannot_score_its_columns_is_not_loadable(cfg, models):
    publish_ckpt(models, 1, model=fixed_model([0.6, 0.25, 0.1]))  # expects 3 inputs, lists 2
    with pytest.raises(infer.CannotLoad, match="can't score 2 feature columns"):
        load_checkpoint(newest_checkpoint(cfg))


def test_unscorable_newer_model_does_not_stall_scoring(cfg, models, tmp_data_dir, capsys,
                                                       monkeypatch):
    # Regression: it used to be swapped in, then every file failed on every poll.
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)

    def between(i):
        if i == 1:
            publish_ckpt(models, 2, model=fixed_model([0.6, 0.25, 0.1]))
            write_features(tmp_data_dir, 2)
            write_features(tmp_data_dir, 3)

    _polls(monkeypatch, 3, between)
    assert infer.run(cfg, poll_seconds=0.01) == (3, 15, "model_v0001")
    out = capsys.readouterr().out
    assert out.count("cannot load model_v0002") == 1
    assert "inference failed" not in out


def test_damaged_newer_sidecar_is_logged_once(cfg, models, tmp_data_dir, capsys, monkeypatch):
    publish_ckpt(models, 1)
    write_features(tmp_data_dir, 1)

    def between(i):
        if i == 1:
            (models / "model_v0002.json").write_text("{damaged")

    _polls(monkeypatch, 4, between)
    assert infer.run(cfg, poll_seconds=0.01)[2] == "model_v0001"
    assert capsys.readouterr().out.count("cannot load model_v0002: unreadable sidecar") == 1


def test_only_damaged_sidecars_waits_with_accurate_message(cfg, models, tmp_data_dir, capsys):
    models.mkdir(parents=True)
    (models / "model_v0001.json").write_text("{damaged")
    write_features(tmp_data_dir, 1)
    assert infer.run(cfg, poll_seconds=0.01, once=True) == (0, 0, None)
    out = capsys.readouterr().out
    assert "cannot load model_v0001: unreadable sidecar" in out
    assert "none of the published models could be loaded" in out


class _NanModel:
    """Runs, but gives no usable probability."""

    def predict_proba(self, X):
        return np.full((len(X), 2), np.nan)


class _OneColumnModel:
    """Runs, but returns one column, so there's no 'late' probability to read."""

    def predict_proba(self, X):
        return np.ones((len(X), 1))


@pytest.mark.parametrize("bad_model", [_NanModel(), _OneColumnModel()], ids=["nan", "one-column"])
def test_model_without_usable_probabilities_is_not_loadable(cfg, models, bad_model):
    publish_ckpt(models, 1, model=bad_model)
    with pytest.raises(infer.CannotLoad, match="no usable late probability"):
        load_checkpoint(newest_checkpoint(cfg))
