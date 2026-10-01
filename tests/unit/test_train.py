from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta

import joblib
import pytest

from pipeline import train
from pipeline.config import load_config
from pipeline.train import (
    SIDECAR_KEYS,
    NotReady,
    latest_published,
    load_rows,
    publish,
    rows_available,
    train_model,
)

pytestmark = pytest.mark.unit

HEADER = ("order_id", "timestamp", "hour", "is_peak", "distance_km", "prep_minutes",
          "order_value", "was_late")
START = datetime(2026, 9, 28, 10, 0, 0)


def make_rows(n, seed=1, start_index=0, labels=None):
    """Feature rows with a real signal: late when prep + 3*distance (+ noise) > 30."""
    rng = random.Random(seed)
    rows = []
    for i in range(start_index, start_index + n):
        distance = round(rng.uniform(0.5, 12), 1)
        prep = rng.randint(5, 30)
        late = int(prep + 3 * distance + rng.gauss(0, 4) > 30) if labels is None else labels
        ts = START + timedelta(minutes=i)
        rows.append({
            "order_id": f"ORD-{i:010d}", "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hour": str(ts.hour), "is_peak": "0", "distance_km": f"{distance:.1f}",
            "prep_minutes": str(prep), "order_value": "20.00", "was_late": str(late),
        })
    return rows


def write_features(data_dir, seq, rows, header=HEADER):
    folder = data_dir / "features"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"features_20260928T{seq:06d}Z_{seq:04d}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def as_tuples(rows):
    return [(r["distance_km"], r["prep_minutes"], r["was_late"], r["timestamp"]) for r in rows]


@pytest.fixture
def cfg(tmp_data_dir, monkeypatch):
    monkeypatch.setenv("TRAIN_EVERY_N_EVENTS", "100")
    return load_config()


# ---------- counting ----------

def test_rows_available_counts_across_files_and_ignores_others(cfg, tmp_data_dir):
    write_features(tmp_data_dir, 1, make_rows(30))
    write_features(tmp_data_dir, 2, make_rows(12, start_index=30))
    feats = tmp_data_dir / "features"
    (feats / ".features_20260928T000003Z_0003.csv.tmp").write_text("half")
    (feats / "notes.csv").write_text("order_id\nx\n")
    assert rows_available(cfg) == 42


def test_file_missing_columns_is_skipped_and_logged_once(cfg, tmp_data_dir, capsys):
    write_features(tmp_data_dir, 1, make_rows(10))
    bad_header = tuple(c for c in HEADER if c != "prep_minutes")
    write_features(tmp_data_dir, 2, make_rows(10), header=bad_header)
    cache = {}
    assert rows_available(cfg, cache) == 10
    assert rows_available(cfg, cache) == 10
    assert capsys.readouterr().out.count("skipped features_20260928T000002Z_0002.csv: "
                                         "missing columns") == 1


def test_load_rows_is_oldest_first(cfg, tmp_data_dir):
    write_features(tmp_data_dir, 2, make_rows(3, start_index=3))
    write_features(tmp_data_dir, 1, make_rows(3))
    stamps = [r[3] for r in load_rows(cfg)]
    assert stamps == sorted(stamps)


# ---------- latest_published ----------

def test_latest_published_with_no_models(cfg):
    assert latest_published(cfg) == (0, 0)


def test_latest_published_ignores_joblib_without_sidecar(cfg, tmp_data_dir):
    model, metrics = train_model(as_tuples(make_rows(200)))
    assert publish(cfg, model, metrics) == 1
    (tmp_data_dir / "models" / "model_v0002.joblib").write_bytes(b"half-published")
    assert latest_published(cfg) == (1, 200)
    # ...but the next version still steps past it rather than colliding.
    assert publish(cfg, model, metrics) == 3


# ---------- trigger ----------

def test_trigger_fires_at_exactly_n_not_n_minus_1(cfg, tmp_data_dir, capsys):
    write_features(tmp_data_dir, 1, make_rows(99))
    assert train.run(cfg, poll_seconds=0.01, once=True) == 0
    assert "waiting: 99/100 new rows" in capsys.readouterr().out

    write_features(tmp_data_dir, 2, make_rows(1, start_index=99))
    assert train.run(cfg, poll_seconds=0.01, once=True) == 1
    out = capsys.readouterr().out
    assert "100 new rows (threshold 100) → training on 100 rows" in out
    assert "published model_v0001" in out

    # rows_total moved forward, so the count starts again from zero.
    assert train.run(cfg, poll_seconds=0.01, once=True) == 0
    assert "waiting: 0/100 new rows since v0001" in capsys.readouterr().out


# ---------- train_model ----------

def test_holdout_is_the_newest_20_percent_without_shuffling(monkeypatch):
    rows = as_tuples(make_rows(200))
    fitted = {}
    real_fit = train.LogisticRegression.fit

    def spy_fit(self, X, y):
        fitted["X"] = X.tolist()
        return real_fit(self, X, y)

    monkeypatch.setattr(train.LogisticRegression, "fit", spy_fit)
    _, metrics = train_model(rows)
    assert (metrics["n_train"], metrics["n_test"]) == (160, 40)
    assert metrics["n_train"] + metrics["n_test"] == metrics["rows_total"] == 200
    assert fitted["X"] == [[float(r[0]), float(r[1])] for r in rows[:160]]
    assert metrics["data_through"] == rows[-1][3]


def test_same_rows_give_identical_model_and_metrics():
    rows = as_tuples(make_rows(300, seed=5))
    (m1, a), (m2, b) = train_model(rows), train_model(rows)
    assert a == b
    assert m1.coef_.tolist() == m2.coef_.tolist()


def test_metrics_are_rounded_and_complete():
    _, metrics = train_model(as_tuples(make_rows(300)))
    for key in ("late_rate_test", "baseline_accuracy", "accuracy", "precision", "recall",
                "f1", "roc_auc", "intercept"):
        assert metrics[key] == round(metrics[key], 3)
    assert set(metrics["coefficients"]) == {"distance_km", "prep_minutes"}


@pytest.mark.parametrize("rows,reason", [
    (as_tuples(make_rows(49)), "need at least 50 rows"),
    (as_tuples(make_rows(200, labels=0)), "need both on-time and late orders"),
    (as_tuples(make_rows(200, labels=1)), "need both on-time and late orders"),
])
def test_guards(rows, reason):
    with pytest.raises(NotReady, match=reason):
        train_model(rows)


def test_guard_publishes_nothing_and_logs_waiting(cfg, tmp_data_dir, capsys):
    write_features(tmp_data_dir, 1, make_rows(150, labels=0))
    assert train.run(cfg, poll_seconds=0.01, once=True) == 0
    assert "waiting: need both on-time and late orders" in capsys.readouterr().out
    assert not list((tmp_data_dir / "models").iterdir())


# ---------- publish ----------

def test_publish_writes_loadable_versions_in_sequence(cfg, tmp_data_dir):
    model, metrics = train_model(as_tuples(make_rows(200)))
    assert publish(cfg, model, metrics) == 1
    assert publish(cfg, model, metrics) == 2
    names = sorted(p.name for p in (tmp_data_dir / "models").iterdir())
    assert names == ["model_v0001.joblib", "model_v0001.json",
                     "model_v0002.joblib", "model_v0002.json"]

    payload = joblib.load(tmp_data_dir / "models" / "model_v0002.joblib")
    assert payload["version"] == 2
    assert payload["feature_columns"] == ["distance_km", "prep_minutes"]
    proba = payload["model"].predict_proba([[12.0, 25]])[0]
    assert 0.5 < proba[1] <= 1  # a long, slow order is probably late

    sidecar = json.loads((tmp_data_dir / "models" / "model_v0002.json").read_text())
    assert tuple(sidecar) == SIDECAR_KEYS
    assert sidecar["model_file"] == "model_v0002.joblib"
    assert sidecar["rows_new"] == 0  # same rows as v0001
    assert sidecar["train_every_n_events"] == 100


def test_publish_refuses_to_overwrite(cfg, tmp_data_dir, monkeypatch):
    model, metrics = train_model(as_tuples(make_rows(200)))
    publish(cfg, model, metrics)
    before = (tmp_data_dir / "models" / "model_v0001.json").read_bytes()
    monkeypatch.setattr(train, "next_version", lambda cfg: 1)
    with pytest.raises(FileExistsError):
        publish(cfg, model, metrics)
    assert (tmp_data_dir / "models" / "model_v0001.json").read_bytes() == before


def test_failed_sidecar_write_leaves_no_tmp_and_is_not_published(cfg, tmp_data_dir, monkeypatch):
    model, metrics = train_model(as_tuples(make_rows(200)))
    real_replace = train.os.replace

    def fail_on_sidecar(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(train.os, "replace", fail_on_sidecar)
    with pytest.raises(OSError):
        publish(cfg, model, metrics)
    assert not list((tmp_data_dir / "models").glob(".*.tmp"))
    assert latest_published(cfg) == (0, 0)  # .joblib alone isn't published


# ---------- run loop ----------

def test_training_error_is_logged_and_loop_survives(cfg, tmp_data_dir, capsys, monkeypatch):
    write_features(tmp_data_dir, 1, make_rows(150))
    attempts = []

    def boom(rows):
        attempts.append(1)
        raise RuntimeError("bad luck")

    polls = []

    def fake_sleep(_seconds):
        polls.append(1)
        if len(polls) == 1:  # new data arrives, so the next poll retries
            write_features(tmp_data_dir, 2, make_rows(5, start_index=150))
        if len(polls) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(train, "train_model", boom)
    monkeypatch.setattr(train.time, "sleep", fake_sleep)
    assert train.run(cfg, poll_seconds=0.01) == 0
    out = capsys.readouterr().out
    assert out.count("training failed: RuntimeError: bad luck") == 2
    assert len(attempts) == 2  # retried once data changed; not on the idle poll
    assert "stopped after publishing 0 models (no models yet)" in out


def test_progress_is_logged_only_when_the_count_changes(cfg, tmp_data_dir, capsys, monkeypatch):
    write_features(tmp_data_dir, 1, make_rows(10))
    polls = []

    def fake_sleep(_seconds):
        polls.append(1)
        if len(polls) == 2:
            write_features(tmp_data_dir, 2, make_rows(10, start_index=10))
        if len(polls) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(train.time, "sleep", fake_sleep)
    train.run(cfg, poll_seconds=0.01)
    lines = [l for l in capsys.readouterr().out.splitlines() if "waiting:" in l]
    assert [l.split("] ")[1] for l in lines] == [
        "waiting: 10/100 new rows", "waiting: 20/100 new rows",
    ]


def test_ctrl_c_stops_cleanly(cfg, tmp_data_dir, capsys, monkeypatch):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(train.time, "sleep", interrupt)
    assert train.run(cfg, poll_seconds=5) == 0
    assert "[train] stopped after publishing 0 models (no models yet)" in capsys.readouterr().out
    assert not list(tmp_data_dir.rglob("*.tmp"))


def test_training_emits_no_runtime_warnings():
    # Regression: NumPy 2.0 + macOS Accelerate printed spurious matmul warnings on
    # every retrain, flooding the room's terminal and logs/train.log.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        train_model(as_tuples(make_rows(2000)))


def test_non_finite_model_is_refused(monkeypatch):
    import numpy as np

    real_fit = train.LogisticRegression.fit

    def broken_fit(self, X, y):
        real_fit(self, X, y)
        self.coef_ = np.array([[np.nan, 0.1]])
        return self

    monkeypatch.setattr(train.LogisticRegression, "fit", broken_fit)
    with pytest.raises(ValueError, match="non-finite"):
        train_model(as_tuples(make_rows(200)))


def test_sidecar_keeps_short_lists_on_one_line(cfg, tmp_data_dir):
    model, metrics = train_model(as_tuples(make_rows(200)))
    publish(cfg, model, metrics)
    text = (tmp_data_dir / "models" / "model_v0001.json").read_text()
    lines = text.splitlines()

    # Same layout as the plan's example: one key per line, lists and dicts inline.
    assert lines[0] == "{" and lines[-1] == "}"
    assert len(lines) == len(SIDECAR_KEYS) + 2
    assert '  "feature_columns": ["distance_km", "prep_minutes"],' in lines
    assert any(line.startswith('  "coefficients": {"distance_km": ') for line in lines)
    # Still ordinary JSON, read back exactly.
    sidecar = json.loads(text)
    assert tuple(sidecar) == SIDECAR_KEYS
    assert text == train.format_sidecar(sidecar)


@pytest.mark.parametrize("count,expected", [
    (1, "stopped after publishing 1 model (latest v0001)"),
    (2, "stopped after publishing 2 models (latest v0002)"),
])
def test_stop_message_singular_and_plural(cfg, tmp_data_dir, capsys, monkeypatch, count, expected):
    model, metrics = train_model(as_tuples(make_rows(200)))
    calls = []

    def fake_sleep(_seconds):
        calls.append(1)
        if len(calls) == count:
            raise KeyboardInterrupt
        write_features(tmp_data_dir, 10 + len(calls), make_rows(100, start_index=1000 * len(calls)))

    write_features(tmp_data_dir, 1, make_rows(100))
    monkeypatch.setattr(train.time, "sleep", fake_sleep)
    assert train.run(cfg, poll_seconds=0.01) == count
    assert expected in capsys.readouterr().out


# ---------- review: one bad file or sidecar must never block every future model ----------

def _write_bytes(data_dir, seq, content):
    folder = data_dir / "features"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"features_20260928T{seq:06d}Z_{seq:04d}.csv"
    path.write_bytes(content)
    return path


_HEADER_BYTES = ",".join(HEADER).encode()
BAD_FEATURE_FILES = {
    # Regression: each used to raise out of load_rows on every poll, so no model
    # was ever published again (and the non-UTF-8 case logged every poll).
    "not-utf8": (_HEADER_BYTES + b"\nORD-\xe9,2026-09-28T14:00:00Z,14,0,1.0,10,20.00,0\n",
                 "not UTF-8 text"),
    "nul-byte": (_HEADER_BYTES + b"\nORD-1,2026-09-28T14:00:00Z,14,0,1.0,10\x00,20.00,0\n",
                 "malformed CSV"),
    "text-distance": (_HEADER_BYTES + b"\nORD-1,2026-09-28T14:00:00Z,14,0,abc,10,20.00,0\n",
                      "unusable value on line 2"),
    "nan-prep": (_HEADER_BYTES + b"\nORD-1,2026-09-28T14:00:00Z,14,0,1.0,nan,20.00,0\n",
                 "unusable value on line 2"),
    "text-label": (_HEADER_BYTES + b"\nORD-1,2026-09-28T14:00:00Z,14,0,1.0,10,20.00,yes\n",
                   "unusable value on line 2"),
    "short-row": (_HEADER_BYTES + b"\nORD-1,2026-09-28T14:00:00Z,14,0,1.0\n",
                  "unusable value on line 2"),
}


@pytest.mark.parametrize("content,reason", BAD_FEATURE_FILES.values(), ids=list(BAD_FEATURE_FILES))
def test_bad_feature_file_is_skipped_once_and_the_rest_still_trains(
    cfg, tmp_data_dir, capsys, monkeypatch, content, reason
):
    bad = _write_bytes(tmp_data_dir, 1, content)
    write_features(tmp_data_dir, 2, make_rows(150))
    polls = []

    def fake_sleep(_seconds):
        polls.append(1)
        if len(polls) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(train.time, "sleep", fake_sleep)
    assert train.run(cfg, poll_seconds=0.01) == 1
    out = capsys.readouterr().out
    assert out.count(f"skipped {bad.name}: {reason}") == 1
    assert "training failed" not in out
    assert latest_published(cfg) == (1, 150)


def test_corrupt_sidecar_is_ignored_and_next_version_published(cfg, tmp_data_dir, capsys):
    write_features(tmp_data_dir, 1, make_rows(150))
    assert train.run(cfg, poll_seconds=0.01, once=True) == 1
    sidecar = tmp_data_dir / "models" / "model_v0001.json"
    sidecar.write_text('{"version": 1, "rows_to')  # damaged, e.g. by a hand edit
    damaged = sidecar.read_bytes()
    capsys.readouterr()

    assert latest_published(cfg) == (0, 0)
    # Regression: this used to fail every poll, and crash the final summary (exit 1).
    assert train.run(cfg, poll_seconds=0.01, once=True) == 1
    out = capsys.readouterr().out
    assert "ignoring model_v0001.json: unreadable sidecar" in out
    assert "stopped after publishing 1 model (latest v0002)" in out
    assert latest_published(cfg) == (2, 150)
    assert sidecar.read_bytes() == damaged  # never rewritten or deleted


@pytest.mark.parametrize("content", ["", "[]", '{"version": 1}', '{"rows_total": "many"}'],
                         ids=["empty", "not-object", "no-rows_total", "bad-rows_total"])
def test_unusable_sidecar_does_not_count_as_published(cfg, tmp_data_dir, content):
    models = tmp_data_dir / "models"
    models.mkdir(parents=True)
    (models / "model_v0001.json").write_text(content)
    assert latest_published(cfg) == (0, 0)
    assert train.next_version(cfg) == 2  # its number is still never reused


def test_repeated_identical_failure_is_logged_once(cfg, tmp_data_dir, capsys, monkeypatch):
    write_features(tmp_data_dir, 1, make_rows(150))

    def broken(_cfg, _cache=None):
        raise PermissionError("no access")

    polls = []

    def fake_sleep(_seconds):
        polls.append(1)
        if len(polls) == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr(train, "load_rows", broken)
    monkeypatch.setattr(train.time, "sleep", fake_sleep)
    train.run(cfg, poll_seconds=0.01)
    assert capsys.readouterr().out.count("training failed: PermissionError: no access") == 1


# ---------- review: the model uses exactly distance_km and prep_minutes ----------

def test_model_ignores_every_column_but_its_two_features(cfg, tmp_data_dir):
    rows = make_rows(300)
    scrambled = [
        {**r, "hour": str(i % 24), "is_peak": str(i % 2), "order_value": f"{i * 3.7:.2f}",
         "order_id": f"OTHER-{i}"}
        for i, r in enumerate(rows)
    ]
    plain_model, plain = train_model(as_tuples(rows))
    other_model, other = train_model(as_tuples(scrambled))
    assert plain == other
    assert plain_model.n_features_in_ == other_model.n_features_in_ == 2


def test_published_model_takes_exactly_the_recorded_features(cfg, tmp_data_dir):
    write_features(tmp_data_dir, 1, make_rows(150))
    train.run(cfg, poll_seconds=0.01, once=True)
    payload = joblib.load(tmp_data_dir / "models" / "model_v0001.joblib")
    sidecar = json.loads((tmp_data_dir / "models" / "model_v0001.json").read_text())
    assert payload["feature_columns"] == sidecar["feature_columns"] == [
        "distance_km", "prep_minutes",
    ]
    assert type(payload["model"]).__name__ == sidecar["model_type"] == "LogisticRegression"
    assert payload["model"].n_features_in_ == 2
    assert list(sidecar["coefficients"]) == sidecar["feature_columns"]
