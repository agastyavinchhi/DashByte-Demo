"""The predictions file format, pinned field by field.

A predictions file has exactly four columns, and no timestamp column on purpose
(see the Stage 5 plan). Where each prediction came from is still fully traceable:

- **Model version:** ``checkpoint_id`` is ``model_v<NNNN>``, the stem of the
  ``data/models`` checkpoint that scored the row. Its sidecar records the
  version number and ``trained_at``.
- **Timestamp:** the file is ``predictions_<YYYYmmddTHHMMSSZ>_<seq>.csv``. That UTC
  stamp is the batch time, shared with ``orders_…`` and ``features_…``. Every
  order in the batch is from the 15 simulated minutes up to that stamp.

Any change to the format, such as column names, order or count, number formatting,
quoting, line endings, the id formats, or the threshold rule, should fail here. If
the change is deliberate, update this file and the Stage 5 plan together.

The checkpoint is published by the real ``train`` stage, not hand-built, so its
on-disk format is the one infer meets in production. (Tests may import stages;
stages may not import each other.)
"""
from __future__ import annotations

import csv
import json
import random
import re
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pytest

from pipeline import infer, preprocess, simulator, train
from pipeline.config import load_config
from pipeline.paths import ensure_data_dirs, raw_dir

pytestmark = pytest.mark.regression

HEADER_LINE = "order_id,late_probability,predicted_late,checkpoint_id"
LINE_ENDING = "\r\n"  # csv module default; part of the byte-level format
# One data row, exactly: no quotes, no spaces, four fields.
ROW_PATTERN = re.compile(r"(ORD-[0-9a-f]{10}),(0\.\d{3}|1\.000),([01]),(model_v\d{4})")
FILE_PATTERN = re.compile(r"predictions_(\d{8}T\d{6}Z)_(\d{4})\.csv")
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
START = datetime(2026, 9, 28, 17, 0, 0, tzinfo=timezone.utc)
BATCHES = 5


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Simulated raw batches → real preprocess → real train publish → real infer pass."""
    data = tmp_path_factory.mktemp("prediction_format") / "data"
    cfg = load_config({"DASHBITE_DATA_DIR": str(data), "TRAIN_EVERY_N_EVENTS": "50"})
    ensure_data_dirs(cfg)
    for seq in range(1, BATCHES + 1):
        now = START + timedelta(minutes=15 * seq)
        rows = simulator.generate_batch(random.Random(2026 + seq), 20, now, messy_rate=0.05,
                                        span_seconds=900)
        simulator.write_batch(rows, raw_dir(cfg), now, seq)
    assert preprocess.run(cfg, poll_seconds=0, once=True)[0] == BATCHES
    assert train.run(cfg, poll_seconds=0, once=True) == 1
    assert infer.run(cfg, poll_seconds=0, once=True)[0] == BATCHES
    return data


def _prediction_files(data):
    files = sorted((data / "predictions").glob("predictions_*.csv"))
    assert len(files) == BATCHES
    return files


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_columns_are_exactly_these_four():
    assert infer.PREDICTION_COLUMNS == (
        "order_id", "late_probability", "predicted_late", "checkpoint_id",
    )


def test_every_line_has_the_exact_byte_format(data):
    for path in _prediction_files(data):
        text = path.read_bytes().decode("utf-8")  # strict: must be UTF-8
        assert text.endswith(LINE_ENDING), path.name
        header, *lines = text[: -len(LINE_ENDING)].split(LINE_ENDING)
        assert header == HEADER_LINE
        assert lines, f"{path.name} has no rows"
        for line in lines:
            assert ROW_PATTERN.fullmatch(line), f"{path.name}: {line!r}"


def test_field_types_and_meaning(data):
    for path in _prediction_files(data):
        for row in _rows(path):
            assert list(row) == list(infer.PREDICTION_COLUMNS)
            assert isinstance(row["order_id"], str) and row["order_id"].startswith("ORD-")

            probability = float(row["late_probability"])  # a float in [0, 1], 3 decimals
            assert 0.0 <= probability <= 1.0
            assert row["late_probability"] == f"{probability:.3f}"

            predicted = int(row["predicted_late"])  # an int, 0 or 1
            assert predicted in (0, 1) and row["predicted_late"] == str(predicted)
            assert predicted == int(probability >= infer.PREDICT_THRESHOLD == 0.5)


def test_checkpoint_id_names_the_model_version_that_scored_the_row(data):
    models = data / "models"
    for path in _prediction_files(data):
        rows = _rows(path)
        (checkpoint_id,) = {r["checkpoint_id"] for r in rows}  # one checkpoint per file
        version = int(checkpoint_id[len("model_v"):])
        assert checkpoint_id == f"model_v{version:04d}" == "model_v0001"

        # The id points at a real published checkpoint, whose sidecar says the same.
        sidecar = json.loads((models / f"{checkpoint_id}.json").read_text())
        assert sidecar["version"] == version
        assert sidecar["model_file"] == f"{checkpoint_id}.joblib"
        datetime.strptime(sidecar["trained_at"], ISO_FORMAT)  # when that model was trained

        # ...and that checkpoint really produced these probabilities.
        payload = joblib.load(models / sidecar["model_file"])
        features = {r["order_id"]: r for r in _rows(data / "features" / path.name.replace(
            "predictions_", "features_", 1))}
        X = np.array([[float(features[r["order_id"]][c]) for c in sidecar["feature_columns"]]
                      for r in rows])
        with np.errstate(all="ignore"):
            expected = payload["model"].predict_proba(X)[:, 1]
        assert [r["late_probability"] for r in rows] == [f"{p:.3f}" for p in expected]


def test_timestamp_is_traceable_through_the_file_name(data):
    for path in _prediction_files(data):
        match = FILE_PATTERN.fullmatch(path.name)
        assert match, path.name
        stamp, seq = match.groups()
        batch_time = datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=timezone.utc)
        assert batch_time == START + timedelta(minutes=15 * int(seq))

        # The same <stamp>_<seq> names the raw batch and the features it was scored from.
        assert (data / "raw" / f"orders_{stamp}_{seq}.csv").is_file()
        features = _rows(data / "features" / f"features_{stamp}_{seq}.csv")

        # Same orders, same order, and the stamp is the batch's newest order time.
        assert [r["order_id"] for r in _rows(path)] == [r["order_id"] for r in features]
        order_times = [datetime.strptime(r["timestamp"], ISO_FORMAT).replace(tzinfo=timezone.utc)
                       for r in features]
        assert max(order_times) <= batch_time
        assert batch_time - min(order_times) <= timedelta(minutes=15)


@pytest.mark.parametrize("probability,written,predicted", [
    (0.5, "0.500", "1"),       # exactly at the threshold counts as late
    (0.4996, "0.500", "1"),    # decided on the written value, so the file agrees with itself
    (0.4994, "0.499", "0"),
    (0.0001, "0.000", "0"),
    (0.9999, "1.000", "1"),
])
def test_threshold_and_rounding_at_the_boundary(tmp_path, probability, written, predicted):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()  # fixed weights: every row gets exactly `probability`
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((1, 2))
    model.intercept_ = np.array([np.log(probability / (1 - probability))])
    model.n_features_in_ = 2

    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path / "data")})
    ensure_data_dirs(cfg)
    features = tmp_path / "data" / "features" / "features_20260928T140203Z_0001.csv"
    features.write_text(
        "order_id,timestamp,hour,is_peak,distance_km,prep_minutes,order_value,was_late\r\n"
        "ORD-0123456789,2026-09-28T14:02:03Z,14,0,6.4,18,42.50,1\r\n"
    )
    sidecar_path = tmp_path / "data" / "models" / "model_v0007.json"
    checkpoint = infer.Checkpoint(7, sidecar_path, {"rows_total": 1})
    loaded = infer.Loaded(checkpoint, model, ("distance_km", "prep_minutes"))

    out = infer.score_file(features, loaded, cfg).predictions_path
    assert out.read_bytes().decode() == (
        f"{HEADER_LINE}{LINE_ENDING}ORD-0123456789,{written},{predicted},model_v0007{LINE_ENDING}"
    )
