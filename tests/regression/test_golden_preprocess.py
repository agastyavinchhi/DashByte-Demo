"""Golden preprocess: the pinned raw batch must keep producing exactly these outputs.

Input is the Stage 2 golden raw batch (seed 2027, 20 rows, all six defects).
Expected output is two fixtures, compared byte for byte:

- tests/fixtures/golden_features_seed2027.csv
- tests/fixtures/golden_rejects_seed2027.csv

If a change to preprocess is deliberate, regenerate and review the diff:

    .venv/bin/python -m tests.regression.test_golden_preprocess
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.preprocess import process_file
from pipeline.simulator import MESSY_DEFECTS, find_defect
from tests.regression.test_golden_batch import GOLDEN as GOLDEN_RAW

pytestmark = pytest.mark.regression

FIXTURES = GOLDEN_RAW.parent
GOLDEN_FEATURES = FIXTURES / "golden_features_seed2027.csv"
GOLDEN_REJECTS = FIXTURES / "golden_rejects_seed2027.csv"
RAW_NAME = "orders_20260928T140203Z_0001.csv"

# Which preprocess reason each simulator defect must end up as.
DEFECT_TO_REASON = {
    "blank": "blank",
    "negative_distance": "out_of_range",
    "missing_prep": "not_a_number",
    "text_label": "bad_label",
    "outlier_distance": "out_of_range",
    "duplicate_id": "duplicate_id",
}


def preprocess_golden(work_dir: Path):
    """Run the real process_file on the golden raw batch inside ``work_dir``."""
    cfg = load_config({"DASHBITE_DATA_DIR": str(work_dir)})
    (work_dir / "raw").mkdir(parents=True)
    (work_dir / "features").mkdir()
    (work_dir / "quality").mkdir()
    raw = work_dir / "raw" / RAW_NAME
    shutil.copyfile(GOLDEN_RAW, raw)
    return process_file(raw, cfg)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_features_match_golden_byte_for_byte(tmp_path):
    result = preprocess_golden(tmp_path)
    assert result.features_path.read_bytes() == GOLDEN_FEATURES.read_bytes()


def test_rejects_match_golden_byte_for_byte(tmp_path):
    result = preprocess_golden(tmp_path)
    assert result.rejects_path.read_bytes() == GOLDEN_REJECTS.read_bytes()


def test_every_defect_is_rejected_with_its_reason():
    raw_rows = _rows(GOLDEN_RAW)
    rejects = _rows(GOLDEN_REJECTS)
    seen, expected = set(), []
    for row in raw_rows:
        defect = find_defect(row, seen)
        seen.add(row["order_id"])
        if defect is not None:
            expected.append((row["order_id"], DEFECT_TO_REASON[defect]))
    assert [(r["order_id"], r["reason"]) for r in rejects] == expected
    assert {DEFECT_TO_REASON[d] for d in MESSY_DEFECTS} <= {r["reason"] for r in rejects}
    assert len(_rows(GOLDEN_FEATURES)) + len(rejects) == len(raw_rows)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        result = preprocess_golden(Path(tmp))
        shutil.copyfile(result.features_path, GOLDEN_FEATURES)
        shutil.copyfile(result.rejects_path, GOLDEN_REJECTS)
    print(f"wrote {GOLDEN_FEATURES}\nwrote {GOLDEN_REJECTS}", file=sys.stderr)
