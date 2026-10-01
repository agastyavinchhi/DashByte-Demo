"""Stage 6 contracts: Model Pulse reads what the stages write, read-only, and the page keeps its shape.

Golden page numbers come from a fixed scenario built by the real stages
(tests/pulse_scenario.py). If a change is deliberate, regenerate and review:

    .venv/bin/python -m tests.regression.test_pulse_contracts
"""
from __future__ import annotations

import ast
import inspect
import json
import math
import os
import shutil
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest

from pipeline import infer, preprocess, pulse, pulse_charts, runner, simulator, train
from pipeline.config import load_config
from tests import pulse_scenario

pytestmark = pytest.mark.regression

GOLDEN = Path(__file__).resolve().parent.parent / "fixtures" / "golden_pulse.json"
PROJECT = Path(__file__).resolve().parent.parent.parent
# The wall clock for golden numbers: 20s after the newest order, so the feed reads as live.
NOW = pd.Timestamp(pulse_scenario.END) + pd.Timedelta(seconds=20)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    data = tmp_path_factory.mktemp("pulse") / "data"
    pulse_scenario.build(data)
    return data


def view_for(data: Path, now=NOW):
    cfg = load_config({"DASHBITE_DATA_DIR": str(data)})
    cache = pulse.Cache()
    return pulse.build_view(pulse.load_artifacts(cfg, cache), cfg, now=now, cache=cache)


def page_numbers(view) -> dict:
    """The numbers the page shows, in a JSON-friendly shape."""
    clean = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 3)  # noqa: E731
    return {
        "kpis": [list(k) for k in view.kpis],
        "titles": [view.volume_title, view.flag_title, view.failures_title],
        "volume": [int(v) for v in view.volume],
        "volume_total": int(view.volume.sum()),
        "flag_rate": [clean(float(v)) for v in view.flag_rate],
        "failures": {label: int(n) for label, n in view.failures.items()},
        "drop_rate": clean(view.numbers["drop_rate"]),
        "flag_rate_kpi": clean(view.numbers["flag_rate"]),
    }


# ---------- contracts match the producers ----------

def test_columns_and_globs_match_producers():
    assert set(pulse.FEATURE_COLUMNS) <= set(preprocess.FEATURE_COLUMNS)
    assert pulse.PREDICTION_COLUMNS == infer.PREDICTION_COLUMNS
    assert pulse.FEATURES_GLOB == infer.FEATURES_GLOB == train.FEATURES_GLOB
    assert "reason" in preprocess.REJECT_COLUMNS


def test_batch_key_parsing_matches_producers():
    raw = simulator.batch_filename(pulse_scenario.START, 7)
    key = raw[len("orders_"):-len(".csv")]
    assert preprocess.batch_key(Path(raw)) == key
    assert pulse.batch_key(f"features_{key}.csv", "features_") == key
    assert pulse.batch_key(f"predictions_{key}.csv", "predictions_") == key
    assert pulse.batch_key(f"rejects_{key}.csv", "rejects_") == key
    assert pulse.batch_key(f".features_{key}.csv.tmp", "features_") is None
    assert pulse.batch_time(key) == pd.Timestamp(pulse_scenario.START)


def test_every_reject_reason_has_a_label_and_phrase():
    assert set(pulse.REASON_LABELS) == set(preprocess.REJECT_REASONS)
    assert set(pulse.REASON_PHRASES) == set(pulse.REASON_LABELS.values())


@pytest.mark.parametrize("case", ["clean", "orphan", "damaged", "empty"])
def test_newest_model_rule_matches_train_and_infer(tmp_path, case):
    models = tmp_path / "models"
    models.mkdir()
    for v in (1, 2, 3):
        if case == "empty":
            break
        (models / f"model_v{v:04d}.json").write_text(json.dumps({"rows_total": v}))
    if case == "orphan":
        (models / "model_v0003.json").unlink()
        (models / "model_v0003.joblib").write_bytes(b"x")
    if case == "damaged":
        (models / "model_v0003.json").write_text("{nope")
    cfg = load_config({"DASHBITE_DATA_DIR": str(tmp_path)})
    newest = pulse.newest_model(cfg)
    infer_newest = infer.newest_checkpoint(cfg)
    assert (newest.version if newest else 0) == train.latest_published(cfg).version == (
        infer_newest.version if infer_newest else 0)


# ---------- boundaries ----------

def _imports(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    found = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    found |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    for node in ast.walk(tree):  # `from pipeline import pulse` imports pipeline.pulse
        if isinstance(node, ast.ImportFrom) and node.module == "pipeline":
            found |= {f"pipeline.{a.name}" for a in node.names}
    return found


def _dashboard_imports() -> set[str]:
    source = (PROJECT / "pipeline" / "dashboard.py").read_text()
    tree = ast.parse(source)
    found = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    found |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pipeline":
            found |= {f"pipeline.{a.name}" for a in node.names}
    return found


ALLOWED = {"pipeline", "pipeline.config", "pipeline.paths", "pipeline.pulse", "pipeline.pulse_charts"}


def test_dashboard_modules_import_no_stage():
    for name, imported in [("pulse", _imports(pulse)), ("pulse_charts", _imports(pulse_charts)),
                           ("dashboard", _dashboard_imports())]:
        assert {m for m in imported if m.startswith("pipeline")} <= ALLOWED, name
    for module in (pulse, pulse_charts):
        assert not any(m.split(".")[0] == "streamlit" for m in _imports(module))


def test_dashboard_runs_last_in_the_background_stack():
    # Full stack (2026-10-01): make run starts the dashboard too, after the pipeline,
    # through the same launcher as make dashboard. It's still only a reader of data/.
    assert list(runner.PROCESSES)[-1] == "dashboard"
    assert runner.PROCESSES["dashboard"] == ["-m", "pipeline.dashboard_server"]


def test_helpers_are_read_only(scenario):
    def snapshot():
        return {p: (p.read_bytes(), os.stat(p).st_mtime_ns)
                for p in scenario.rglob("*") if p.is_file()}

    before = snapshot()
    view = view_for(scenario)
    cfg = load_config({"DASHBITE_DATA_DIR": str(scenario)})
    art = pulse.load_artifacts(cfg, pulse.Cache())
    start, end = view.numbers["window"]
    pulse.score_summary(art.predictions)
    pulse.drop_rate(art.features, art.rejects, start, end)
    pulse.failure_counts(art.rejects, start, end)
    assert snapshot() == before


# ---------- golden page numbers ----------

def test_golden_page_numbers(scenario):
    assert page_numbers(view_for(scenario)) == json.loads(GOLDEN.read_text())


def test_golden_scenario_reads_as_a_live_feed(scenario):
    view = view_for(scenario)
    assert view.warning is None
    assert len(view.volume) == 60 and view.volume_title.startswith("Orders are steady")
    assert view.numbers["orders"] == int(view.volume.sum())


# ---------- page shape (rendered headlessly) ----------

def _chart_titles(at) -> list[str]:
    titles = []
    for element in at.get("arrow_vega_lite_chart"):
        spec = json.loads(element.proto.spec)
        t = spec["title"]
        titles.append(t["text"] if isinstance(t, dict) else t)
    return titles


def _axis_titles(at) -> set:
    found = set()
    for element in at.get("arrow_vega_lite_chart"):
        spec = json.loads(element.proto.spec)
        for part in spec.get("layer", [spec]):
            for channel in ("x", "y"):
                enc = part.get("encoding", {}).get(channel, {})
                if "axis" in enc:
                    found.add((enc["axis"] or {}).get("title"))
    return found


def test_page_shape(scenario, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DASHBITE_DATA_DIR", str(scenario))
    at = AppTest.from_file(str(PROJECT / "pipeline" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 3
    assert len(at.dataframe) == 0 and len(at.table) == 0
    assert len(at.get("arrow_vega_lite_chart")) == 3

    expected = view_for(scenario, now=pd.Timestamp.now(tz=timezone.utc))
    assert _chart_titles(at) == [expected.volume_title, expected.flag_title,
                                 expected.failures_title]  # hero, flag rate, failures
    raw = {"predicted_late", "late_probability", "reason", "checkpoint_id", "order_id"}
    assert not (set(_chart_titles(at)) | _axis_titles(at)) & raw
    for metric in at.metric:
        text = f"{metric.label} {metric.value}".lower()
        assert "value" not in text and "revenue" not in text and "$" not in text


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        pulse_scenario.build(data)
        numbers = page_numbers(view_for(data))
        GOLDEN.write_text("{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}"
                                             for k, v in numbers.items()) + "\n}\n")
    print(f"wrote {GOLDEN}", file=sys.stderr)


def test_reset_while_dashboard_runs_shows_only_the_new_run(tmp_path):
    # `make clean-data` with the page open: the long-lived parse cache must not
    # keep showing deleted batches, or mix them into the next run's numbers.
    data = tmp_path / "data"
    cfg = load_config({"DASHBITE_DATA_DIR": str(data)})
    cache = pulse.Cache()
    pulse_scenario.build(data, batches=30)
    first = pulse.load_artifacts(cfg, cache)
    assert first.counts["features"] == 30 and first.newest_model is not None

    shutil.rmtree(data)
    empty = pulse.load_artifacts(cfg, cache)
    assert (len(empty.features), len(empty.predictions), len(empty.rejects)) == (0, 0, 0)
    assert empty.newest_model is None
    view = pulse.build_view(empty, cfg, now=NOW)
    assert [k.value for k in view.kpis] == ["—", "—", "—"]

    later = pd.Timestamp(pulse_scenario.START) + pd.Timedelta(hours=4)
    pulse_scenario.build(data, batches=12, start=later.to_pydatetime())
    second = pulse.load_artifacts(cfg, cache)
    assert second.counts == {"features": 12, "predictions": 12, "rejects": 12}
    assert second.features["timestamp"].min() >= later - pd.Timedelta(minutes=1)
    assert set(second.predictions["batch"]) == set(second.features["batch"])
