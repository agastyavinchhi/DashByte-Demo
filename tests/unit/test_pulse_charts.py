from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.pulse_charts import ACCENT, failures_chart, flag_rate_chart, orders_chart

pytestmark = pytest.mark.unit

T0 = pd.Timestamp("2026-09-28T14:00:00Z")
RAW_COLUMNS = {"predicted_late", "late_probability", "reason", "order_id", "checkpoint_id",
               "orders", "pct", "rows"}


def series(values, minutes=1):
    return pd.Series(values, index=pd.date_range(T0, periods=len(values), freq=f"{minutes}min"),
                     dtype="float64")


def layers(chart):
    spec = chart.to_dict()
    return spec, spec.get("layer", [spec])


def axis_titles(chart):
    _, parts = layers(chart)
    titles = {}
    for part in parts:
        for channel in ("x", "y"):
            enc = part.get("encoding", {}).get(channel)
            if enc and "axis" in enc:
                titles[channel] = (enc["axis"] or {}).get("title")
    return titles


def title(chart):
    t = chart.to_dict()["title"]
    return t["text"] if isinstance(t, dict) else t


def assert_one_series(chart):
    _, parts = layers(chart)
    for part in parts:
        assert "color" not in part.get("encoding", {})  # no legend, no second series


def test_orders_chart():
    chart = orders_chart(series([400, 380, 0, 410]), "Orders are steady at ~400 per minute")
    assert title(chart) == "Orders are steady at ~400 per minute"
    assert chart.to_dict()["title"]["anchor"] == "start"
    assert axis_titles(chart) == {"x": "Time (UTC)", "y": "Orders per minute"}
    spec = chart.to_dict()
    x = spec["encoding"]["x"]
    assert x["scale"]["type"] == "utc" and x["axis"]["format"] == "%H:%M"
    assert x["axis"]["labelAngle"] == 0
    assert spec["mark"]["color"] == ACCENT
    assert_one_series(chart)


def test_orders_chart_names_a_wider_bucket():
    chart = orders_chart(series([60, 61], minutes=15), "t")
    assert axis_titles(chart)["y"] == "Orders per 15 min"


def test_flag_rate_chart_is_a_fixed_scale_step_line_with_gaps():
    chart = flag_rate_chart(series([40, 45, np.nan, np.nan, 50]), "The model is flagging …")
    spec, parts = layers(chart)
    line, rule = parts
    assert line["mark"]["interpolate"] == "step-after"
    assert line["encoding"]["y"]["scale"]["domain"] == [0, 100]
    assert axis_titles(chart) == {"x": "Time (UTC)", "y": "% flagged late"}
    assert rule["mark"]["type"] == "rule" and rule["mark"]["strokeDash"]

    data = spec["datasets"][line["data"]["name"]]
    assert [row["pct"] for row in data] == [40, 45, 50]  # NaN buckets are not drawn...
    assert data[1]["segment"] != data[2]["segment"]  # ...and the line breaks across them
    assert_one_series(chart)


def test_failures_chart_is_horizontal_sorted_with_labels_and_no_table():
    counts = pd.Series([41, 30, 3], index=["Value out of range", "Missing value",
                                           "Late flag not 0/1"])
    chart = failures_chart(counts, "Out-of-range values cause most dropped rows (55%)")
    spec, (bars, labels) = layers(chart)
    assert bars["mark"]["type"] == "bar"
    assert bars["encoding"]["y"]["field"] == "reason"  # reason on y: horizontal bars
    assert bars["encoding"]["y"]["sort"] == ["Value out of range", "Missing value",
                                             "Late flag not 0/1"]
    assert bars["encoding"]["x"]["axis"]["title"] == "Rows dropped"
    assert bars["encoding"]["y"]["axis"]["title"] is None
    assert labels["mark"]["type"] == "text" and labels["encoding"]["text"]["field"] == "rows"
    assert title(chart) == "Out-of-range values cause most dropped rows (55%)"
    assert_one_series(chart)


def test_no_title_or_axis_is_a_raw_column_name():
    charts = [
        orders_chart(series([1, 2]), "Orders are steady at ~2 per minute"),
        flag_rate_chart(series([10, 20]), "The model is flagging a steady 15% of orders late"),
        failures_chart(pd.Series([1], index=["Missing value"]), "Missing values cause most …"),
    ]
    for chart in charts:
        assert title(chart) not in RAW_COLUMNS
        assert not set(axis_titles(chart).values()) & RAW_COLUMNS


def test_time_charts_share_the_full_window():
    # Regression: with predictions only in the newest minute, the flag chart's
    # axis shrank to that minute and repeated "01:31" across every tick.
    volume = series([0] * 58 + [190, 400])
    flags = series([np.nan] * 59 + [41.0])
    domains = [layers(orders_chart(volume, "t"))[1][0]["encoding"]["x"]["scale"]["domain"],
               layers(flag_rate_chart(flags, "t"))[1][0]["encoding"]["x"]["scale"]["domain"]]
    assert domains[0] == domains[1]
    start, end = domains[0]
    assert (start["hours"], start["minutes"], end["hours"], end["minutes"]) == (14, 0, 15, 0)
    assert start["utc"] is end["utc"] is True
