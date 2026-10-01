"""Model Pulse chart builders: pure Altair, no Streamlit.

Each builder takes a series from ``pipeline.pulse`` plus the headline that
states its finding, and returns a chart object, so tests can check titles and
axes without a browser. One accent colour for data, grey for reference marks,
no gridlines, left-aligned titles; theme defaults handle light and dark mode.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from pipeline.pulse import bucket_minutes_of, per_bucket

ACCENT = "#3b6fd8"
REFERENCE = "#9e9e9e"
HEIGHT = 220
TIME_AXIS = alt.Axis(title="Time (UTC)", format="%H:%M", labelAngle=0, grid=False)


def _title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, anchor="start")


def _time_x(series: pd.Series) -> alt.X:
    """The full window on the x-axis, in UTC, so every time chart lines up with the hero."""
    bucket = pd.Timedelta(minutes=bucket_minutes_of(series))
    start, end = series.index[0], series.index[-1] + bucket
    return alt.X("time:T", axis=TIME_AXIS,
                 scale=alt.Scale(type="utc", domain=[_utc(start), _utc(end)]))


def _utc(moment: pd.Timestamp) -> alt.DateTime:
    """A Vega-Lite UTC date, which survives Streamlit's data transport unchanged."""
    moment = moment.tz_convert("UTC")
    return alt.DateTime(year=moment.year, month=moment.month, date=moment.day,
                        hours=moment.hour, minutes=moment.minute, seconds=moment.second,
                        utc=True)


def _time_frame(series: pd.Series, value: str) -> pd.DataFrame:
    frame = pd.DataFrame({"time": series.index, value: series.to_numpy()})
    frame["label"] = frame["time"].dt.strftime("%H:%M")  # tooltip text, already in UTC
    return frame


def orders_chart(series: pd.Series, title: str) -> alt.Chart:
    """The hero: orders per bucket over the recent window, one line."""
    unit = per_bucket(bucket_minutes_of(series))
    frame = _time_frame(series, "orders")
    return (
        alt.Chart(frame, title=_title(title), height=HEIGHT)
        .mark_line(color=ACCENT, point=alt.OverlayMarkDef(color=ACCENT, size=18))
        .encode(
            x=_time_x(series),
            y=alt.Y("orders:Q", axis=alt.Axis(title=f"Orders {unit}", grid=False)),
            tooltip=[alt.Tooltip("label:N", title="Minute (UTC)"),
                     alt.Tooltip("orders:Q", title="Orders", format=",")],
        )
    )


def flag_rate_chart(series: pd.Series, title: str) -> alt.LayerChart:
    """Model output: % flagged late per bucket as a step line, fixed 0-100, gaps where empty."""
    frame = _time_frame(series, "pct")
    # Each unbroken run of buckets is its own segment, so an empty bucket is a
    # visible gap in every Vega-Lite version rather than a line drawn across it.
    frame["segment"] = frame["pct"].isna().cumsum()
    frame = frame.dropna(subset=["pct"])
    average = float(frame["pct"].mean())

    line = (
        alt.Chart(frame)
        .mark_line(color=ACCENT, interpolate="step-after")
        .encode(
            x=_time_x(series),
            y=alt.Y("pct:Q", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(title="% flagged late", grid=False)),
            detail="segment:N",
            tooltip=[alt.Tooltip("label:N", title="Minute (UTC)"),
                     alt.Tooltip("pct:Q", title="% flagged late", format=".0f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"pct": [average]}))
        .mark_rule(color=REFERENCE, strokeDash=[4, 4], strokeWidth=1)
        .encode(y="pct:Q",
                tooltip=[alt.Tooltip("pct:Q", title="Window average %", format=".0f")])
    )
    return alt.layer(line, rule, title=_title(title), height=HEIGHT)


def failures_chart(counts: pd.Series, title: str) -> alt.LayerChart:
    """Field failures: one horizontal bar per reason, largest on top, count at each bar's end."""
    frame = pd.DataFrame({"reason": counts.index, "rows": counts.to_numpy()})
    order = frame["reason"].tolist()  # already sorted, most common first
    y = alt.Y("reason:N", sort=order, axis=alt.Axis(title=None, labelLimit=260))
    bars = (
        alt.Chart(frame)
        .mark_bar(color=ACCENT)
        .encode(
            # Headroom past the longest bar so its count label isn't clipped.
            x=alt.X("rows:Q", scale=alt.Scale(domain=[0, int(frame["rows"].max() * 1.15) + 1]),
                    axis=alt.Axis(title="Rows dropped", grid=False, tickMinStep=1, tickCount=5)),
            y=y,
            tooltip=[alt.Tooltip("reason:N", title="Reason"),
                     alt.Tooltip("rows:Q", title="Rows dropped", format=",")],
        )
    )
    # Neutral grey reads on both Streamlit themes; the default text colour vanished on dark.
    labels = bars.mark_text(align="left", baseline="middle", dx=4, color=REFERENCE).encode(
        text=alt.Text("rows:Q", format=",")
    )
    # A per-row step (not a fixed pixel height) is how Vega-Lite sizes a
    # categorical axis; a fixed height gets squashed when Streamlit renders it.
    return alt.layer(bars, labels, title=_title(title), height=alt.Step(28))
