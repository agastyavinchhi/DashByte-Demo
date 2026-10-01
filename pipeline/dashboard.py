"""Model Pulse: a one-page ML health dashboard that only reads data/.

Run it with ``make dashboard``. Layout only: every number, series and title
comes from ``pipeline.pulse`` and every chart from ``pipeline.pulse_charts``.
Three KPIs and three charts, each titled with the finding it shows.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline import pulse, pulse_charts
from pipeline.config import load_config

st.set_page_config(page_title="Model Pulse", layout="centered")
st.title("Model Pulse")

try:
    cfg = load_config()
except ValueError as exc:
    st.error(f"config error: {exc}")
    st.stop()


@st.cache_resource
def _cache_for(data_dir: str) -> pulse.Cache:
    """One parse cache per data folder, kept for the life of the dashboard process."""
    return pulse.Cache()


@st.fragment(run_every=cfg.dashboard_refresh_seconds)
def body() -> None:
    cache = _cache_for(str(cfg.data_dir))
    view = pulse.build_view(pulse.load_artifacts(cfg, cache), cfg,
                            now=pd.Timestamp.now(tz="UTC"), cache=cache)
    print(f"[dashboard {datetime.now():%H:%M:%S}] {view.log_line}", flush=True)

    st.caption(view.caption)
    if view.warning:
        st.warning(view.warning)

    for column, kpi in zip(st.columns(3), view.kpis):
        column.metric(kpi.label, kpi.value, delta=kpi.delta, delta_color=kpi.delta_color)

    if view.volume is not None:
        st.altair_chart(pulse_charts.orders_chart(view.volume, view.volume_title),
                        use_container_width=True)
    else:
        st.info(view.volume_title)

    if view.flag_rate is not None:
        st.altair_chart(pulse_charts.flag_rate_chart(view.flag_rate, view.flag_title),
                        use_container_width=True)
    else:
        st.info(view.flag_title)

    if view.failures is not None:
        st.altair_chart(pulse_charts.failures_chart(view.failures, view.failures_title),
                        use_container_width=True)
    else:
        st.info(view.failures_title)


body()
