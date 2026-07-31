"""The report tab: the six report columns, filters, and the theme rollup."""

from __future__ import annotations

import streamlit as st

from streamlit_app.formatting import (
    distribution_dataframe,
    filter_rows,
    report_dataframe,
)


def render(results: dict) -> None:
    rows = results.get("rows", [])
    if not rows:
        st.info("No calls in this batch yet.")
        return

    controls = st.columns([2, 2])
    text = controls[0].text_input(
        "Filter", placeholder="file, theme, issue or reason...", key="row_filter"
    )
    themes = sorted({row.get("theme", "") for row in rows if row.get("theme")})
    selected = controls[1].multiselect("Theme", options=themes, key="theme_filter")

    visible = filter_rows(rows, text, selected)
    st.caption(
        f"Showing {len(visible)} of {len(rows)} calls."
        + (
            "  Transcripts are shortened here - open the Call detail tab or the "
            "Excel report for the full text."
            if results.get("transcripts_truncated")
            else ""
        )
    )

    if not visible:
        st.warning("No rows match the filter.")
        return

    st.dataframe(
        report_dataframe(visible, results["columns"], results["fields"]),
        hide_index=True,
        use_container_width=True,
        height=min(620, 90 + 35 * len(visible)),
        column_config={
            "Sr. No": st.column_config.NumberColumn(width="small"),
            "FileName": st.column_config.TextColumn(width="medium"),
            "Theme": st.column_config.TextColumn(width="medium"),
            "Specific Issue for the call": st.column_config.TextColumn(width="medium"),
            "Transcription": st.column_config.TextColumn(width="large"),
            "AI Reasoning": st.column_config.TextColumn(width="large"),
        },
    )


def render_distribution(results: dict) -> None:
    rows = results.get("rows", [])
    frame = distribution_dataframe(rows)
    if frame.empty:
        st.info("Nothing classified yet.")
        return

    st.caption(
        "Counts cover successfully classified calls only; failed calls are "
        "excluded so they cannot inflate a theme."
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Calls": st.column_config.NumberColumn(width="small"),
            "Share": st.column_config.TextColumn(width="small"),
        },
    )
