"""Batch header: status, counters, progress, and the batch-level actions."""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError
from streamlit_app.config import UiSettings
from streamlit_app.formatting import (
    call_progress_dataframe,
    fmt_timestamp,
    is_terminal,
    status_badge,
)


def render(api: ApiClient, settings: UiSettings, job: dict) -> None:
    running = not is_terminal(job["status"])

    st.subheader(job["name"] or job["id"])
    st.caption(
        f"`{job['id']}` - created {fmt_timestamp(job['created_at'])}"
        + (f" - finished {fmt_timestamp(job['finished_at'])}" if job.get("finished_at") else "")
    )

    counters = st.columns(5)
    counters[0].metric("Status", status_badge(job["status"]))
    counters[1].metric("Calls", job["total"])
    counters[2].metric("Classified", job["completed"])
    counters[3].metric("Failed", job["failed"])
    counters[4].metric("Pending", job["pending"])

    st.progress(
        min(100, max(0, job["progress"])) / 100,
        text=f"{job['progress']}% complete",
    )

    if job.get("error"):
        st.warning(job["error"])

    _actions(api, settings, job, running)

    if running:
        st.markdown("**Live progress**")
        st.dataframe(
            call_progress_dataframe(job.get("calls", [])),
            hide_index=True,
            use_container_width=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Progress": st.column_config.ProgressColumn(
                    "Progress", min_value=0, max_value=100, format="%d%%"
                ),
            },
        )


def _actions(api: ApiClient, settings: UiSettings, job: dict, running: bool) -> None:
    buttons = st.columns([2, 1, 1, 1])

    if job.get("excel_available"):
        try:
            workbook = _cached_export(
                settings.api_url,
                job["id"],
                # Refetch only when the batch actually changes.
                f"{job['status']}:{job['completed']}:{job['failed']}",
                settings.export_timeout,
            )
            safe_name = (job["name"] or job["id"]).replace(" ", "_").replace("/", "-")
            buttons[0].download_button(
                "Download Excel report",
                data=workbook,
                file_name=f"theme_analysis_{safe_name}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                ),
                type="primary",
                use_container_width=True,
            )
        except ApiError as exc:
            buttons[0].error(f"Report unavailable: {exc.message}")
    else:
        buttons[0].button(
            "Download Excel report",
            disabled=True,
            use_container_width=True,
            help="Available once the batch finishes.",
        )

    if buttons[1].button("Refresh", use_container_width=True):
        st.rerun()

    if running:
        if buttons[2].button("Cancel", use_container_width=True):
            try:
                api.cancel_job(job["id"])
                st.toast("Cancellation requested - running calls will finish.")
            except ApiError as exc:
                st.error(exc.message)
            st.rerun()
    else:
        # Two-step delete: Streamlit 1.30 has no modal, and a single-click
        # delete next to a Download button is too easy to hit by accident.
        confirm_key = f"confirm_delete_{job['id']}"
        if st.session_state.get(confirm_key):
            if buttons[2].button("Confirm delete", type="primary", use_container_width=True):
                try:
                    api.delete_job(job["id"])
                    st.session_state.selected_job = None
                    st.session_state.selected_call = None
                    st.session_state[confirm_key] = False
                    st.toast("Batch deleted")
                except ApiError as exc:
                    st.error(exc.message)
                st.rerun()
            if buttons[3].button("Keep it", use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()
        elif buttons[2].button("Delete", use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()


@st.cache_data(show_spinner=False, ttl=900, max_entries=16)
def _cached_export(api_url: str, job_id: str, cache_key: str, timeout: float) -> bytes:
    """Fetched through the UI rather than linked directly, so the download
    works even when the browser cannot reach the API host (SSH tunnel, or the
    API bound to loopback). `cache_key` changes with the batch state, so the
    workbook is pulled once per state rather than on every rerun."""
    return ApiClient(api_url).export(job_id, timeout=timeout)
