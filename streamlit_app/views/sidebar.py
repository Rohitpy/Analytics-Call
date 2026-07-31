"""Sidebar: backend health, the upload form, and the batch picker."""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError
from streamlit_app.config import UiSettings
from streamlit_app.formatting import fmt_timestamp, status_badge


def render(api: ApiClient, settings: UiSettings) -> None:
    with st.sidebar:
        st.title(settings.page_title)
        _readiness(api)
        st.divider()
        _upload(api, settings)
        st.divider()
        _job_picker(api)


# --------------------------------------------------------------------------
def _readiness(api: ApiClient) -> None:
    try:
        ready = api.readiness()
    except ApiError as exc:
        st.error(f"Backend unreachable\n\n{exc.message}")
        st.caption(
            "Start it with `./run.sh --api-only`, or set THEME_ANALYTICS_API_URL "
            "if it runs on another host."
        )
        return

    ok = ready.get("status") == "ready"
    summary = (
        f"STT `{ready.get('stt_backend')}` | "
        f"LLM `{ready.get('llm_model') or 'unknown'}` | "
        f"{ready.get('workers')} workers"
    )
    if ok:
        st.success(f"Backend ready\n\n{summary}")
    else:
        st.warning(f"Backend degraded\n\n{summary}")

    with st.expander("Dependency detail", expanded=not ok):
        st.write(
            {
                "ffmpeg available": ready.get("ffmpeg_available"),
                "STT model loaded": ready.get("stt_model_loaded"),
                "LLM reachable": ready.get("llm_reachable"),
                "LLM endpoint": ready.get("llm_base_url"),
                "themes loaded": ready.get("taxonomy_themes"),
                "queue depth": ready.get("queue_depth"),
            }
        )
        for key, value in (ready.get("details") or {}).items():
            st.caption(f"**{key}**: {value}")


# --------------------------------------------------------------------------
def _upload(api: ApiClient, settings: UiSettings) -> None:
    st.subheader("New batch")

    try:
        backend_config = _cached_config(settings.api_url)
    except ApiError:
        backend_config = {
            "allowed_extensions": [".wav", ".mp3", ".m4a"],
            "max_files_per_job": 200,
            "max_upload_mb": 200,
        }

    extensions = [e.lstrip(".") for e in backend_config.get("allowed_extensions", [])]
    max_files = backend_config.get("max_files_per_job", 200)

    name = st.text_input("Batch name", placeholder="e.g. Retail queue - 30 Jul")

    # Bumping the key resets the widget after a successful upload; Streamlit
    # has no other way to clear a file_uploader.
    uploaded = st.file_uploader(
        "Call recordings",
        type=extensions or None,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_generation', 0)}",
        help=f"Up to {max_files} files, {backend_config.get('max_upload_mb', 200)} MB each.",
    )

    count = len(uploaded or [])
    if count > max_files:
        st.error(f"{count} files selected; the backend accepts {max_files} per batch.")

    if st.button(
        f"Process {count} call{'s' if count != 1 else ''}" if count else "Process calls",
        type="primary",
        disabled=not count or count > max_files,
        use_container_width=True,
    ):
        _submit(api, settings, uploaded, name)


def _submit(api: ApiClient, settings: UiSettings, uploaded, name: str) -> None:
    payload = [
        (f.name, f.getvalue(), getattr(f, "type", "") or "application/octet-stream")
        for f in uploaded
    ]
    total_mb = sum(len(c) for _, c, _ in payload) / 1048576

    with st.spinner(f"Uploading {len(payload)} file(s), {total_mb:.1f} MB..."):
        try:
            created = api.create_job(payload, name, timeout=settings.upload_timeout)
        except ApiError as exc:
            st.error(exc.message)
            return

    accepted = len(created.get("accepted", []))
    rejected = created.get("rejected", [])
    st.session_state.selected_job = created["job_id"]
    st.session_state.uploader_generation = (
        st.session_state.get("uploader_generation", 0) + 1
    )
    st.session_state.selected_call = None

    st.toast(f"Batch queued - {accepted} call(s) accepted", icon="\N{WHITE HEAVY CHECK MARK}")
    for item in rejected:
        st.toast(
            f"Rejected {item.get('filename')}: {item.get('reason')}",
            icon="\N{WARNING SIGN}",
        )
    st.rerun()


# --------------------------------------------------------------------------
def _job_picker(api: ApiClient) -> None:
    header = st.columns([3, 1])
    header[0].subheader("Batches")
    if header[1].button("Refresh", use_container_width=True):
        st.rerun()

    try:
        jobs = api.list_jobs().get("jobs", [])
    except ApiError as exc:
        st.error(exc.message)
        return

    if not jobs:
        st.caption("No batches yet.")
        return

    ids = [job["id"] for job in jobs]
    labels = {
        job["id"]: (
            f"{status_badge(job['status'])} - {job['name'] or job['id']}\n\n"
            f"{job['completed']}/{job['total']} classified"
            + (f" - {job['failed']} failed" if job["failed"] else "")
            + f" - {fmt_timestamp(job['created_at'])}"
        )
        for job in jobs
    }

    current = st.session_state.get("selected_job")
    index = ids.index(current) if current in ids else 0

    chosen = st.radio(
        "Select a batch",
        options=ids,
        index=index,
        format_func=lambda job_id: labels[job_id],
        label_visibility="collapsed",
    )
    if chosen != current:
        st.session_state.selected_job = chosen
        st.session_state.selected_call = None
        st.rerun()


@st.cache_data(show_spinner=False, ttl=300)
def _cached_config(api_url: str) -> dict:
    """Upload limits change only on a backend restart."""
    return ApiClient(api_url).config()
