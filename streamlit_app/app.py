"""Theme Analytics - Streamlit UI.

    streamlit run streamlit_app/app.py

Talks to the FastAPI backend over HTTP only. Point it elsewhere with
THEME_ANALYTICS_API_URL if the API runs on another host.

The page never reruns on a timer. Every rerun is something the user did -
a button, a widget, or the browser's rerun shortcut. A timed rerun loop
(time.sleep + st.rerun) also blocks the script thread, which makes Ctrl+C
slow to take effect, so removing it keeps shutdown immediate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run` puts the script's own directory on sys.path, not the repo
# root, so the package import below would fail without this.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamlit_app.api_client import ApiClient, ApiError  # noqa: E402
from streamlit_app.config import get_ui_settings  # noqa: E402
from streamlit_app.formatting import is_terminal  # noqa: E402
from streamlit_app.views import (  # noqa: E402
    call_detail,
    job_panel,
    report,
    sidebar,
    taxonomy,
)

SETTINGS = get_ui_settings()

st.set_page_config(
    page_title=SETTINGS.page_title,
    page_icon=SETTINGS.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "selected_job": None,
        "selected_call": None,
        "uploader_generation": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_welcome() -> None:
    st.markdown(
        """
Upload call recordings in the sidebar. Each call is:

1. converted to 16 kHz mono PCM with **ffmpeg**
2. transcribed by **Whisper large-v3** with silero VAD chunking
3. translated to English by the **LLM**, only when it is not already English
4. classified into a **theme**, a **specific issue**, and the **reason** for it

The result is an Excel report with `Sr. No, FileName, Theme, Specific Issue for
the call, Transcription, AI Reasoning`, downloadable here or from the API.
        """
    )
    st.info("Select an existing batch in the sidebar, or upload new recordings.")


def main() -> None:
    init_state()
    api = ApiClient(SETTINGS.api_url, timeout=SETTINGS.request_timeout)

    sidebar.render(api, SETTINGS)

    # A static heading, not user input, so the hardcoded markup is safe as-is.
    # st.title() has no alignment option, hence the raw <h1>.
    st.markdown(
        "<h1 style='text-align: center;'>Theme Analytics</h1>",
        unsafe_allow_html=True,
    )

    job_id = st.session_state.get("selected_job")
    if not job_id:
        render_welcome()
        return

    try:
        job = api.get_job(job_id)
    except ApiError as exc:
        if exc.is_not_found:
            # Deleted from another session or purged by retention.
            st.session_state.selected_job = None
            st.warning("That batch no longer exists.")
            st.rerun()
        st.error(exc.message)
        return

    job_panel.render(api, SETTINGS, job)
    st.divider()

    running = not is_terminal(job["status"])

    try:
        results = api.results(
            job_id, transcript_chars=SETTINGS.transcript_preview_chars
        )
    except ApiError as exc:
        st.error(exc.message)
        return

    tabs = st.tabs(["Report", "Call detail", "Theme distribution", "Taxonomy"])
    with tabs[0]:
        report.render(results)
    with tabs[1]:
        call_detail.render(api, job)
    with tabs[2]:
        report.render_distribution(results)
    with tabs[3]:
        taxonomy.render(api)

    if running:
        st.divider()
        st.info(
            "This batch is still processing. The page does not refresh on its "
            "own - press **R**, or use the Refresh button above, to see the "
            "latest progress."
        )


main()
