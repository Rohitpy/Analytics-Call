"""Per-call detail: classification, evidence, timings and both transcripts.

Replaces the drawer from the old JS frontend. Streamlit 1.30 has no dataframe
row-selection callback and no modal, so the call is chosen from a selectbox and
the detail renders inline.
"""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError
from streamlit_app.formatting import (
    call_options,
    fmt_duration,
    fmt_timings,
    looks_arabic,
    status_badge,
)


def render(api: ApiClient, job: dict) -> None:
    calls = job.get("calls", [])
    if not calls:
        st.info("No calls in this batch yet.")
        return

    options = call_options(calls)
    ids = [call_id for _, call_id in options]
    labels = dict((call_id, label) for label, call_id in options)

    current = st.session_state.get("selected_call")
    index = ids.index(current) if current in ids else 0

    chosen = st.selectbox(
        "Call",
        options=ids,
        index=index,
        format_func=lambda call_id: labels.get(call_id, call_id),
    )
    st.session_state.selected_call = chosen

    try:
        detail = api.call_detail(job["id"], chosen)
    except ApiError as exc:
        st.error(exc.message)
        return

    _render_detail(detail)


def _render_detail(detail: dict) -> None:
    classification = detail.get("classification") or {}
    transcription = detail.get("transcription") or {}
    translation = detail.get("translation") or {}

    st.markdown(f"### {detail['filename']}")
    st.caption(f"Sr. No {detail['sr_no']} - `{detail['id']}`")

    if detail.get("error"):
        st.error(
            f"Failed at stage **{detail.get('failed_stage') or 'unknown'}**\n\n"
            f"{detail['error']}"
        )

    if classification:
        top = st.columns(4)
        top[0].metric("Theme", classification.get("theme", "-"))
        top[1].metric("Specific issue", classification.get("issue", "-"))
        top[2].metric("Confidence", f"{classification.get('confidence', 0):.2f}")
        top[3].metric("Sentiment", classification.get("sentiment") or "-")

        # These flags are how a gap in the taxonomy surfaces - worth showing
        # rather than burying.
        if not classification.get("theme_matched", True):
            st.warning(
                "The model proposed a theme outside the taxonomy; it was "
                "remapped to the fallback. Consider adding it to themes.yaml."
            )
        elif not classification.get("issue_matched", True):
            st.info(
                "This issue is not one of the predefined issues for the theme. "
                "The model's own wording was kept - a candidate to add to "
                "themes.yaml."
            )

        if classification.get("reason"):
            st.markdown("**Reason for the issue**")
            st.write(classification["reason"])

        if classification.get("reasoning"):
            st.markdown("**AI reasoning**")
            st.write(classification["reasoning"])

        evidence = classification.get("evidence") or []
        if evidence:
            st.markdown("**Evidence from the call**")
            for quote in evidence:
                st.markdown(f"> {quote}")

    meta = st.columns(4)
    meta[0].metric("Status", status_badge(detail.get("status", "")))
    meta[1].metric("Language", transcription.get("language", "-"))
    meta[2].metric("Duration", fmt_duration(transcription.get("duration_seconds")))
    meta[3].metric(
        "Translated", "yes" if translation.get("translated") else "no (English)"
    )
    st.caption(f"Stage timings: {fmt_timings(detail.get('timings'))}")

    flags = [k for k, v in (transcription.get("silence_flags") or {}).items() if v]
    if flags:
        st.caption(f"Silence flags: {', '.join(flags)}")

    final_text = translation.get("text") or transcription.get("text") or ""
    original_text = transcription.get("text") or ""

    tabs = st.tabs(["Final transcript (English)", "Original transcript"])
    with tabs[0]:
        if final_text:
            st.text_area(
                "Final transcript", final_text, height=340, label_visibility="collapsed"
            )
        else:
            st.info("No transcript - no speech was detected in this call.")
    with tabs[1]:
        if not original_text:
            st.info("No original transcript.")
        elif looks_arabic(original_text):
            # text_area cannot render right-to-left, so Arabic goes through
            # markdown with a dir attribute instead.
            st.markdown(
                f"<div dir='rtl' style='white-space:pre-wrap;line-height:1.7;"
                f"max-height:340px;overflow-y:auto'>{_escape(original_text)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.text_area(
                "Original transcript",
                original_text,
                height=340,
                label_visibility="collapsed",
            )


def _escape(text: str) -> str:
    """The transcript is model output - never inject it as raw HTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
