"""Taxonomy tab - read the active themes and hot-reload themes.yaml.

Useful during prompt engineering: edit backend/data/themes.yaml on the server,
hit Reload, and the next call is classified against the new taxonomy without a
backend restart.
"""

from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError


def render(api: ApiClient) -> None:
    header = st.columns([3, 1])
    header[0].markdown(
        "The taxonomy below is rendered into the classification prompt for "
        "every call. Edit `backend/data/themes.yaml` on the server, then "
        "reload."
    )
    if header[1].button("Reload themes.yaml", use_container_width=True):
        try:
            api.reload_themes()
            st.toast("Taxonomy reloaded", icon="\N{WHITE HEAVY CHECK MARK}")
        except ApiError as exc:
            # The backend keeps serving the previous taxonomy on a parse error.
            st.error(f"Reload failed, previous taxonomy still active:\n\n{exc.message}")
        st.rerun()

    try:
        payload = api.themes()
    except ApiError as exc:
        st.error(exc.message)
        return

    if not payload.get("loaded"):
        st.error(
            f"Taxonomy file not loaded - a built-in fallback is in use.\n\n"
            f"{payload.get('load_error') or ''}"
        )

    st.caption(
        f"{payload.get('theme_count')} themes, {payload.get('issue_count')} issues "
        f"- `{payload.get('source_file')}`"
    )

    taxonomy = payload.get("taxonomy", {})
    for theme in taxonomy.get("themes", []):
        with st.expander(f"{theme['name']}  -  {len(theme.get('issues', []))} issues"):
            if theme.get("description"):
                st.write(theme["description"])
            if theme.get("keywords"):
                st.caption("Cues: " + ", ".join(theme["keywords"]))
            for issue in theme.get("issues", []):
                st.markdown(f"**{issue['name']}**")
                if issue.get("description"):
                    st.caption(issue["description"])
                if issue.get("reasons"):
                    st.markdown(
                        "\n".join(f"- {reason}" for reason in issue["reasons"])
                    )
                if issue.get("examples"):
                    for example in issue["examples"]:
                        st.markdown(f"> {example}")
