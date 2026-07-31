"""Streamlit UI for Theme Analytics.

A pure client of the FastAPI backend - it holds no pipeline logic and reaches
nothing but the documented REST endpoints, exactly as the previous browser
frontend did. Run the two side by side:

    ./run.sh                 # API on :8000 and this UI on :8501
    streamlit run streamlit_app/app.py --server.port 8501

Targets Streamlit 1.30 (the version on the deployment box), so nothing here
uses st.fragment, st.dialog, or dataframe row-selection callbacks.
"""

__all__ = ["api_client", "config", "formatting", "views"]
