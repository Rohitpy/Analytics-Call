"""Presentation helpers.

Deliberately free of Streamlit imports so the table-shaping and filtering logic
can be tested without a script run context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "cancelled"}

STATUS_ICON = {
    "queued": "\N{HOURGLASS WITH FLOWING SAND}",
    "running": "\N{HIGH VOLTAGE SIGN}",
    "completed": "\N{WHITE HEAVY CHECK MARK}",
    "completed_with_errors": "\N{WARNING SIGN}",
    "failed": "\N{CROSS MARK}",
    "cancelled": "\N{BLACK CIRCLE FOR RECORD}",
}

STATUS_LABEL = {
    "queued": "Queued",
    "running": "Running",
    "completed": "Completed",
    "completed_with_errors": "Completed with errors",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def status_badge(status: str) -> str:
    return f"{STATUS_ICON.get(status, '')} {STATUS_LABEL.get(status, status)}".strip()


def fmt_timestamp(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (ValueError, AttributeError):
        return str(value)


def fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def fmt_timings(timings: dict[str, Any] | None) -> str:
    if not timings:
        return "-"
    labels = [
        ("convert", "convert"),
        ("transcribe", "STT"),
        ("translate", "translate"),
        ("classify", "classify"),
        ("total", "total"),
    ]
    parts = [f"{label} {timings[key]}s" for key, label in labels if timings.get(key)]
    return " | ".join(parts) or "-"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def report_dataframe(
    rows: list[dict[str, Any]], columns: list[str], fields: list[str]
) -> pd.DataFrame:
    """Build the report table using the column names and field keys the API
    publishes, so the table follows the report definition rather than a second
    hardcoded copy of it."""
    if not rows:
        return pd.DataFrame(columns=[*columns, "Status"])

    frame = pd.DataFrame(
        {label: [row.get(field, "") for row in rows] for label, field in zip(columns, fields)}
    )
    frame["Status"] = [status_badge(row.get("status", "")) for row in rows]
    return frame


def filter_rows(
    rows: list[dict[str, Any]], text: str = "", themes: list[str] | None = None
) -> list[dict[str, Any]]:
    needle = (text or "").strip().lower()
    selected = set(themes or [])

    def keep(row: dict[str, Any]) -> bool:
        if selected and row.get("theme") not in selected:
            return False
        if not needle:
            return True
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("file_name", "theme", "specific_issue", "reason_for_issue")
        ).lower()
        return needle in haystack

    return [row for row in rows if keep(row)]


def distribution_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Theme x issue counts over successfully classified calls only - failures
    would otherwise show up as a fake 'PROCESSING FAILED' theme."""
    done = [r for r in rows if r.get("status") == "completed"]
    if not done:
        return pd.DataFrame(columns=["Theme", "Specific issue", "Calls", "Share"])

    counts: dict[tuple[str, str], int] = {}
    for row in done:
        key = (row.get("theme", ""), row.get("specific_issue", ""))
        counts[key] = counts.get(key, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = len(done)
    return pd.DataFrame(
        {
            "Theme": [theme for (theme, _), _ in ordered],
            "Specific issue": [issue for (_, issue), _ in ordered],
            "Calls": [count for _, count in ordered],
            "Share": [f"{count / total * 100:.1f}%" for _, count in ordered],
        }
    )


def call_progress_dataframe(calls: list[dict[str, Any]]) -> pd.DataFrame:
    if not calls:
        return pd.DataFrame(columns=["#", "File", "Stage", "Status", "Progress"])
    return pd.DataFrame(
        {
            "#": [c.get("sr_no") for c in calls],
            "File": [c.get("filename") for c in calls],
            "Stage": [
                c.get("stage") if c.get("status") == "running" else "-" for c in calls
            ],
            "Status": [status_badge(c.get("status", "")) for c in calls],
            "Progress": [int(c.get("progress", 0)) for c in calls],
        }
    )


def call_options(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(label, call_id) pairs for the detail picker."""
    options = []
    for call in sorted(calls, key=lambda c: c.get("sr_no", 0)):
        icon = STATUS_ICON.get(call.get("status", ""), "")
        options.append(
            (f"{icon} {call.get('sr_no')}. {call.get('filename')}", call.get("id", ""))
        )
    return options


def looks_arabic(text: str) -> bool:
    """Cheap script check, only used to decide RTL rendering in the UI.

    Compares codepoints rather than literal Arabic characters so this file
    stays pure ASCII - literal ranges break if any transfer re-encodes them.
    """
    if not text:
        return False
    arabic = sum(
        1
        for ch in text
        if 0x0600 <= ord(ch) <= 0x06FF or 0xFB50 <= ord(ch) <= 0xFEFC
    )
    return arabic > len(text) * 0.1
