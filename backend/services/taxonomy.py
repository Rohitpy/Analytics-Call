"""Loads and serves the theme taxonomy.

Held in memory and hot-reloadable, so an analyst can edit themes.yaml and
POST /api/v1/themes/reload without dropping in-flight jobs.
"""

from __future__ import annotations

import threading
from pathlib import Path

from backend.core.config import Settings
from backend.core.errors import ValidationError
from backend.core.logging_config import get_logger
from backend.core.yaml_compat import load_file
from backend.schemas.theme import IssueDefinition, Taxonomy, ThemeDefinition

logger = get_logger(__name__)


# Used only when themes.yaml is missing or unreadable, so that the service can
# still start and report the problem through /health/ready.
_MINIMAL_FALLBACK = Taxonomy(
    version="fallback",
    domain="Contact centre",
    themes=[
        ThemeDefinition(
            name="Others",
            description="Fallback theme - the taxonomy file could not be loaded.",
            issues=[IssueDefinition(name="Unclassified")],
        )
    ],
)


class TaxonomyService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._path: Path = settings.themes_file
        self._lock = threading.RLock()
        self._taxonomy: Taxonomy = _MINIMAL_FALLBACK
        self._loaded_ok = False
        self._load_error: str | None = None

    # ---- lifecycle ---------------------------------------------------------
    def load(self) -> Taxonomy:
        """Read themes.yaml from disk. Never raises - falls back instead."""
        with self._lock:
            try:
                raw = load_file(self._path)
                if not isinstance(raw, dict):
                    raise ValueError("themes.yaml must contain a mapping at the top level")
                taxonomy = Taxonomy.model_validate(raw)
                if not taxonomy.themes:
                    raise ValueError("themes.yaml defines no themes")

                self._validate(taxonomy)
                self._taxonomy = taxonomy
                self._loaded_ok = True
                self._load_error = None
                logger.info(
                    "Taxonomy loaded: %d themes, %d issues (version %s)",
                    len(taxonomy.themes),
                    taxonomy.total_issues(),
                    taxonomy.version,
                )
            except Exception as exc:  # missing file, bad YAML, schema error
                self._taxonomy = _MINIMAL_FALLBACK
                self._loaded_ok = False
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.error("Failed to load taxonomy from %s: %s", self._path, exc)
            return self._taxonomy

    def reload(self) -> Taxonomy:
        logger.info("Reloading taxonomy from %s", self._path)
        taxonomy = self.load()
        if not self._loaded_ok:
            raise ValidationError(
                "Taxonomy reload failed; the previous definition is still active.",
                details={"error": self._load_error or "unknown"},
            )
        return taxonomy

    @staticmethod
    def _validate(taxonomy: Taxonomy) -> None:
        seen: set[str] = set()
        for theme in taxonomy.themes:
            key = theme.name.strip().lower()
            if key in seen:
                raise ValueError(f"Duplicate theme name: {theme.name}")
            seen.add(key)

            issue_seen: set[str] = set()
            for issue in theme.issues:
                ikey = issue.name.strip().lower()
                if ikey in issue_seen:
                    raise ValueError(
                        f"Duplicate issue '{issue.name}' inside theme '{theme.name}'"
                    )
                issue_seen.add(ikey)

        # fallback_theme is deliberately allowed to be OUTSIDE the themes list -
        # it is a label for system-generated rows (no speech detected, or a
        # defensive catch if the model's answer cannot be matched at all), not
        # a choice the model itself is ever offered. See classification.yaml.
        if taxonomy.fallback_theme and any(
            t.name.strip().lower() == taxonomy.fallback_theme.strip().lower()
            for t in taxonomy.themes
        ):
            logger.warning(
                "fallback_theme '%s' matches a real theme name - the model "
                "may pick it directly during normal classification, which is "
                "probably not intended for a fallback/system label.",
                taxonomy.fallback_theme,
            )

    # ---- accessors ---------------------------------------------------------
    @property
    def taxonomy(self) -> Taxonomy:
        return self._taxonomy

    @property
    def is_loaded(self) -> bool:
        return self._loaded_ok

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def source_path(self) -> Path:
        return self._path
