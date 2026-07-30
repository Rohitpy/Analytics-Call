"""YAML loading that works with either PyYAML or ruamel.yaml.

The target environment ships ruamel.yaml but not necessarily PyYAML, and the
taxonomy/prompt files are far more pleasant to maintain as YAML than JSON.
This picks whichever library is present and raises a clear error if neither
is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_BACKEND: str | None = None
_loader: Any = None

try:  # PyYAML first - it is the more common of the two
    import yaml as _pyyaml

    _BACKEND = "pyyaml"
    _loader = _pyyaml
except ImportError:  # pragma: no cover - environment dependent
    try:
        from ruamel.yaml import YAML as _RuamelYAML

        _BACKEND = "ruamel"
        _loader = _RuamelYAML(typ="safe", pure=True)
    except ImportError:
        _BACKEND = None
        _loader = None


class YamlUnavailableError(RuntimeError):
    pass


def backend_name() -> str | None:
    """Which YAML library is in use ("pyyaml", "ruamel", or None)."""
    return _BACKEND


def safe_load(text: str) -> Any:
    if _BACKEND == "pyyaml":
        return _loader.safe_load(text)
    if _BACKEND == "ruamel":
        return _loader.load(text)
    raise YamlUnavailableError(
        "Neither PyYAML nor ruamel.yaml is installed - cannot read YAML config. "
        "Install one with: pip install PyYAML"
    )


def load_file(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    return safe_load(path.read_text(encoding="utf-8"))
