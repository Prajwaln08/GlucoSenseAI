"""
Jinja2 templating + small shared helpers for the server-rendered patient UI.
"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _fmt_dt(value, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Jinja filter: format a datetime, tolerant of None / strings."""
    if value is None:
        return "—"
    try:
        return value.strftime(fmt)
    except AttributeError:
        return str(value)


def _round(value, ndigits: int = 1):
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


templates.env.filters["fmt_dt"] = _fmt_dt
templates.env.filters["round1"] = _round
