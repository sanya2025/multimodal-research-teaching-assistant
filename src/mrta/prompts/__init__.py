"""mrta.prompts — Jinja2 template loader."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader

_env = Environment(loader=PackageLoader("mrta", "prompts"))

MODES: dict[str, str] = {
    "default": "_base",
    "beginner": "beginner",
    "expert": "expert",
    "interview": "interview",
    "quiz": "quiz",
    "lecture_notes": "lecture_notes",
    "explain": "explain",
}


def load_prompt(name: str, **kwargs: object) -> str:
    """Render src/mrta/prompts/{name}.j2 with kwargs."""
    return _env.get_template(f"{name}.j2").render(**kwargs)


__all__ = ["load_prompt", "MODES"]
