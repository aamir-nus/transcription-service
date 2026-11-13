"""output_format_adapters.py — render/save a transcription script in a chosen format.

A *script* is the list of segment dicts produced by ``transcribe.run_pipeline``::

    [{"time": {"start": float, "end": float}, "speaker": str | None, "text": str}, ...]

A ``speaker`` of ``None`` (``--no-diarize`` runs) omits the speaker label.

Supported formats
-----------------
    - ``md``  : Markdown transcript                 (default)
    - ``vtt`` : WebVTT with ``<v SPEAKER>`` voice tags

Format resolution in :func:`save`
---------------------------------
    explicit ``fmt``  >  path suffix  >  :data:`DEFAULT_FORMAT` (``md``)

Usage
-----
    from output_format_adapters import save, render

    save(script, "Output/meeting.md")        # format inferred from suffix -> md
    save(script, "Output/meeting.vtt")       # -> vtt
    save(script, "Output/meeting")           # no suffix -> default md
    save(script, "Output/meeting", fmt="vtt")# explicit format overrides
    text = render(script, "vtt")             # just the rendered string
"""

from pathlib import Path
from typing import Optional

SUPPORTED_FORMATS = ("md", "vtt")
DEFAULT_FORMAT = "md"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _format_timestamp(seconds: float) -> str:
    """Seconds (float) -> ``HH:MM:SS.mmm`` (WebVTT style, always 3-digit millis)."""
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    total, ms = divmod(millis, 1000)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_from_suffix(path: Path) -> Optional[str]:
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix in SUPPORTED_FORMATS else None


def _normalize_format(fmt: Optional[str]) -> str:
    """Lower-case, strip a leading dot, and validate against supported formats."""
    if not fmt:
        return DEFAULT_FORMAT
    cleaned = fmt.lower().lstrip(".")
    if cleaned not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported output format {fmt!r}. Supported: {SUPPORTED_FORMATS}"
        )
    return cleaned


def _ensure_extension(path: Path, fmt: str) -> Path:
    """Append ``.<fmt>`` when ``path`` has no suffix; otherwise respect the name."""
    if path.suffix:
        return path
    return path.with_name(f"{path.name}.{fmt}")


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _render_md(script: list[dict]) -> str:
    lines: list[str] = ["# Transcript", ""]
    for seg in script:
        speaker = seg.get("speaker")
        start = _format_timestamp(seg["time"]["start"])
        end = _format_timestamp(seg["time"]["end"])
        text = (seg.get("text") or "").strip()
        label = f" {speaker}" if speaker else ""
        lines.append(f"**[{start} → {end}]{label}**")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_vtt(script: list[dict]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for idx, seg in enumerate(script, start=1):
        speaker = seg.get("speaker")
        start = _format_timestamp(seg["time"]["start"])
        end = _format_timestamp(seg["time"]["end"])
        text = (seg.get("text") or "").strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(f"<v {speaker}>{text}" if speaker else text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render(script: list[dict], fmt: str = DEFAULT_FORMAT) -> str:
    """Return ``script`` serialized as a string in ``fmt``."""
    fmt = _normalize_format(fmt)
    if fmt == "md":
        return _render_md(script)
    if fmt == "vtt":
        return _render_vtt(script)
    # Unreachable: _normalize_format guards this.
    raise ValueError(f"unsupported output format {fmt!r}")


def save(
    script: list[dict],
    path: str | Path,
    fmt: Optional[str] = None,
) -> Path:
    """Render ``script`` and write it to ``path``.

    The format is resolved as: explicit ``fmt``  >  ``path`` suffix  >  md.
    If ``path`` has no suffix, the chosen format's extension is appended.
    Parent directories are created as needed. Returns the resolved path.
    """
    if fmt:
        out_format = _normalize_format(fmt)
    else:
        out_format = _format_from_suffix(Path(path)) or DEFAULT_FORMAT

    out_path = _ensure_extension(Path(path), out_format)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(script, out_format), encoding="utf-8")
    return out_path
