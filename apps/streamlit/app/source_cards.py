"""Pure helpers for Streamlit source-card summary and media rendering."""

from __future__ import annotations

from html import escape
from urllib.parse import urlparse, parse_qs
import re

_EMOJI_OR_SYMBOL_PATTERN = re.compile(
    r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\u200D\uFE0E\uFE0F]"
)
_SOURCE_SUMMARY_PREVIEW_CHARS = 200
_SOURCE_SUMMARY_FULL_CHARS = 3200
_SUMMARY_LEADING_PATTERNS = (
    re.compile(
        r"^(?:this|the)\s+(?:article|paper|source|report|post|study)\s+"
        r"(?:discusses|explores|describes|covers|focuses on|is about)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:in\s+this\s+(?:article|paper|source)|according\s+to\s+the\s+(?:article|paper|source)),?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r'^(?:the\s+)?(?:article|paper|source|report)\s+"[^"]+"\s+'
        r"(?:discusses|explores|describes|covers|focuses on|is about)\s+",
        re.IGNORECASE,
    ),
)


def _strip_emoji_symbols(text: object) -> str:
    """Remove emoji-like symbols so unsupported fonts do not render tofu boxes."""
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = _EMOJI_OR_SYMBOL_PATTERN.sub("", raw)
    return " ".join(cleaned.split())


def _clean_summary_style(text: str) -> str:
    """Trim formulaic summary openers so the card copy reads directly."""
    cleaned = str(text or "").strip().strip('"')
    for pattern in _SUMMARY_LEADING_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    cleaned = cleaned.lstrip(",:;- ").strip()
    if not cleaned:
        return ""
    if cleaned[0].islower():
        cleaned = f"{cleaned[0].upper()}{cleaned[1:]}"
    return cleaned


def normalize_source_summary(raw: object) -> str:
    """Normalize source summary text into a bounded single-space string."""
    text = _strip_emoji_symbols(raw)
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    normalized = _clean_summary_style(text)
    if not normalized:
        normalized = text
    return normalized[:_SOURCE_SUMMARY_FULL_CHARS].strip()


def _truncate_summary(text: str, max_chars: int) -> str:
    """Return summary truncated to max_chars with ellipsis when needed."""
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_chars:
        return clean
    return f"{clean[:max_chars].rstrip()}..."


def build_source_card_summaries(raw_summary: object) -> tuple[str, str]:
    """Return the front preview copy and full back-of-card summary copy."""
    full = normalize_source_summary(raw_summary)
    if not full:
        return "", ""
    return _truncate_summary(full, _SOURCE_SUMMARY_PREVIEW_CHARS), full


def should_show_source_summary_button(raw_summary: object) -> bool:
    """Return whether the source card should expose the summary flip action."""
    return bool(normalize_source_summary(raw_summary))


def _source_domain(url: str) -> str:
    """Extract normalized domain from URL for visual fallback media."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        return parsed.netloc.removeprefix("www.")
    except Exception:
        return ""


def _youtube_embed_url(url: object) -> str:
    """Return an embeddable YouTube URL when the input points to a YouTube page."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    host = parsed.netloc.lower().removeprefix("www.")
    video_id = ""
    if host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]
    elif host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    if not video_id:
        return ""
    return f"https://www.youtube.com/embed/{escape(video_id)}"


def _looks_like_direct_video(url: object) -> bool:
    """Return True when the URL looks like a directly playable video asset."""
    raw = str(url or "").strip().lower()
    if not raw:
        return False
    if any(token in raw for token in (".mp4", ".mov", ".m4v", ".webm", ".m3u8")):
        return True
    return any(
        token in raw
        for token in (
            "video.twimg.com",
            "play-",
            "tiktokcdn",
        )
    )


def build_source_card_media(
    *,
    source: object,
    url: object,
    cover_url: object,
    video_url: object,
    source_label: str,
) -> str:
    """Return source-card media HTML using platform-aware fallbacks."""
    source_key = str(source or "").strip().lower()
    link_url = str(url or "").strip()
    cover = str(cover_url or "").strip()
    video = str(video_url or "").strip()

    youtube_embed = _youtube_embed_url(video or link_url)
    if youtube_embed:
        return (
            '<div class="discover-media-wrap">'
            f'<iframe class="discover-media-embed" src="{youtube_embed}" '
            'title="YouTube video preview" loading="lazy" '
            'referrerpolicy="strict-origin-when-cross-origin" '
            'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
            "allowfullscreen></iframe></div>"
        )

    if video and _looks_like_direct_video(video):
        poster_attr = f' poster="{escape(cover)}"' if cover else ""
        return (
            '<div class="discover-media-wrap">'
            f'<video autoplay muted loop playsinline preload="metadata"{poster_attr} '
            'style="width:100%;height:228px;object-fit:cover;background:#000;">'
            f'<source src="{escape(video)}" type="video/mp4"></video></div>'
        )

    if cover:
        return (
            '<div class="discover-media-wrap">'
            f'<img class="discover-media-image" src="{escape(cover)}" />'
            "</div>"
        )

    if video and source_key in {"social_instagram", "social_tiktok"}:
        return (
            '<div class="discover-media-wrap discover-media-fallback">'
            f'<div class="discover-media-domain">{escape(source_label)} video</div>'
            "</div>"
        )

    if link_url:
        domain = _source_domain(link_url)
        favicon_url = (
            f"https://www.google.com/s2/favicons?domain={escape(domain)}&sz=128"
            if domain
            else ""
        )
        return (
            '<div class="discover-media-wrap discover-media-fallback">'
            f'<img class="discover-media-favicon" src="{favicon_url}" alt="{escape(domain or source_label)} logo" />'
            f'<div class="discover-media-domain">{escape(domain or source_label)}</div>'
            "</div>"
        )

    return '<div class="discover-media-wrap discover-media-empty">No media</div>'
