"""
Beauty Social AI - Project-first Streamlit UI.

Flow:
1. Land on Home.
2. Create or select a project.
3. Redirect into project Chat.

Discover and project management remain available once a project is selected.
"""

import importlib.util
import re
import sys
import uuid
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo
from collections.abc import Generator

import streamlit as st
from agent import get_database
from source_cards import (
    build_source_card_media,
    build_source_card_summaries,
    should_show_source_summary_button,
)

IST_TZ = ZoneInfo("Asia/Kolkata")
_WIZARD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wizard-ui")
_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="upload-ui")
_WIZARD_PROGRESS_TIMEOUT_SECONDS = 25.0
_EMOJI_OR_SYMBOL_PATTERN = re.compile(
    r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\u200D\uFE0E\uFE0F]"
)


def _strip_emoji_symbols(text: object) -> str:
    """Remove emoji-like symbols so unsupported fonts do not render tofu boxes."""
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = _EMOJI_OR_SYMBOL_PATTERN.sub("", raw)
    return " ".join(cleaned.split())


def _load_local_api_module() -> ModuleType:
    """Load the local `_api.py` file explicitly to avoid module-resolution ambiguity."""
    module_path = Path(__file__).with_name("_api.py")
    spec = importlib.util.spec_from_file_location("googly_streamlit_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load API client module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    import _api as _api_module
except Exception:
    _api_module = _load_local_api_module()

if not hasattr(_api_module, "get_insights"):
    _api_module = _load_local_api_module()

_api = _api_module

# ============ THEME CONFIG ============
THEME_PRESETS: dict[str, dict[str, str]] = {
    "Rose Glow": {
        "bg": "#fdf4ef",
        "panel": "rgba(255, 248, 244, 0.76)",
        "panel_soft": "rgba(255, 244, 240, 0.68)",
        "bg_radial_1": "rgba(244, 177, 160, 0.42)",
        "bg_radial_2": "rgba(186, 233, 223, 0.42)",
        "bg_radial_3": "rgba(247, 208, 222, 0.34)",
        "bg_grad_start": "#fef6f1",
        "bg_grad_mid": "#fcf3f7",
        "bg_grad_end": "#f9f5fb",
        "sidebar_grad_1": "rgba(255, 236, 224, 0.92)",
        "sidebar_grad_2": "rgba(241, 255, 251, 0.88)",
        "sidebar_grad_3": "rgba(253, 247, 255, 0.92)",
        "button_grad_1": "rgba(250, 197, 179, 0.68)",
        "button_grad_2": "rgba(196, 240, 231, 0.62)",
        "accent": "#f26d62",
        "accent_2": "#f6b26b",
        "accent_3": "#2fb9aa",
        "line": "rgba(242, 109, 98, 0.28)",
        "text": "#2f2632",
        "muted": "#756878",
        "dark_bg": "#141826",
        "dark_panel": "rgba(31, 36, 54, 0.9)",
        "dark_panel_soft": "rgba(35, 41, 60, 0.84)",
        "dark_bg_radial_1": "rgba(201, 129, 120, 0.26)",
        "dark_bg_radial_2": "rgba(92, 165, 151, 0.24)",
        "dark_bg_radial_3": "rgba(145, 118, 176, 0.22)",
        "dark_bg_grad_start": "#131726",
        "dark_bg_grad_mid": "#1a1f31",
        "dark_bg_grad_end": "#1d2236",
        "dark_sidebar_grad_1": "rgba(36, 43, 62, 0.95)",
        "dark_sidebar_grad_2": "rgba(32, 40, 58, 0.95)",
        "dark_sidebar_grad_3": "rgba(27, 34, 50, 0.96)",
        "dark_button_grad_1": "rgba(177, 117, 108, 0.34)",
        "dark_button_grad_2": "rgba(85, 155, 142, 0.32)",
        "dark_accent": "#f09986",
        "dark_accent_2": "#efc082",
        "dark_accent_3": "#6bd0c3",
        "dark_line": "rgba(240, 153, 134, 0.3)",
        "dark_text": "#f6f3fb",
        "dark_muted": "#c7c1d9",
    },
    "Fresh Mint": {
        "bg": "#f3fbf8",
        "panel": "rgba(244, 255, 251, 0.76)",
        "panel_soft": "rgba(238, 252, 247, 0.7)",
        "bg_radial_1": "rgba(176, 236, 222, 0.42)",
        "bg_radial_2": "rgba(198, 229, 255, 0.36)",
        "bg_radial_3": "rgba(226, 248, 222, 0.36)",
        "bg_grad_start": "#f7fffd",
        "bg_grad_mid": "#f1fbff",
        "bg_grad_end": "#f4fff7",
        "sidebar_grad_1": "rgba(230, 252, 246, 0.92)",
        "sidebar_grad_2": "rgba(239, 250, 255, 0.9)",
        "sidebar_grad_3": "rgba(236, 255, 244, 0.92)",
        "button_grad_1": "rgba(178, 236, 222, 0.74)",
        "button_grad_2": "rgba(198, 229, 255, 0.62)",
        "accent": "#249f90",
        "accent_2": "#57b58f",
        "accent_3": "#2f8fb9",
        "line": "rgba(36, 159, 144, 0.24)",
        "text": "#2f2632",
        "muted": "#756878",
        "dark_bg": "#121b20",
        "dark_panel": "rgba(26, 40, 46, 0.9)",
        "dark_panel_soft": "rgba(29, 44, 52, 0.84)",
        "dark_bg_radial_1": "rgba(99, 179, 163, 0.28)",
        "dark_bg_radial_2": "rgba(88, 138, 183, 0.24)",
        "dark_bg_radial_3": "rgba(129, 175, 135, 0.24)",
        "dark_bg_grad_start": "#121a1f",
        "dark_bg_grad_mid": "#172329",
        "dark_bg_grad_end": "#1a2830",
        "dark_sidebar_grad_1": "rgba(31, 49, 55, 0.95)",
        "dark_sidebar_grad_2": "rgba(29, 46, 54, 0.95)",
        "dark_sidebar_grad_3": "rgba(24, 40, 48, 0.96)",
        "dark_button_grad_1": "rgba(89, 166, 151, 0.36)",
        "dark_button_grad_2": "rgba(95, 148, 190, 0.32)",
        "dark_accent": "#6ad2c4",
        "dark_accent_2": "#89d9af",
        "dark_accent_3": "#66b6d9",
        "dark_line": "rgba(106, 210, 196, 0.28)",
        "dark_text": "#edf8f6",
        "dark_muted": "#bbd1cb",
    },
}


def _apply_theme(theme_name: str) -> None:
    """Apply selected palette by overriding CSS variables."""
    preset = THEME_PRESETS.get(theme_name, THEME_PRESETS["Rose Glow"])
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {preset["bg"]};
            --panel: {preset["panel"]};
            --panel-soft: {preset["panel_soft"]};
            --bg-radial-1: {preset["bg_radial_1"]};
            --bg-radial-2: {preset["bg_radial_2"]};
            --bg-radial-3: {preset["bg_radial_3"]};
            --bg-grad-start: {preset["bg_grad_start"]};
            --bg-grad-mid: {preset["bg_grad_mid"]};
            --bg-grad-end: {preset["bg_grad_end"]};
            --sidebar-grad-1: {preset["sidebar_grad_1"]};
            --sidebar-grad-2: {preset["sidebar_grad_2"]};
            --sidebar-grad-3: {preset["sidebar_grad_3"]};
            --button-grad-1: {preset["button_grad_1"]};
            --button-grad-2: {preset["button_grad_2"]};
            --accent: {preset["accent"]};
            --accent-2: {preset["accent_2"]};
            --accent-3: {preset["accent_3"]};
            --line: {preset["line"]};
            --text: {preset["text"]};
            --muted: {preset["muted"]};
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: {preset["dark_bg"]};
                --panel: {preset["dark_panel"]};
                --panel-soft: {preset["dark_panel_soft"]};
                --bg-radial-1: {preset["dark_bg_radial_1"]};
                --bg-radial-2: {preset["dark_bg_radial_2"]};
                --bg-radial-3: {preset["dark_bg_radial_3"]};
                --bg-grad-start: {preset["dark_bg_grad_start"]};
                --bg-grad-mid: {preset["dark_bg_grad_mid"]};
                --bg-grad-end: {preset["dark_bg_grad_end"]};
                --sidebar-grad-1: {preset["dark_sidebar_grad_1"]};
                --sidebar-grad-2: {preset["dark_sidebar_grad_2"]};
                --sidebar-grad-3: {preset["dark_sidebar_grad_3"]};
                --button-grad-1: {preset["dark_button_grad_1"]};
                --button-grad-2: {preset["dark_button_grad_2"]};
                --accent: {preset["dark_accent"]};
                --accent-2: {preset["dark_accent_2"]};
                --accent-3: {preset["dark_accent_3"]};
                --line: {preset["dark_line"]};
                --text: {preset["dark_text"]};
                --muted: {preset["dark_muted"]};
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Beauty Social AI",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ============ SESSION STATE INITIALIZATION ============
query_params = st.query_params

if "user_id" not in st.session_state:
    import os

    st.session_state.user_id = os.getenv("STREAMLIT_USER_ID", "default_user")

if "session_id" not in st.session_state:
    if "sid" in query_params:
        st.session_state.session_id = query_params["sid"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params["sid"] = st.session_state.session_id

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Project state
if "projects" not in st.session_state:
    st.session_state.projects = []
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = None
if "project_messages" not in st.session_state:
    st.session_state.project_messages = {}  # project_id -> list[dict]
if "projects_loaded" not in st.session_state:
    st.session_state.projects_loaded = False
if "project_history_loaded" not in st.session_state:
    st.session_state.project_history_loaded = set()
if "show_sources_drawer" not in st.session_state:
    st.session_state.show_sources_drawer = False
if "drawer_insight_id" not in st.session_state:
    st.session_state.drawer_insight_id = None
if "selected_insight_id" not in st.session_state:
    st.session_state.selected_insight_id = None
if "insight_report_cache" not in st.session_state:
    st.session_state.insight_report_cache = {}
if "insight_report_requested" not in st.session_state:
    st.session_state.insight_report_requested = set()

_apply_theme("Rose Glow")


# ============ HELPER FUNCTIONS ============


def reload_projects():
    """Fetch projects from FastAPI and update session state."""
    st.session_state.projects = _api.list_projects()
    st.session_state.projects_loaded = True


def _selected_project() -> dict | None:
    """Return the currently selected project from local session cache."""
    project_id = st.session_state.selected_project_id
    if not project_id:
        return None
    for project in st.session_state.projects:
        if project.get("id") == project_id:
            return project
    return None


def _upload_bootstrap_files(
    project_id: str, files: list | None
) -> tuple[list[str], list[str]]:
    """Upload bootstrap files in parallel and return (successful_upload_ids, failed_names)."""
    payloads: list[tuple[str, bytes]] = []
    for file in files or []:
        name = str(getattr(file, "name", "") or "").strip()
        if not name:
            continue
        content = file.read()
        if not isinstance(content, bytes) or not content:
            continue
        payloads.append((name, content))

    if not payloads:
        return [], []

    futures: list[tuple[str, Future]] = [
        (name, _UPLOAD_EXECUTOR.submit(_api.upload_document, project_id, name, content))
        for name, content in payloads
    ]
    upload_ids: list[str] = []
    failed_names: list[str] = []

    for name, future in futures:
        try:
            result = future.result()
        except Exception:
            result = None
        if result and result.get("upload_id"):
            upload_ids.append(str(result["upload_id"]))
        else:
            failed_names.append(name)

    return upload_ids, failed_names


def _clear_source_card_flip_state(insight_id: str | None = None) -> None:
    """Clear source-card flip state keys for one insight or globally."""
    prefix = "source_card_state_flip_"
    keys_to_delete: list[str] = []
    marker = f"{insight_id}_" if insight_id else ""
    for key in list(st.session_state.keys()):
        if not key.startswith(prefix):
            continue
        if marker and not key.startswith(f"{prefix}{marker}"):
            continue
        keys_to_delete.append(key)
    for key in keys_to_delete:
        del st.session_state[key]


def _reset_insight_view_state() -> None:
    """Clear insight-detail and sources-drawer state."""
    _clear_source_card_flip_state()
    st.session_state.selected_insight_id = None
    st.session_state.drawer_insight_id = None
    st.session_state.show_sources_drawer = False
    st.session_state.insight_report_requested = set()


def _select_project(project_id: str, target_page: str) -> None:
    """Switch active project and clear project-scoped UI state when project changes."""
    previous_project_id = str(st.session_state.selected_project_id or "")
    next_project_id = str(project_id or "")
    if previous_project_id and previous_project_id != next_project_id:
        _reset_insight_view_state()
    st.session_state.selected_project_id = project_id
    st.session_state.current_page = target_page


def _sync_project_chunk_count_from_ingest(
    project_id: str, status_payload: dict | None
) -> None:
    """Sync local project list chunk counters from latest ingest status payload."""
    if not project_id or not isinstance(status_payload, dict):
        return

    total_chunks_raw = status_payload.get("total_chunks")
    try:
        total_chunks = int(total_chunks_raw)
    except (TypeError, ValueError):
        return

    if total_chunks < 0:
        return

    refreshed_at = status_payload.get("finished_at") or status_payload.get("updated_at")
    for project in st.session_state.projects:
        if project.get("id") != project_id:
            continue
        current_chunks = int(project.get("kb_chunk_count") or 0)
        if current_chunks == total_chunks:
            return
        project["kb_chunk_count"] = total_chunks
        if refreshed_at:
            project["last_refreshed_at"] = refreshed_at
        return


def _project_chat_session_id(project_id: str) -> str:
    """Stable session ID scoped to user + project for the SSE chat endpoint."""
    return f"streamlit_{st.session_state.user_id}_{project_id}"


def _parse_to_ist(value: str | None) -> datetime | None:
    """Parse an ISO datetime-like string and convert it to IST."""
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST_TZ)


def _short_date(value: str | None) -> str:
    """Return YYYY-MM-DD in IST from an ISO datetime-like string."""
    dt = _parse_to_ist(value)
    if not dt:
        return "-"
    return dt.strftime("%Y-%m-%d")


def _short_datetime_ist(value: str | None) -> str:
    """Return YYYY-MM-DD HH:MM IST from an ISO datetime-like string."""
    dt = _parse_to_ist(value)
    if not dt:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M IST")


def _wizard_state_key(form_key: str) -> str:
    return f"{form_key}_wizard_state"


def _wizard_async_state_key(form_key: str) -> str:
    return f"{form_key}_wizard_async_refresh"


def _wizard_run_refresh_job(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict],
    max_questions: int,
    source_toggles: dict[str, bool],
) -> dict:
    """Run evaluate (and synthesize when complete) off the main Streamlit thread."""
    evaluate_payload, evaluate_error = _api.wizard_evaluate(
        title,
        description,
        qa_pairs=qa_pairs,
        max_questions=max_questions,
    )
    if evaluate_error or not evaluate_payload:
        return {
            "error": evaluate_error or "Could not evaluate wizard state.",
            "evaluate_payload": None,
            "review_payload": None,
        }

    review_payload = None
    if evaluate_payload.get("should_stop"):
        review_payload, synth_error = _api.wizard_synthesize(
            title=title,
            description=description,
            qa_pairs=qa_pairs,
            structured_intent={},
            source_toggles=source_toggles,
        )
        if synth_error or not review_payload:
            return {
                "error": synth_error or "Could not synthesize review.",
                "evaluate_payload": evaluate_payload,
                "review_payload": None,
            }

    return {
        "error": None,
        "evaluate_payload": evaluate_payload,
        "review_payload": review_payload,
    }


def _start_wizard_async_refresh(
    *,
    form_key: str,
    title: str,
    description: str,
    qa_pairs: list[dict],
    max_questions: int,
    source_toggles: dict[str, bool],
    action_label: str,
) -> None:
    """Submit wizard refresh work to background thread and store task handle."""
    future = _WIZARD_EXECUTOR.submit(
        _wizard_run_refresh_job,
        title=title,
        description=description,
        qa_pairs=list(qa_pairs),
        max_questions=max_questions,
        source_toggles=dict(source_toggles),
    )
    st.session_state[_wizard_async_state_key(form_key)] = {
        "future": future,
        "started_at": time.time(),
        "cancelled": False,
        "action_label": action_label,
    }


def _consume_wizard_async_refresh(form_key: str, state: dict) -> bool:
    """
    Render in-flight async status and apply completed results.

    Returns True when async flow handled the current render path.
    """
    task_key = _wizard_async_state_key(form_key)
    task = st.session_state.get(task_key)
    if not isinstance(task, dict):
        return False

    future = task.get("future")
    if not isinstance(future, Future):
        st.session_state.pop(task_key, None)
        return False

    if not future.done():
        elapsed = max(0.0, time.time() - float(task.get("started_at") or time.time()))
        progress = min(0.95, elapsed / _WIZARD_PROGRESS_TIMEOUT_SECONDS)
        action_label = str(task.get("action_label") or "Processing wizard request")
        st.markdown(
            f"""
            <div class="wizard-phase-banner">
                <span class="wizard-phase-pill">Working</span>
                <strong>{escape(action_label)}</strong>
                <small>You can cancel this request and continue editing.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(progress, text=f"In progress… {elapsed:.1f}s")
        if st.button(
            "Cancel request",
            key=f"{form_key}_wizard_cancel_async",
            use_container_width=True,
        ):
            task["cancelled"] = True
            future.cancel()
            st.session_state.pop(task_key, None)
            st.warning("Cancelled current wizard request.")
            st.rerun()
        time.sleep(0.15)
        st.rerun()
        return True

    st.session_state.pop(task_key, None)
    if bool(task.get("cancelled", False)):
        return False

    try:
        result = future.result()
    except Exception as exc:
        st.error(f"Wizard request failed: {exc}")
        return False

    if not isinstance(result, dict):
        st.error("Wizard request returned an invalid response.")
        return False

    if result.get("error"):
        st.error(str(result["error"]))
        return False

    evaluate_payload = result.get("evaluate_payload") or {}
    state["scores"] = evaluate_payload.get("scores", {})
    state["pending_question"] = evaluate_payload.get("next_question") or ""
    state["pending_dimension"] = evaluate_payload.get("next_dimension") or ""

    if evaluate_payload.get("should_stop"):
        review_payload = result.get("review_payload")
        if not isinstance(review_payload, dict):
            st.error("Wizard finished but review payload was missing.")
            return False
        state["review_payload"] = review_payload
        state["source_toggles"] = review_payload.get("target_sources") or state.get(
            "source_toggles", {}
        )
        state["phase"] = "review"
    else:
        state["phase"] = "qa"

    st.rerun()
    return True


def _wizard_default_sources(defaults: dict | None = None) -> dict[str, bool]:
    cfg = defaults or {}
    return {
        "tiktok": bool(cfg.get("tiktok_enabled", True)),
        "instagram": bool(cfg.get("instagram_enabled", True)),
        "youtube": bool(cfg.get("youtube_enabled", True)),
        "reddit": bool(cfg.get("reddit_enabled", True)),
        "x": bool(cfg.get("x_enabled", True)),
        "papers": bool(cfg.get("papers_enabled", True)),
        "patents": bool(cfg.get("patents_enabled", True)),
        "news": bool(cfg.get("perigon_enabled", True)),
        "web_tavily": bool(cfg.get("tavily_enabled", True)),
        "web_exa": bool(cfg.get("exa_enabled", True)),
    }


def _reset_wizard_state(form_key: str, defaults: dict | None = None) -> None:
    state_key = _wizard_state_key(form_key)
    if state_key in st.session_state:
        del st.session_state[state_key]
    async_key = _wizard_async_state_key(form_key)
    if async_key in st.session_state:
        del st.session_state[async_key]

    prefixes = (
        f"{form_key}_wizard_title",
        f"{form_key}_wizard_description",
        f"{form_key}_wizard_answer_",
        f"{form_key}_wizard_domain_focus",
        f"{form_key}_wizard_entities",
        f"{form_key}_wizard_must_terms",
        f"{form_key}_wizard_time_horizon",
        f"{form_key}_wizard_refresh_strategy",
        f"{form_key}_wizard_enriched_description",
        f"{form_key}_wizard_edit_answer_",
        f"{form_key}_wizard_src_",
    )
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]

    cfg = defaults or {}
    st.session_state[state_key] = {
        "phase": "intro",
        "title": str(cfg.get("title", "")),
        "description": str(cfg.get("description", "")),
        "refresh_strategy": str(cfg.get("refresh_strategy", "once")),
        "source_toggles": _wizard_default_sources(cfg),
        "qa_pairs": [],
        "pending_question": "",
        "pending_dimension": "",
        "scores": {},
        "max_questions": 5,
        "review_payload": {},
    }


def _ensure_wizard_state(form_key: str, defaults: dict | None = None) -> dict:
    state_key = _wizard_state_key(form_key)
    if state_key not in st.session_state:
        _reset_wizard_state(form_key, defaults)
    return st.session_state[state_key]


def _ensure_review_widget_defaults(form_key: str, state: dict) -> None:
    payload = state.get("review_payload") or {}
    source_toggles = payload.get("target_sources") or state.get("source_toggles") or {}

    defaults = {
        f"{form_key}_wizard_domain_focus": str(payload.get("domain_focus") or ""),
        f"{form_key}_wizard_entities": list(payload.get("key_entities") or []),
        f"{form_key}_wizard_must_terms": list(payload.get("must_match_terms") or []),
        f"{form_key}_wizard_time_horizon": str(
            payload.get("time_horizon") or "last 1 year"
        ),
        f"{form_key}_wizard_refresh_strategy": str(
            state.get("refresh_strategy") or "once"
        ),
        f"{form_key}_wizard_enriched_description": str(
            payload.get("enriched_description") or ""
        ),
        f"{form_key}_wizard_src_tiktok": bool(source_toggles.get("tiktok", True)),
        f"{form_key}_wizard_src_instagram": bool(source_toggles.get("instagram", True)),
        f"{form_key}_wizard_src_youtube": bool(source_toggles.get("youtube", True)),
        f"{form_key}_wizard_src_reddit": bool(source_toggles.get("reddit", True)),
        f"{form_key}_wizard_src_x": bool(source_toggles.get("x", True)),
        f"{form_key}_wizard_src_papers": bool(source_toggles.get("papers", True)),
        f"{form_key}_wizard_src_patents": bool(source_toggles.get("patents", True)),
        f"{form_key}_wizard_src_news": bool(source_toggles.get("news", True)),
        f"{form_key}_wizard_src_web_tavily": bool(
            source_toggles.get("web_tavily", True)
        ),
        f"{form_key}_wizard_src_web_exa": bool(source_toggles.get("web_exa", True)),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _wizard_answer_guidance(dimension: str) -> tuple[str, str]:
    guidance_map: dict[str, tuple[str, str]] = {
        "objective_clarity": (
            "Be specific about the decision this research should support and who will use it.",
            "Example: Decide whether to prioritize niacinamide or ceramide claims for Q3 launch messaging in India.",
        ),
        "pain_point_clarity": (
            "Describe the risk, uncertainty, or bottleneck you need to reduce.",
            "Example: We see engagement but low conversion and need to know which claim themes are causing drop-off.",
        ),
        "output_clarity": (
            "Describe the final artifact format and what actions it should enable.",
            "Example: Provide a ranked opportunity matrix with top 5 claim angles and recommended next experiments.",
        ),
        "domain_specificity": (
            "Name the sources/platforms to prioritize so retrieval focuses on the right evidence pool.",
            "Example: Prioritize TikTok + Instagram for signal discovery, OpenAlex for validation, and patents for whitespace.",
        ),
    }
    return guidance_map.get(
        dimension or "",
        (
            "Give concrete context, constraints, and expected outcome.",
            "Example: Focus on recent evidence, target audience, and decision criteria.",
        ),
    )


def _render_create_project_form(
    *,
    form_key: str,
    submit_label: str = "Create Project",
    defaults: dict | None = None,
) -> None:
    """Render and handle the two-phase project wizard."""
    cfg = defaults or {}
    state = _ensure_wizard_state(form_key, cfg)

    st.markdown(
        """
        <div class="wizard-shell">
            <div class="wizard-shell-title">Project wizard</div>
            <div class="wizard-shell-subtitle">
                Phase 1 clarifies intent with focused questions. Phase 2 lets you review and refine before creation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if _consume_wizard_async_refresh(form_key, state):
        return

    if state.get("phase") == "intro":
        st.markdown(
            """
            <div class="wizard-phase-banner">
                <span class="wizard-phase-pill">Phase 1</span>
                <strong>Set the project frame</strong>
                <small>Start with a clear title and baseline description.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        title = st.text_input(
            "Title",
            max_chars=255,
            key=f"{form_key}_wizard_title",
            value=state.get("title", ""),
        )
        description = st.text_area(
            "Description",
            help="Describe what this project researches (min 10 chars)",
            max_chars=5000,
            key=f"{form_key}_wizard_description",
            value=state.get("description", ""),
        )

        col_start, col_reset = st.columns([3, 1])
        with col_start:
            if st.button(
                "Start Wizard", key=f"{form_key}_wizard_start", use_container_width=True
            ):
                title_value = (title or "").strip()
                description_value = (description or "").strip()
                if not title_value:
                    st.error("Title is required.")
                    return
                if len(description_value) < 10:
                    st.error("Description must be at least 10 characters.")
                    return
                state["title"] = title_value
                state["description"] = description_value
                state["qa_pairs"] = []
                state["scores"] = {}
                state["pending_question"] = ""
                state["pending_dimension"] = ""
                _start_wizard_async_refresh(
                    form_key=form_key,
                    title=title_value,
                    description=description_value,
                    qa_pairs=[],
                    max_questions=int(state.get("max_questions") or 5),
                    source_toggles=state.get("source_toggles") or {},
                    action_label="Evaluating project clarity",
                )
                st.rerun()
        with col_reset:
            if st.button(
                "Reset", key=f"{form_key}_wizard_intro_reset", use_container_width=True
            ):
                _reset_wizard_state(form_key, cfg)
                st.rerun()
        return

    if state.get("phase") == "qa":
        scores = state.get("scores") or {}
        answered_pairs = list(state.get("qa_pairs") or [])
        pending_question = str(state.get("pending_question") or "").strip()
        pending_dimension = str(state.get("pending_dimension") or "").strip()
        max_questions = int(state.get("max_questions") or 5)

        st.markdown(
            f"""
            <div class="wizard-phase-banner">
                <span class="wizard-phase-pill">Phase 1</span>
                <strong>Answer focused questions</strong>
                <small>{len(answered_pairs)} / {max_questions} answered</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress = (
            0.0 if max_questions <= 0 else min(1.0, len(answered_pairs) / max_questions)
        )
        st.progress(
            progress, text=f"Progress: {len(answered_pairs)} of {max_questions}"
        )
        st.markdown(
            f"""
            <div class="wizard-score-grid">
                <div class="wizard-score-card"><span>Objective</span><strong>{scores.get("objective_clarity", 0.0):.2f}</strong></div>
                <div class="wizard-score-card"><span>Pain point</span><strong>{scores.get("pain_point_clarity", 0.0):.2f}</strong></div>
                <div class="wizard-score-card"><span>Output</span><strong>{scores.get("output_clarity", 0.0):.2f}</strong></div>
                <div class="wizard-score-card"><span>Domain</span><strong>{scores.get("domain_specificity", 0.0):.2f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels: list[str] = []
        if pending_question:
            labels.append(f"Current {len(answered_pairs) + 1}")
        labels.extend([f"Answered {idx + 1}" for idx in range(len(answered_pairs))])
        tabs = st.tabs(labels or ["Question 1"])

        if pending_question:
            answer_key = f"{form_key}_wizard_answer_{len(answered_pairs)}"
            with tabs[0]:
                st.markdown(
                    """
                    <div class="wizard-question-banner">
                        <span class="wizard-phase-pill">Current question</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Question:** {pending_question}")
                guidance, example = _wizard_answer_guidance(pending_dimension)
                st.caption(guidance)
                st.caption(example)
                answer = st.text_area(
                    "Your answer",
                    key=answer_key,
                    height=130,
                    placeholder="Add specific context and constraints...",
                )
                col_submit, col_restart = st.columns(2)
                with col_submit:
                    if st.button(
                        "Submit Answer",
                        key=f"{form_key}_wizard_submit_{len(answered_pairs)}",
                        use_container_width=True,
                    ):
                        answer_value = (answer or "").strip()
                        if not answer_value:
                            st.error("Please answer the current question.")
                            return
                        answered_pairs.append(
                            {
                                "question": pending_question,
                                "answer": answer_value,
                                "dimension": pending_dimension,
                            }
                        )
                        state["qa_pairs"] = answered_pairs
                        _start_wizard_async_refresh(
                            form_key=form_key,
                            title=str(state.get("title") or ""),
                            description=str(state.get("description") or ""),
                            qa_pairs=answered_pairs,
                            max_questions=int(state.get("max_questions") or 5),
                            source_toggles=state.get("source_toggles") or {},
                            action_label="Refining wizard context",
                        )
                        st.rerun()
                with col_restart:
                    if st.button(
                        "Start Over",
                        key=f"{form_key}_wizard_qa_reset",
                        use_container_width=True,
                    ):
                        _reset_wizard_state(form_key, cfg)
                        st.rerun()

        answered_offset = 1 if pending_question else 0
        for idx, item in enumerate(answered_pairs):
            with tabs[idx + answered_offset]:
                st.markdown(f"**Question:** {item.get('question', '')}")
                st.caption("You can edit this answer before Phase 2.")
                answer_key = f"{form_key}_wizard_edit_answer_{idx}"
                if answer_key not in st.session_state:
                    st.session_state[answer_key] = str(item.get("answer", ""))
                st.text_area(
                    "Answer",
                    key=answer_key,
                    height=130,
                    label_visibility="collapsed",
                )
                col_save, col_restore = st.columns(2)
                with col_save:
                    if st.button(
                        "Save & Re-evaluate",
                        key=f"{form_key}_wizard_edit_save_{idx}",
                        use_container_width=True,
                    ):
                        edited_answer = str(
                            st.session_state.get(answer_key) or ""
                        ).strip()
                        if not edited_answer:
                            st.error("Answer cannot be empty.")
                            return
                        if edited_answer != str(item.get("answer", "")).strip():
                            updated_pairs = list(answered_pairs)
                            updated_pairs[idx] = {
                                "question": str(item.get("question", "")),
                                "answer": edited_answer,
                                "dimension": str(item.get("dimension", "")),
                            }
                            state["qa_pairs"] = updated_pairs
                            _start_wizard_async_refresh(
                                form_key=form_key,
                                title=str(state.get("title") or ""),
                                description=str(state.get("description") or ""),
                                qa_pairs=updated_pairs,
                                max_questions=int(state.get("max_questions") or 5),
                                source_toggles=state.get("source_toggles") or {},
                                action_label="Re-evaluating edited answers",
                            )
                        st.rerun()
                with col_restore:
                    if st.button(
                        "Restore",
                        key=f"{form_key}_wizard_edit_restore_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state[answer_key] = str(item.get("answer", ""))
                        st.rerun()
        return

    if state.get("phase") != "review":
        _reset_wizard_state(form_key, cfg)
        st.rerun()
        return

    _ensure_review_widget_defaults(form_key, state)
    review_payload = state.get("review_payload") or {}

    st.markdown(
        """
        <div class="wizard-phase-banner">
            <span class="wizard-phase-pill">Phase 2</span>
            <strong>Review and confirm</strong>
            <small>Refine brief, intent terms, and source strategy before creating the project.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Conversation Summary", expanded=False):
        st.markdown(f"**Title:** {state.get('title', '')}")
        st.markdown(f"**Description:** {state.get('description', '')}")
        for idx, item in enumerate(state.get("qa_pairs") or []):
            st.markdown(f"{idx + 1}. **Q:** {item.get('question', '')}")
            st.markdown(f"   **A:** {item.get('answer', '')}")

    with st.container(border=True):
        st.markdown("##### Research Brief")
        st.text_area(
            "Enriched Description",
            key=f"{form_key}_wizard_enriched_description",
            height=240,
            help="Dense semantic summary used for downstream ranking and retrieval.",
        )

    with st.container(border=True):
        st.markdown("##### Intent Controls")
        domain_focus = st.text_input(
            "Domain Focus",
            key=f"{form_key}_wizard_domain_focus",
            max_chars=255,
        )

        key_entity_options = sorted(
            set(
                list(review_payload.get("key_entities") or [])
                + list(review_payload.get("must_match_terms") or [])
            )
        )
        st.multiselect(
            "Key Entities",
            options=key_entity_options,
            key=f"{form_key}_wizard_entities",
            accept_new_options=True,
        )
        st.multiselect(
            "Must-Match Terms",
            options=key_entity_options,
            key=f"{form_key}_wizard_must_terms",
            accept_new_options=True,
        )

        horizon_options = [
            "last 6 months",
            "last 1 year",
            "last 2 years",
            "last 5 years",
            "all-time",
        ]
        current_horizon = st.session_state.get(
            f"{form_key}_wizard_time_horizon", "last 1 year"
        )
        if current_horizon not in horizon_options:
            horizon_options.append(current_horizon)
        st.selectbox(
            "Time Horizon",
            horizon_options,
            key=f"{form_key}_wizard_time_horizon",
        )

        st.selectbox(
            "Refresh Strategy",
            ["once", "daily", "weekly", "on_demand"],
            key=f"{form_key}_wizard_refresh_strategy",
            help="How often to refresh the KB from social and research sources.",
        )

    with st.container(border=True):
        st.markdown("##### Target data sources")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("*Social*")
            st.toggle("TikTok", key=f"{form_key}_wizard_src_tiktok")
            st.toggle("Instagram", key=f"{form_key}_wizard_src_instagram")
            st.toggle("YouTube", key=f"{form_key}_wizard_src_youtube")
            st.toggle("Reddit", key=f"{form_key}_wizard_src_reddit")
            st.toggle("X", key=f"{form_key}_wizard_src_x")
        with col_s2:
            st.markdown("*Research & Discovery*")
            st.toggle("Papers", key=f"{form_key}_wizard_src_papers")
            st.toggle("Patents", key=f"{form_key}_wizard_src_patents")
            st.toggle("News", key=f"{form_key}_wizard_src_news")
            st.toggle("Web (Tavily)", key=f"{form_key}_wizard_src_web_tavily")
            st.toggle("Web (Exa)", key=f"{form_key}_wizard_src_web_exa")

    bootstrap_files = st.file_uploader(
        "Optional bootstrap docs (pdf/docx/txt/md)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"{form_key}_bootstrap_files",
    )

    target_sources = {
        "tiktok": bool(st.session_state.get(f"{form_key}_wizard_src_tiktok", True)),
        "instagram": bool(
            st.session_state.get(f"{form_key}_wizard_src_instagram", True)
        ),
        "youtube": bool(st.session_state.get(f"{form_key}_wizard_src_youtube", True)),
        "reddit": bool(st.session_state.get(f"{form_key}_wizard_src_reddit", True)),
        "x": bool(st.session_state.get(f"{form_key}_wizard_src_x", True)),
        "papers": bool(st.session_state.get(f"{form_key}_wizard_src_papers", True)),
        "patents": bool(st.session_state.get(f"{form_key}_wizard_src_patents", True)),
        "news": bool(st.session_state.get(f"{form_key}_wizard_src_news", True)),
        "web_tavily": bool(
            st.session_state.get(f"{form_key}_wizard_src_web_tavily", True)
        ),
        "web_exa": bool(st.session_state.get(f"{form_key}_wizard_src_web_exa", True)),
    }

    col_create, col_back, col_reset = st.columns([3, 1, 1])
    with col_create:
        if st.button(
            submit_label, key=f"{form_key}_wizard_create", use_container_width=True
        ):
            with st.spinner("Creating project..."):
                result, create_error = _api.wizard_create(
                    title=state.get("title", ""),
                    description=state.get("description", ""),
                    qa_pairs=list(state.get("qa_pairs") or []),
                    refresh_strategy=str(
                        st.session_state.get(
                            f"{form_key}_wizard_refresh_strategy", "once"
                        )
                    ),
                    enriched_description=str(
                        st.session_state.get(
                            f"{form_key}_wizard_enriched_description", ""
                        )
                    ),
                    domain_focus=domain_focus,
                    key_entities=list(
                        st.session_state.get(f"{form_key}_wizard_entities", [])
                    ),
                    must_match_terms=list(
                        st.session_state.get(f"{form_key}_wizard_must_terms", [])
                    ),
                    time_horizon=str(
                        st.session_state.get(
                            f"{form_key}_wizard_time_horizon", "last 1 year"
                        )
                    ),
                    target_sources=target_sources,
                )

            if not result:
                st.error(create_error or "Failed to create project. Please try again.")
                return

            upload_ids, failed_uploads = _upload_bootstrap_files(
                result["id"],
                list(bootstrap_files or []),
            )
            if failed_uploads:
                st.warning(
                    f"{len(failed_uploads)} bootstrap file(s) failed to upload: "
                    + ", ".join(failed_uploads[:3])
                    + ("..." if len(failed_uploads) > 3 else "")
                )

            with st.spinner("Starting project setup..."):
                _api.bootstrap_project(result["id"], upload_ids)

            _select_project(result["id"], "Chat")
            _reset_wizard_state(form_key, cfg)
            reload_projects()
            st.success(f"Project '{result['title']}' created.")
            st.rerun()
    with col_back:
        if st.button(
            "Back", key=f"{form_key}_wizard_back_to_qa", use_container_width=True
        ):
            state["phase"] = "qa"
            st.rerun()
    with col_reset:
        if st.button(
            "Reset", key=f"{form_key}_wizard_review_reset", use_container_width=True
        ):
            _reset_wizard_state(form_key, cfg)
            st.rerun()


# Preload projects once so Home can render without sidebar interaction.
if not st.session_state.projects_loaded:
    reload_projects()

# Hide sidebar on Home for full-screen first impression.
if st.session_state.current_page == "Home":
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: block !important; }
        [data-testid="collapsedControl"] { display: flex !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("Beauty Social")
        st.caption("P&G Beauty Intelligence")

        st.divider()

        # Navigation: Home-first. Chat/Discover unlock only when a project is selected.
        nav_options = ["Home", "Projects"]
        if st.session_state.selected_project_id:
            nav_options.extend(["Chat", "Discover"])
        allowed_pages = set(nav_options) | {"InsightDetail"}
        if st.session_state.current_page not in allowed_pages:
            st.session_state.current_page = "Home"

        nav_page = st.session_state.current_page
        if nav_page == "InsightDetail" and "Discover" in nav_options:
            nav_page = "Discover"

        page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(nav_page),
            label_visibility="collapsed",
        )
        if page != nav_page:
            st.session_state.current_page = page
            if page == "Home":
                reload_projects()
            st.rerun()

        st.divider()

        active_project = _selected_project()
        active_label = _strip_emoji_symbols(
            active_project.get("title") if active_project else ""
        )
        if not active_label:
            active_label = "None selected"

        st.markdown(f"**Active Project:** {active_label}")

        if active_project:
            base_description = str(active_project.get("description") or "").strip()
            enriched_description = str(
                active_project.get("enriched_description") or ""
            ).strip()
            base_html = escape(
                base_description or "No base description available."
            ).replace("\n", "<br>")
            enriched_html = escape(
                enriched_description
                or "No enriched description yet. Complete wizard and synthesis to generate one."
            ).replace("\n", "<br>")
            brief_state = "Ready" if enriched_description else "Pending"
            with st.expander("Project description", expanded=False):
                st.markdown(
                    f"""
                    <div class="sidebar-brief-card">
                        <div class="sidebar-brief-head">
                            <span class="sidebar-brief-pill">{brief_state}</span>
                            <span class="sidebar-brief-meta">Project brief</span>
                        </div>
                        <div class="sidebar-brief-label">Base description</div>
                        <div class="sidebar-brief-text">{base_html}</div>
                        <div class="sidebar-brief-label">Enriched description</div>
                        <div class="sidebar-brief-text sidebar-brief-enriched">{enriched_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if st.button("Refresh Projects", use_container_width=True):
            reload_projects()
            st.rerun()

        if st.session_state.selected_project_id and st.button(
            "Back to Home", use_container_width=True
        ):
            st.session_state.current_page = "Home"
            st.rerun()

        if st.session_state.selected_project_id and st.button(
            "Clear Selection", use_container_width=True
        ):
            st.session_state.selected_project_id = None
            _reset_insight_view_state()
            st.session_state.current_page = "Home"
            st.rerun()

        if st.button("Create Project", use_container_width=True):
            st.session_state.current_page = "Home"
            st.session_state["home_show_create"] = True
            st.rerun()

        st.divider()

        # Database / memory status - checked once at startup only
        if "db_status" not in st.session_state:
            st.session_state.db_status = get_database() is not None
        if st.session_state.db_status:
            st.markdown(
                '<span class="memory-status-pill memory-active">'
                '<span class="memory-status-dot"></span>Memory Active'
                "</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="memory-status-pill memory-offline">'
                '<span class="memory-status-dot"></span>Memory Offline'
                "</span>",
                unsafe_allow_html=True,
            )

# ============ MAIN CONTENT ============

# ----------------------------------------------------------------
# HOME PAGE - default landing with project gallery + quick create
# ----------------------------------------------------------------
if st.session_state.current_page == "Home":
    st.markdown(
        """
        <div class="home-topbar">
            <div class="home-topbar-brand">
                <span class="home-topbar-logo">PG</span>
                <div>
                    <div class="home-topbar-title">Beauty Social Intelligence</div>
                    <div class="home-topbar-subtitle">Project Workspace</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-hero">
            <div class="home-hero-top">
                <div class="home-logo-mark">PG</div>
                <div class="home-hero-eyebrow">P&G Beauty Intelligence</div>
            </div>
            <h1>Build Insights Across Beauty, Cosmetics, and Consumer Signals</h1>
            <p>
                Start from a focused project, ingest social and research evidence,
                then move into chat with a grounded context.
            </p>
            <div class="home-pill-row">
                <span class="home-pill">Skincare</span>
                <span class="home-pill">Cosmetics</span>
                <span class="home-pill">Consumer Signals</span>
                <span class="home-pill">Competitive Research</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    projects = st.session_state.projects
    total_projects = len(projects)
    kb_ready_count = sum(1 for p in projects if (p.get("kb_chunk_count") or 0) > 0)
    total_chunks = sum(int(p.get("kb_chunk_count") or 0) for p in projects)
    papers_projects = sum(1 for p in projects if p.get("papers_enabled", True))
    refresh_values = [
        p.get("last_refreshed_at") for p in projects if p.get("last_refreshed_at")
    ]
    latest_refresh = _short_date(max(refresh_values)) if refresh_values else "-"

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">Projects</div>
                <div class="home-stat-value">{total_projects}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">KB-Ready Projects</div>
                <div class="home-stat-value">{kb_ready_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f"""
            <div class="home-stat">
                <div class="home-stat-label">Total KB Chunks</div>
                <div class="home-stat-value">{total_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="home-signal-strip">
            <div class="home-signal-title">Workspace Snapshot</div>
            <div class="home-signal-chips">
                <span class="home-signal-chip">Latest refresh: {latest_refresh}</span>
                <span class="home-signal-chip">KB Ready: {kb_ready_count}</span>
                <span class="home-signal-chip">Papers enabled: {papers_projects}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""<div class="home-divider"></div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Your Projects")
    with right:
        actions_col1, actions_col2 = st.columns(2)
        with actions_col1:
            if st.button("New Project", use_container_width=True):
                st.session_state["home_show_create"] = True
        with actions_col2:
            if st.button("Refresh", use_container_width=True):
                reload_projects()
                st.rerun()
    if "home_show_create" not in st.session_state:
        st.session_state["home_show_create"] = len(st.session_state.projects) == 0

    if st.session_state["home_show_create"]:
        with st.expander("Create New Project", expanded=True):
            _render_create_project_form(form_key="home_create_project_form")
            if st.button("Close", key="home_create_close"):
                st.session_state["home_show_create"] = False
                st.rerun()

    if not projects:
        st.markdown(
            """
            <div class="home-empty">
                <h3>No projects yet</h3>
                <p>Create your first project to start beauty-focused research and social insight tracking.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        grid_cols = 3 if len(projects) >= 3 else len(projects)
        cols = st.columns(grid_cols)
        for idx, proj in enumerate(projects):
            pid = proj["id"]
            source_tags: list[str] = []
            if proj.get("instagram_enabled", True):
                source_tags.append("Instagram")
            if proj.get("tiktok_enabled", True):
                source_tags.append("TikTok")
            if proj.get("youtube_enabled", True):
                source_tags.append("YouTube")
            if proj.get("reddit_enabled", True):
                source_tags.append("Reddit")
            if proj.get("x_enabled", True):
                source_tags.append("X")
            if proj.get("papers_enabled", True):
                source_tags.append("Papers")
            if proj.get("patents_enabled", True):
                source_tags.append("Patents")
            if proj.get("perigon_enabled", True):
                source_tags.append("News")
            if proj.get("tavily_enabled", True):
                source_tags.append("Web (Tavily)")
            if proj.get("exa_enabled", True):
                source_tags.append("Web (Exa)")
            if not source_tags:
                source_tags.append("No Sources")
            tags_html = "".join(
                [f'<span class="project-tag">{tag}</span>' for tag in source_tags]
            )

            kb_status = "Ready" if (proj.get("kb_chunk_count") or 0) > 0 else "Pending"
            refreshed_at = _short_date(proj.get("last_refreshed_at"))
            created_at = _short_date(proj.get("created_at"))

            with cols[idx % grid_cols]:
                st.markdown(
                    f"""
                    <div class="project-card">
                        <div class="project-card-title">{proj["title"]}</div>
                        <div class="project-card-meta">
                            Strategy: {proj["refresh_strategy"]} | Chunks: {proj.get("kb_chunk_count", 0)}
                        </div>
                        <div class="project-tags">{tags_html}</div>
                        <div class="project-card-desc">{(proj.get("description") or "")[:160]}</div>
                        <div class="project-card-foot">
                            <span>KB: {kb_status}</span>
                            <span>Refreshed: {refreshed_at}</span>
                            <span>Created: {created_at}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(
                        "Open Chat", key=f"home_chat_{pid}", use_container_width=True
                    ):
                        _select_project(pid, "Chat")
                        st.rerun()
                with c2:
                    if st.button(
                        "Discover", key=f"home_discover_{pid}", use_container_width=True
                    ):
                        _select_project(pid, "Discover")
                        st.rerun()
                with c3:
                    if st.button(
                        "Manage", key=f"home_manage_{pid}", use_container_width=True
                    ):
                        _select_project(pid, "Projects")
                        st.rerun()

# ----------------------------------------------------------------
# PROJECTS PAGE - create / delete projects, upload docs, view KB
# ----------------------------------------------------------------
elif st.session_state.current_page == "Projects":
    st.header("Projects")

    # --- Create project form ---
    with st.expander("Create New Project", expanded=False):
        _render_create_project_form(form_key="projects_create_project_form")

    st.divider()

    # --- Project list ---
    if not st.session_state.projects:
        st.info("No projects yet. Create one above.")
    else:
        # Pre-fetch all KB statuses before the loop to avoid N+1 HTTP calls.
        kb_statuses: dict[str, dict] = {
            proj["id"]: (_api.get_kb_status(proj["id"]) or {})
            for proj in st.session_state.projects
        }
        for proj in st.session_state.projects:
            pid = proj["id"]
            with st.container():
                col_title, col_delete = st.columns([5, 1])
                with col_title:
                    st.subheader(_strip_emoji_symbols(proj["title"]))
                    st.caption(
                        f"Strategy: {proj['refresh_strategy']} | "
                        f"Chunks: {proj.get('kb_chunk_count', 0)} | "
                        f"Created: {_short_date(proj.get('created_at'))}"
                    )
                with col_delete:
                    if st.button("Delete", key=f"del_{pid}", help="Delete project"):
                        with st.spinner("Deleting..."):
                            ok = _api.delete_project(pid)
                        if ok:
                            if st.session_state.selected_project_id == pid:
                                st.session_state.selected_project_id = None
                                _reset_insight_view_state()
                            reload_projects()
                            st.rerun()
                        else:
                            st.error("Delete failed.")

                # KB Status (pre-fetched above to avoid N+1 HTTP calls)
                kb = kb_statuses.get(pid) or {}
                if kb.get("status"):
                    status_color = "green" if kb.get("status") == "ready" else "orange"
                    st.markdown(
                        f"KB: :{status_color}[{kb['status']}] - "
                        f"{kb['kb_chunk_count']} chunks"
                        + (
                            f" | last refreshed {_short_datetime_ist(kb.get('last_refreshed_at'))}"
                            if kb.get("last_refreshed_at")
                            else ""
                        )
                    )

                # Document upload
                uploaded_file = st.file_uploader(
                    "Upload document (pdf/docx/txt/md)",
                    type=["pdf", "docx", "txt", "md"],
                    key=f"upload_{pid}",
                )
                if uploaded_file is not None:
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        result = _api.upload_document(
                            pid, uploaded_file.name, uploaded_file.read()
                        )
                    if result:
                        st.success(
                            f"Uploaded {result['filename']} (ID: {result['upload_id'][:8]}...). "
                            "Processing in background - refresh KB status to track progress."
                        )
                    else:
                        st.error("Upload failed. Check backend logs.")

                st.divider()


# ----------------------------------------------------------------
# CHAT PAGE - project SSE mode OR agent mode
# ----------------------------------------------------------------
elif st.session_state.current_page == "Chat":
    project_id = st.session_state.selected_project_id

    # ---- PROJECT MODE ----
    if project_id:
        # Find project name
        proj_name = project_id
        for p in st.session_state.projects:
            if p["id"] == project_id:
                proj_name = _strip_emoji_symbols(p["title"])
                break

        st.header(proj_name)
        st.caption("Chatting with your project knowledge base via FastAPI")

        # Initialise message list for this project and load persisted history once
        if project_id not in st.session_state.project_messages:
            st.session_state.project_messages[project_id] = []
        if project_id not in st.session_state.project_history_loaded:
            session_id = _project_chat_session_id(project_id)
            history = _api.get_project_chat_history(project_id, session_id)
            if history:
                st.session_state.project_messages[project_id] = history
            st.session_state.project_history_loaded.add(project_id)

        msgs = st.session_state.project_messages[project_id]

        # Display history
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask about {proj_name}..."):
            msgs.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                response_text = ""
                session_id = _project_chat_session_id(project_id)
                for token in _api.stream_chat(prompt, project_id, session_id):
                    response_text += token
                    placeholder.markdown(response_text + "|")
                placeholder.markdown(response_text)

            msgs.append({"role": "assistant", "content": response_text})
            st.rerun()

    else:
        st.info("Select a project from Home to start chat.")
        if st.button("Go to Home", key="chat_go_home"):
            st.session_state.current_page = "Home"
            st.rerun()


def _render_ingest_status(status_payload: dict | None) -> None:
    """Show project ingest lifecycle status in Discover."""
    if not status_payload:
        return

    status = str(status_payload.get("status") or "").strip().lower()
    message = str(status_payload.get("message") or "").strip()
    total_chunks = status_payload.get("total_chunks")
    source_counts = status_payload.get("source_counts") or {}

    counts_line = ""
    if isinstance(source_counts, dict) and source_counts:
        non_zero = [f"{k}:{v}" for k, v in source_counts.items() if int(v or 0) > 0]
        if non_zero:
            counts_line = " | Sources: " + ", ".join(non_zero)

    tail = ""
    if total_chunks is not None:
        tail = f" | Chunks: {total_chunks}"
    details = f"{message}{tail}{counts_line}".strip()

    if status in {"queued", "running"}:
        st.info(f"Ingestion {status}. {details}".strip())
    elif status == "failed":
        st.error(f"Ingestion failed. {details}".strip())
    elif status == "empty":
        st.warning(f"Ingestion finished with no new documents. {details}".strip())
    elif status == "ready":
        st.success(f"Ingestion complete. {details}".strip())


def _trend_css_class(signal: str | None) -> str:
    """Map trend signal to CSS class."""
    value = str(signal or "unknown").strip().lower()
    if value in {"rising", "declining", "emerging", "stable"}:
        return f"trend-{value}"
    return "trend-unknown"


def _source_label(source_key: str) -> str:
    """Map source keys to display labels."""
    labels = {
        "social_tiktok": "TikTok",
        "social_instagram": "Instagram",
        "social_youtube": "YouTube",
        "social_reddit": "Reddit",
        "social_x": "X",
        "paper": "Papers",
        "patent": "Patents",
        "news": "News",
        "search": "Web",
        "upload": "Uploads",
    }
    return labels.get(source_key, source_key.title())


def _source_pills_html(source_counts: dict) -> str:
    """Render source count pills for a cluster card."""
    if not isinstance(source_counts, dict) or not source_counts:
        return '<span class="source-pill-cluster">No sources</span>'
    ordered = sorted(source_counts.items(), key=lambda pair: int(pair[1]), reverse=True)
    pills = [
        f'<span class="source-pill-cluster">{int(count)} {_source_label(str(source_key))}</span>'
        for source_key, count in ordered
        if int(count) > 0
    ]
    return (
        "".join(pills)
        if pills
        else '<span class="source-pill-cluster">No sources</span>'
    )


def _render_cluster_card(card: dict) -> None:
    """Render one insight card in feed grid."""
    topic = escape(_strip_emoji_symbols(card.get("topic_label") or "Untitled Insight"))
    summary = escape(_strip_emoji_symbols(card.get("executive_summary") or ""))
    trend = escape(str(card.get("trend_signal") or "unknown").upper())
    trend_class = _trend_css_class(str(card.get("trend_signal") or "unknown"))
    source_counts = card.get("source_type_counts") or {}
    source_total = int(card.get("source_doc_count") or 0)
    if source_total <= 0 and isinstance(source_counts, dict):
        source_total = sum(int(v) for v in source_counts.values())

    st.markdown(
        (
            '<div class="cluster-card">'
            f'<div class="cluster-card-topic">{topic} '
            f'<span class="trend-badge {trend_class}">{trend}</span></div>'
            f'<p class="discover-card-summary">{summary}</p>'
            f"<div>{_source_pills_html(source_counts)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button(
            "Read Full Report ->",
            key=f"read_report_{card.get('id')}",
            use_container_width=True,
        ):
            st.session_state.selected_insight_id = card.get("id")
            st.session_state.current_page = "InsightDetail"
            st.rerun()
    with cols[1]:
        if st.button(
            f"See all {source_total} sources",
            key=f"open_sources_{card.get('id')}",
            use_container_width=True,
        ):
            _clear_source_card_flip_state(str(card.get("id") or ""))
            st.session_state.drawer_insight_id = card.get("id")
            st.session_state.show_sources_drawer = True
            st.rerun()


def _render_insight_feed(project_id: str) -> None:
    """Render the main insights feed."""
    cards = _api.get_insights(project_id)
    if not cards:
        st.info("Insights are being generated...")
        with st.spinner("Generating clustered insights..."):
            st.empty()
        return
    grid_count = 3
    cols = st.columns(grid_count)
    for idx, card in enumerate(cards):
        with cols[idx % grid_count]:
            _render_cluster_card(card)


def _compact_count(value: object) -> str:
    """Format numeric counts into compact human-readable units."""
    try:
        n = int(float(value or 0))
    except Exception:
        return ""
    if n <= 0:
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".rstrip("0").rstrip(".")
    if n >= 1_000:
        return f"{n / 1_000:.1f}K".rstrip("0").rstrip(".")
    return str(n)


def _format_published_label(value: object) -> str:
    """Format source published value into short date label."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    dt = _parse_to_ist(raw)
    if dt:
        return dt.strftime("%d %b %Y")
    numeric = raw.replace(".", "", 1)
    if numeric.isdigit():
        try:
            ts = float(raw)
            if ts <= 0:
                return ""
            if ts > 1_000_000_000_000:
                ts = ts / 1000.0
            return (
                datetime.fromtimestamp(ts, tz=UTC)
                .astimezone(IST_TZ)
                .strftime("%d %b %Y")
            )
        except Exception:
            return ""
    return ""


def _collect_streamed_text(stream: Generator[str, None, None]) -> str:
    """Collect streamed token chunks into one final text payload."""
    parts: list[str] = []
    for chunk in stream:
        if isinstance(chunk, str) and chunk:
            parts.append(chunk)
    return "".join(parts).strip()


def _render_source_card(doc: dict, insight_id: str) -> None:
    """Render one source card with media, short front summary, and flip brief."""
    title = _strip_emoji_symbols(doc.get("title") or "Untitled source")
    source = str(doc.get("source") or "source")
    doc_id = str(doc.get("id") or doc.get("url") or doc.get("title") or "")
    flip_state_key = f"source_card_state_flip_{insight_id}_{doc_id}"
    is_flipped = bool(st.session_state.get(flip_state_key, False))
    flip_button_key = f"source_card_flip_btn_{insight_id}_{doc_id}"
    back_button_key = f"source_card_back_btn_{insight_id}_{doc_id}"
    url = str(doc.get("url") or "").strip()
    summary_preview, summary_full = build_source_card_summaries(doc.get("summary"))
    show_summary_button = should_show_source_summary_button(doc.get("summary"))
    cover_url = str(doc.get("cover_url") or "").strip()
    video_url = str(doc.get("video_url") or "").strip()
    author = _strip_emoji_symbols(doc.get("author") or "").strip()
    views_label = _compact_count(doc.get("views"))
    likes_label = _compact_count(doc.get("likes"))
    published_label = _format_published_label(doc.get("published_at"))
    media_html = build_source_card_media(
        source=source,
        url=url,
        cover_url=cover_url,
        video_url=video_url,
        source_label=_source_label(source),
    )

    title_html = escape(title)
    if url:
        title_html = f'<a href="{escape(url)}" target="_blank">{escape(title)}</a>'
    meta_parts: list[str] = []
    if author:
        meta_parts.append(author)
    if views_label:
        meta_parts.append(f"{views_label} views")
    if likes_label:
        meta_parts.append(f"{likes_label} likes")
    if published_label:
        meta_parts.append(published_label)
    meta_line = escape(" | ".join(meta_parts))

    st.markdown(
        (
            '<div class="source-card-scene">'
            f'<div class="source-card-flip-shell{" is-flipped" if is_flipped else ""}">'
            '<div class="discover-card discover-card-face discover-card-face-front source-card-fixed-height">'
            f"{media_html}"
            '<div class="discover-card-body">'
            f'<div class="discover-source-pill">{escape(_source_label(source))}</div>'
            f'<div class="discover-card-title">{title_html}</div>'
            f'<div class="discover-card-summary discover-card-summary-front">{escape(summary_preview or "No source summary available.")}</div>'
            f'<div class="discover-card-meta">{meta_line}</div>'
            "</div>"
            "</div>"
            '<div class="discover-card discover-card-face discover-card-face-back source-card-fixed-height">'
            '<div class="discover-card-body discover-card-body-back">'
            f'<div class="discover-card-summary discover-card-summary-back source-card-back-copy">{escape(summary_full or "No source summary available.")}</div>'
            "</div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if is_flipped:
        if st.button("Back to Preview", key=back_button_key, use_container_width=True):
            st.session_state[flip_state_key] = False
            st.rerun()
        return

    if show_summary_button and st.button(
        "View Full Summary", key=flip_button_key, use_container_width=True
    ):
        st.session_state[flip_state_key] = True
        st.rerun()


def _render_sources_drawer(project_id: str) -> None:
    """Render sources drawer dialog for selected insight."""
    insight_id = str(st.session_state.get("drawer_insight_id") or "").strip()

    @st.dialog("Sources", width="large", dismissible=False)
    def _dialog() -> None:
        if st.button("Close", key="close_sources_dialog"):
            _clear_source_card_flip_state(insight_id)
            st.session_state.show_sources_drawer = False
            st.session_state.drawer_insight_id = None
            st.rerun()
        # Prefer cached detail from _render_insight_detail to avoid duplicate API call
        detail = st.session_state.get("cached_insight_detail")
        if not detail or str((detail or {}).get("id") or "") != insight_id:
            detail = _api.get_insight_detail(project_id, insight_id)
        source_docs = (detail or {}).get("source_docs") or []
        if not source_docs:
            st.info("No sources found for this insight.")
            return
        st.markdown('<div class="sources-dialog-inner">', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, doc in enumerate(source_docs):
            with cols[idx % len(cols)]:
                _render_source_card(doc, insight_id)
        st.markdown("</div>", unsafe_allow_html=True)

    _dialog()


def _render_insight_detail(project_id: str, insight_id: str) -> None:
    """Render full-page insight detail view."""
    detail = _api.get_insight_detail(project_id, insight_id)
    if not detail:
        st.warning("Insight not found or no longer available.")
        return
    st.session_state["cached_insight_detail"] = detail

    top_cols = st.columns([1, 4, 1])
    with top_cols[0]:
        if st.button("Back to Insights", key="insight_back"):
            _reset_insight_view_state()
            st.session_state.current_page = "Discover"
            st.rerun()
    with top_cols[2]:
        source_total = int(detail.get("source_doc_count") or 0)
        if source_total <= 0:
            source_total = len((detail.get("source_docs") or []))
        if source_total <= 0:
            source_total = sum(
                int(v) for v in (detail.get("source_type_counts") or {}).values()
            )
        if st.button(f"See all {source_total} sources", key="insight_open_sources"):
            _clear_source_card_flip_state(insight_id)
            st.session_state.drawer_insight_id = insight_id
            st.session_state.show_sources_drawer = True
            st.rerun()

    trend = str(detail.get("trend_signal") or "unknown")
    st.markdown(
        (
            '<div class="insight-report-shell">'
            f'<div class="insight-report-title">{escape(_strip_emoji_symbols(detail.get("topic_label") or "Insight"))} '
            f'<span class="trend-badge {_trend_css_class(trend)}">{escape(trend.upper())}</span></div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    report_status = str(detail.get("full_report_status") or "pending")
    report_text = str(detail.get("full_report") or "").strip()
    cache = st.session_state.insight_report_cache
    requested = st.session_state.insight_report_requested
    request_key = f"{project_id}:{insight_id}"
    drawer_open_for_insight = bool(
        st.session_state.get("show_sources_drawer")
        and str(st.session_state.get("drawer_insight_id") or "") == insight_id
    )

    if report_text:
        cache[insight_id] = report_text
        requested.add(request_key)
        st.markdown(report_text)
    elif report_status == "generating":
        requested.add(request_key)
        st.info("Generating full report...")
    else:
        if request_key not in requested and not drawer_open_for_insight:
            requested.add(request_key)
            with st.spinner("Generating full report..."):
                streamed = _collect_streamed_text(
                    _api.stream_full_report(project_id, insight_id)
                )
            if streamed:
                cache[insight_id] = streamed
                st.markdown(streamed)
            elif cache.get(insight_id):
                st.markdown(cache[insight_id])
        elif report_status == "failed":
            st.warning("Full report generation failed. Retry when ready.")
            if st.button("Retry Full Report", key=f"retry_report_{insight_id}"):
                requested.discard(request_key)
                st.rerun()
        else:
            st.info("Full report is pending generation.")
            if st.button("Generate Full Report", key=f"generate_report_{insight_id}"):
                requested.discard(request_key)
                st.rerun()

    history = _api.get_followup_history(insight_id)
    if history:
        for message in history[-8:]:
            role = str(message.get("role") or "assistant")
            content = str(message.get("content") or "")
            with st.chat_message(role):
                st.markdown(content)

    prompt = st.chat_input("Ask a follow-up from this cluster sources...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.write_stream(_api.stream_followup(insight_id, prompt))


# ----------------------------------------------------------------
# INSIGHTS PAGE (Discover replacement)
# ----------------------------------------------------------------
if st.session_state.current_page == "Discover":
    st.header("Insights")
    project_id = st.session_state.selected_project_id
    if project_id:
        ingest_status = _api.get_ingest_status(project_id)
        _sync_project_chunk_count_from_ingest(project_id, ingest_status)
        _render_ingest_status(ingest_status)

        if st.button("Refresh Insights", key="refresh_insights_btn"):
            accepted, message = _api.refresh_insights(project_id)
            st.toast(message)
            if not accepted:
                st.info(message)
            st.rerun()
        _render_insight_feed(project_id)
    else:
        st.info("Select a project from Home to open Insights.")
        if st.button("Go to Home", key="discover_go_home"):
            st.session_state.current_page = "Home"
            st.rerun()


if st.session_state.current_page == "InsightDetail":
    project_id = st.session_state.selected_project_id
    insight_id = st.session_state.selected_insight_id
    if project_id and insight_id:
        _render_insight_detail(project_id, insight_id)
    else:
        st.info("Select an insight from Discover first.")
        if st.button("Back to Insights", key="insightdetail_back_fallback"):
            _reset_insight_view_state()
            st.session_state.current_page = "Discover"
            st.rerun()


if st.session_state.get("current_page") not in {"Discover", "InsightDetail"}:
    st.session_state.show_sources_drawer = False
    st.session_state.drawer_insight_id = None

if (
    st.session_state.get("current_page") in {"Discover", "InsightDetail"}
    and st.session_state.get("show_sources_drawer")
    and st.session_state.get("drawer_insight_id")
    and st.session_state.get("selected_project_id")
):
    _render_sources_drawer(st.session_state.selected_project_id)
