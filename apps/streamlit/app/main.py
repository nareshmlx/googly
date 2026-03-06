"""
Beauty Social AI — Project-first Streamlit UI.

Flow:
1. Land on Home.
2. Create or select a project.
3. Redirect into project Chat.

Discover and project management remain available once a project is selected.
"""

import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import streamlit as st

import _api
from agent import get_database

IST_TZ = ZoneInfo("Asia/Kolkata")

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
    page_icon="💄",
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
if "setup_status_last" not in st.session_state:
    st.session_state.setup_status_last = {}

_apply_theme("Rose Glow")


# ============ HELPER FUNCTIONS ============


def reload_projects():
    """Fetch projects from FastAPI and update session state."""
    st.session_state.projects = _api.list_projects()
    st.session_state.projects_loaded = True


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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST_TZ)


def _short_date(value: str | None) -> str:
    """Return YYYY-MM-DD in IST from an ISO datetime-like string."""
    dt = _parse_to_ist(value)
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d")


def _short_datetime_ist(value: str | None) -> str:
    """Return YYYY-MM-DD HH:MM IST from an ISO datetime-like string."""
    dt = _parse_to_ist(value)
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M IST")


def _wizard_state_key(form_key: str) -> str:
    return f"{form_key}_wizard_state"


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
        f"{form_key}_wizard_time_horizon": str(payload.get("time_horizon") or "last 1 year"),
        f"{form_key}_wizard_refresh_strategy": str(state.get("refresh_strategy") or "once"),
        f"{form_key}_wizard_enriched_description": str(payload.get("enriched_description") or ""),
        f"{form_key}_wizard_src_tiktok": bool(source_toggles.get("tiktok", True)),
        f"{form_key}_wizard_src_instagram": bool(source_toggles.get("instagram", True)),
        f"{form_key}_wizard_src_youtube": bool(source_toggles.get("youtube", True)),
        f"{form_key}_wizard_src_reddit": bool(source_toggles.get("reddit", True)),
        f"{form_key}_wizard_src_x": bool(source_toggles.get("x", True)),
        f"{form_key}_wizard_src_papers": bool(source_toggles.get("papers", True)),
        f"{form_key}_wizard_src_patents": bool(source_toggles.get("patents", True)),
        f"{form_key}_wizard_src_news": bool(source_toggles.get("news", True)),
        f"{form_key}_wizard_src_web_tavily": bool(source_toggles.get("web_tavily", True)),
        f"{form_key}_wizard_src_web_exa": bool(source_toggles.get("web_exa", True)),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_create_project_form(
    *,
    form_key: str,
    submit_label: str = "Create Project",
    defaults: dict | None = None,
) -> None:
    """Render and handle the two-phase project wizard."""
    cfg = defaults or {}
    state = _ensure_wizard_state(form_key, cfg)

    st.markdown("#### Project Wizard")
    st.caption("Phase 1: Dynamic Q&A • Phase 2: Review & Confirm")

    if state.get("phase") == "intro":
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
            if st.button("Start Wizard", key=f"{form_key}_wizard_start", use_container_width=True):
                title_value = (title or "").strip()
                description_value = (description or "").strip()
                if not title_value:
                    st.error("Title is required.")
                    return
                if len(description_value) < 10:
                    st.error("Description must be at least 10 characters.")
                    return
                with st.spinner("Evaluating project clarity..."):
                    payload, err = _api.wizard_evaluate(
                        title_value,
                        description_value,
                        qa_pairs=[],
                        max_questions=int(state.get("max_questions") or 5),
                    )
                if err or not payload:
                    st.error(err or "Could not start wizard.")
                    return

                state["title"] = title_value
                state["description"] = description_value
                state["scores"] = payload.get("scores", {})
                state["qa_pairs"] = []
                state["pending_question"] = payload.get("next_question") or ""
                state["pending_dimension"] = payload.get("next_dimension") or ""

                if payload.get("should_stop"):
                    with st.spinner("Synthesizing review fields..."):
                        review_payload, synth_err = _api.wizard_synthesize(
                            title=title_value,
                            description=description_value,
                            qa_pairs=[],
                            structured_intent={},
                            source_toggles=state.get("source_toggles") or {},
                        )
                    if synth_err or not review_payload:
                        st.error(synth_err or "Could not synthesize review.")
                        return
                    state["review_payload"] = review_payload
                    state["source_toggles"] = review_payload.get("target_sources") or state.get(
                        "source_toggles", {}
                    )
                    state["phase"] = "review"
                else:
                    state["phase"] = "qa"
                st.rerun()
        with col_reset:
            if st.button("Reset", key=f"{form_key}_wizard_intro_reset", use_container_width=True):
                _reset_wizard_state(form_key, cfg)
                st.rerun()
        return

    if state.get("phase") == "qa":
        scores = state.get("scores") or {}
        st.markdown(
            (
                f"Clarity scores: objective **{scores.get('objective_clarity', 0.0):.2f}**, "
                f"pain point **{scores.get('pain_point_clarity', 0.0):.2f}**, "
                f"output **{scores.get('output_clarity', 0.0):.2f}**, "
                f"domain **{scores.get('domain_specificity', 0.0):.2f}**"
            )
        )

        answered_pairs = list(state.get("qa_pairs") or [])
        pending_question = str(state.get("pending_question") or "").strip()
        pending_dimension = str(state.get("pending_dimension") or "").strip()

        labels = [f"Answered {idx + 1}" for idx in range(len(answered_pairs))]
        if pending_question:
            labels.append(f"Question {len(answered_pairs) + 1}")
        tabs = st.tabs(labels or ["Question 1"])

        for idx, item in enumerate(answered_pairs):
            with tabs[idx]:
                st.markdown(f"**Question:** {item.get('question', '')}")
                st.markdown(f"**Answer:** {item.get('answer', '')}")

        if pending_question:
            answer_key = f"{form_key}_wizard_answer_{len(answered_pairs)}"
            with tabs[-1]:
                st.markdown(f"**Question:** {pending_question}")
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
                        with st.spinner("Refining wizard context..."):
                            payload, err = _api.wizard_evaluate(
                                state.get("title", ""),
                                state.get("description", ""),
                                qa_pairs=answered_pairs,
                                max_questions=int(state.get("max_questions") or 5),
                            )
                        if err or not payload:
                            st.error(err or "Failed to process answer.")
                            return

                        state["scores"] = payload.get("scores", {})
                        state["pending_question"] = payload.get("next_question") or ""
                        state["pending_dimension"] = payload.get("next_dimension") or ""

                        if payload.get("should_stop"):
                            with st.spinner("Synthesizing review fields..."):
                                review_payload, synth_err = _api.wizard_synthesize(
                                    title=state.get("title", ""),
                                    description=state.get("description", ""),
                                    qa_pairs=answered_pairs,
                                    structured_intent={},
                                    source_toggles=state.get("source_toggles") or {},
                                )
                            if synth_err or not review_payload:
                                st.error(synth_err or "Could not synthesize review.")
                                return
                            state["review_payload"] = review_payload
                            state["source_toggles"] = review_payload.get("target_sources") or state.get(
                                "source_toggles", {}
                            )
                            state["phase"] = "review"
                        st.rerun()
                with col_restart:
                    if st.button(
                        "Start Over",
                        key=f"{form_key}_wizard_qa_reset",
                        use_container_width=True,
                    ):
                        _reset_wizard_state(form_key, cfg)
                        st.rerun()
        return

    if state.get("phase") != "review":
        _reset_wizard_state(form_key, cfg)
        st.rerun()
        return

    _ensure_review_widget_defaults(form_key, state)
    review_payload = state.get("review_payload") or {}

    st.markdown("#### Review & Confirm")
    st.caption("Edit fields below before creating the final project.")

    with st.expander("Conversation Summary", expanded=False):
        st.markdown(f"**Title:** {state.get('title', '')}")
        st.markdown(f"**Description:** {state.get('description', '')}")
        for idx, item in enumerate(state.get("qa_pairs") or []):
            st.markdown(f"{idx + 1}. **Q:** {item.get('question', '')}")
            st.markdown(f"   **A:** {item.get('answer', '')}")

    st.text_area(
        "Enriched Description",
        key=f"{form_key}_wizard_enriched_description",
        height=220,
        help="Dense semantic summary used for downstream ranking and retrieval.",
    )

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
    current_horizon = st.session_state.get(f"{form_key}_wizard_time_horizon", "last 1 year")
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

    st.markdown("**Target Data Sources**")
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
        "instagram": bool(st.session_state.get(f"{form_key}_wizard_src_instagram", True)),
        "youtube": bool(st.session_state.get(f"{form_key}_wizard_src_youtube", True)),
        "reddit": bool(st.session_state.get(f"{form_key}_wizard_src_reddit", True)),
        "x": bool(st.session_state.get(f"{form_key}_wizard_src_x", True)),
        "papers": bool(st.session_state.get(f"{form_key}_wizard_src_papers", True)),
        "patents": bool(st.session_state.get(f"{form_key}_wizard_src_patents", True)),
        "news": bool(st.session_state.get(f"{form_key}_wizard_src_news", True)),
        "web_tavily": bool(st.session_state.get(f"{form_key}_wizard_src_web_tavily", True)),
        "web_exa": bool(st.session_state.get(f"{form_key}_wizard_src_web_exa", True)),
    }

    col_create, col_back, col_reset = st.columns([3, 1, 1])
    with col_create:
        if st.button(submit_label, key=f"{form_key}_wizard_create", use_container_width=True):
            with st.spinner("Creating project..."):
                result, create_error = _api.wizard_create(
                    title=state.get("title", ""),
                    description=state.get("description", ""),
                    qa_pairs=list(state.get("qa_pairs") or []),
                    refresh_strategy=str(
                        st.session_state.get(f"{form_key}_wizard_refresh_strategy", "once")
                    ),
                    domain_focus=domain_focus,
                    key_entities=list(st.session_state.get(f"{form_key}_wizard_entities", [])),
                    must_match_terms=list(st.session_state.get(f"{form_key}_wizard_must_terms", [])),
                    time_horizon=str(
                        st.session_state.get(f"{form_key}_wizard_time_horizon", "last 1 year")
                    ),
                    target_sources=target_sources,
                )

            if not result:
                st.error(create_error or "Failed to create project. Please try again.")
                return

            upload_ids: list[str] = []
            for upload_file in bootstrap_files or []:
                upload_result = _api.upload_document(
                    result["id"], upload_file.name, upload_file.read()
                )
                if upload_result and upload_result.get("upload_id"):
                    upload_ids.append(upload_result["upload_id"])

            with st.spinner("Starting project setup..."):
                _api.bootstrap_project(result["id"], upload_ids)

            st.session_state.selected_project_id = result["id"]
            st.session_state.current_page = "Chat"
            _reset_wizard_state(form_key, cfg)
            reload_projects()
            st.success(f"Project '{result['title']}' created.")
            st.rerun()
    with col_back:
        if st.button("Back", key=f"{form_key}_wizard_back_to_qa", use_container_width=True):
            state["phase"] = "qa"
            st.rerun()
    with col_reset:
        if st.button("Reset", key=f"{form_key}_wizard_review_reset", use_container_width=True):
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
        st.title("💄 Beauty Social")
        st.caption("P&G Beauty Intelligence")

        st.divider()

        # Navigation: Home-first. Chat/Discover unlock only when a project is selected.
        nav_options = ["Home", "Projects"]
        if st.session_state.selected_project_id:
            nav_options.extend(["Chat", "Discover"])
        if st.session_state.current_page not in nav_options:
            st.session_state.current_page = "Home"

        page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(st.session_state.current_page),
            label_visibility="collapsed",
        )
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            if page == "Home":
                reload_projects()
            st.rerun()

        st.divider()

        active_label = "None selected"
        if st.session_state.selected_project_id:
            for p in st.session_state.projects:
                if p["id"] == st.session_state.selected_project_id:
                    active_label = p["title"]
                    break

        st.markdown(f"**Active Project:** {active_label}")

        if st.button("🔄 Refresh Projects", use_container_width=True):
            reload_projects()
            st.rerun()

        if st.session_state.selected_project_id and st.button(
            "↩ Back to Home", use_container_width=True
        ):
            st.session_state.current_page = "Home"
            st.rerun()

        if st.session_state.selected_project_id and st.button(
            "✕ Clear Selection", use_container_width=True
        ):
            st.session_state.selected_project_id = None
            st.session_state.current_page = "Home"
            st.rerun()

        if st.button("➕ Create Project", use_container_width=True):
            st.session_state.current_page = "Home"
            st.session_state["home_show_create"] = True
            st.rerun()

        st.divider()

        # Database / memory status — checked once at startup only
        if "db_status" not in st.session_state:
            st.session_state.db_status = get_database() is not None
        if st.session_state.db_status:
            st.success("🧠 Memory Active")
        else:
            st.warning("⚠️ Memory Offline")

# ============ MAIN CONTENT ============

# ----------------------------------------------------------------
# HOME PAGE — default landing with project gallery + quick create
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
    latest_refresh = _short_date(max(refresh_values)) if refresh_values else "—"

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
                            Strategy: {proj["refresh_strategy"]} · Chunks: {proj.get("kb_chunk_count", 0)}
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
                        st.session_state.selected_project_id = pid
                        st.session_state.current_page = "Chat"
                        st.rerun()
                with c2:
                    if st.button(
                        "Discover", key=f"home_discover_{pid}", use_container_width=True
                    ):
                        st.session_state.selected_project_id = pid
                        st.session_state.current_page = "Discover"
                        st.rerun()
                with c3:
                    if st.button(
                        "Manage", key=f"home_manage_{pid}", use_container_width=True
                    ):
                        st.session_state.selected_project_id = pid
                        st.session_state.current_page = "Projects"
                        st.rerun()

# ----------------------------------------------------------------
# PROJECTS PAGE — create / delete projects, upload docs, view KB
# ----------------------------------------------------------------
elif st.session_state.current_page == "Projects":
    st.header("🗂 Projects")

    # --- Create project form ---
    with st.expander("➕ Create New Project", expanded=False):
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
                    st.subheader(proj["title"])
                    st.caption(
                        f"Strategy: {proj['refresh_strategy']} | "
                        f"Chunks: {proj.get('kb_chunk_count', 0)} | "
                        f"Created: {_short_date(proj.get('created_at'))}"
                    )
                with col_delete:
                    if st.button("🗑️", key=f"del_{pid}", help="Delete project"):
                        with st.spinner("Deleting..."):
                            ok = _api.delete_project(pid)
                        if ok:
                            if st.session_state.selected_project_id == pid:
                                st.session_state.selected_project_id = None
                            reload_projects()
                            st.rerun()
                        else:
                            st.error("Delete failed.")

                # KB Status (pre-fetched above to avoid N+1 HTTP calls)
                kb = kb_statuses.get(pid) or {}
                if kb.get("status"):
                    status_color = "green" if kb.get("status") == "ready" else "orange"
                    st.markdown(
                        f"KB: :{status_color}[{kb['status']}] — "
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
                            "Processing in background — refresh KB status to track progress."
                        )
                    else:
                        st.error("Upload failed. Check backend logs.")

                st.divider()


# ----------------------------------------------------------------
# CHAT PAGE — project SSE mode OR agent mode
# ----------------------------------------------------------------
elif st.session_state.current_page == "Chat":
    project_id = st.session_state.selected_project_id

    # ---- PROJECT MODE ----
    if project_id:
        # Find project name
        proj_name = project_id
        for p in st.session_state.projects:
            if p["id"] == project_id:
                proj_name = p["title"]
                break

        st.header(f"💬 {proj_name}")
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
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)

            msgs.append({"role": "assistant", "content": response_text})
            st.rerun()

    else:
        st.info("Select a project from Home to start chat.")
        if st.button("Go to Home", key="chat_go_home"):
            st.session_state.current_page = "Home"
            st.rerun()


def _normalize_discover_source(source: str | None) -> str:
    """Normalize backend source names into UI tab categories."""
    raw = (source or "").strip().lower()
    return {
        "social_tiktok": "tiktok",
        "social_instagram": "instagram",
        "social_youtube": "youtube",
        "social_reddit": "reddit",
        "social_x": "x",
        "web": "search",
    }.get(raw, raw)


def _to_discover_item(item: dict) -> dict:
    """Return a normalized discover item with safe defaults."""
    normalized = dict(item)
    normalized["source"] = _normalize_discover_source(item.get("source"))
    normalized["title"] = item.get("title") or "Untitled"
    normalized["summary"] = item.get("summary") or ""
    normalized["metadata"] = item.get("metadata") or {}
    normalized["author"] = (
        item.get("author") or normalized["metadata"].get("author") or ""
    )
    normalized["url"] = item.get("url") or normalized["metadata"].get("url")
    normalized["video_url"] = normalized["metadata"].get("video_url") or ""
    normalized["cover_url"] = (
        item.get("cover_url")
        or normalized["metadata"].get("cover_url")
        or normalized["metadata"].get("thumbnail_url")
        or normalized["metadata"].get("image_url")
        or ""
    )
    return normalized


def _discover_source_label(item: dict) -> str:
    """Human-friendly source labels for discover cards."""
    labels = {
        "instagram": "Instagram",
        "tiktok": "TikTok",
        "youtube": "YouTube",
        "reddit": "Reddit",
        "x": "X",
        "paper": "Research Paper",
        "patent": "Patent",
        "news": "News",
        "search": "Web",
    }
    return labels.get(item.get("source", ""), "Discover")


def _discover_meta_line(item: dict) -> str:
    """Return compact source-specific metadata line."""
    metadata = item.get("metadata") or {}
    source = item.get("source")
    published_raw = (
        item.get("published_at")
        or metadata.get("published_at")
        or metadata.get("published_date")
        or metadata.get("pubDate")
        or metadata.get("year")
    )
    published = str(published_raw)[:10] if published_raw else ""
    if source in {"paper", "patent"}:
        venue = metadata.get("venue") or metadata.get("assignee") or ""
        year = metadata.get("year") or published
        citations = metadata.get("citation_count")
        parts = [str(part) for part in (venue, year) if part]
        if citations is not None:
            parts.append(f"{citations:,} citations")
        return " • ".join(parts)
    if source == "news":
        source_name = metadata.get("source_name") or metadata.get("source") or ""
        return " • ".join([part for part in (source_name, published) if part])
    if source == "search":
        url = item.get("url") or ""
        domain = urlparse(url).netloc if url else ""
        return " • ".join([part for part in (domain, published) if part])
    if source in {"instagram", "tiktok", "youtube"}:
        likes = int(metadata.get("likes") or 0)
        views = int(metadata.get("views") or 0)
        social_stats = []
        if likes > 0:
            social_stats.append(f"{likes:,} likes")
        if views > 0:
            social_stats.append(f"{views:,} views")
        if published:
            social_stats.append(published)
        return " • ".join(social_stats)
    if source == "reddit":
        score = int(metadata.get("score") or 0)
        comments = int(metadata.get("comments") or 0)
        parts = []
        if score > 0:
            parts.append(f"{score:,} score")
        if comments > 0:
            parts.append(f"{comments:,} comments")
        if published:
            parts.append(published)
        return " • ".join(parts)
    if source == "x":
        likes = int(metadata.get("likes") or 0)
        retweets = int(metadata.get("retweets") or 0)
        parts = []
        if likes > 0:
            parts.append(f"{likes:,} likes")
        if retweets > 0:
            parts.append(f"{retweets:,} reposts")
        if published:
            parts.append(published)
        return " • ".join(parts)
    return published


def _social_tile_media_html(item: dict) -> str:
    """Build fixed-height social media HTML so media stays inside the same card block."""
    video_url = item.get("video_url")
    cover_url = item.get("cover_url")
    if video_url:
        safe_video_url = escape(video_url)
        safe_cover_url = escape(cover_url) if cover_url else ""
        poster_attr = f' poster="{safe_cover_url}"' if safe_cover_url else ""
        return f"""
        <div class="discover-media-wrap">
            <video controls preload="metadata"{poster_attr}
                   style="width:100%;height:228px;object-fit:cover;display:block;background:#000;">
                <source src="{safe_video_url}" type="video/mp4">
            </video>
        </div>
        """
    if cover_url:
        return f"""
        <div class="discover-media-wrap">
            <img class="discover-media-image" src="{escape(cover_url)}" />
        </div>
        """
    return '<div class="discover-media-wrap discover-media-empty">No media</div>'


def _render_discover_card(item: dict) -> None:
    """Render one compact discover card tile."""
    source_label = _discover_source_label(item)
    title = escape(item.get("title") or "Untitled")
    url = item.get("url")
    title_html = (
        f'<a href="{escape(url)}" target="_blank" style="color:#0f172a;text-decoration:none;">{title}</a>'
        if url
        else title
    )
    summary = escape((item.get("summary") or "")[:220])
    meta_line = escape(_discover_meta_line(item))
    author = escape(item.get("author") or "")
    media_html = ""
    if item.get("source") in {"instagram", "tiktok", "youtube"}:
        media_html = _social_tile_media_html(item)
    elif item.get("cover_url"):
        media_html = f"""
        <div class="discover-media-wrap">
            <img class="discover-media-image" src="{escape(item['cover_url'])}" />
        </div>
        """
    html = (
        '<div class="discover-card">'
        f"{media_html}"
        '<div class="discover-card-body" style="background:#ffffff;color:#0f172a;">'
        f'<div class="discover-source-pill" style="background:#e8f0ff;color:#1e3a5f;border:1px solid #d9e5ff;">{escape(source_label)}</div>'
        f'<div class="discover-card-title" style="color:#0f172a;">{title_html}</div>'
        f'<div class="discover-card-summary" style="color:#334155;">{summary}</div>'
        f'<div class="discover-card-meta" style="color:#64748b;">{author}{" • " if author and meta_line else ""}{meta_line}</div>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_discover_grid(items: list[dict], *, empty_msg: str) -> None:
    """Render compact, mixed discover tiles in backend-ranked order."""
    if not items:
        st.info(empty_msg)
        return

    column_count = 1 if len(items) == 1 else 3
    columns = st.columns(column_count)
    for idx, item in enumerate(items):
        with columns[idx % column_count]:
            _render_discover_card(item)


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


def _maybe_auto_refresh_discover(project_id: str, discover_item_count: int) -> None:
    """Track discover warm-up conditions without forcing blocking reruns."""
    setup = _api.get_setup_status(project_id)
    if setup:
        st.session_state.setup_status_last[project_id] = setup
    else:
        setup = st.session_state.setup_status_last.get(project_id)
    if not setup:
        return

    status = str(setup.get("status") or "unknown").strip().lower()
    non_terminal = {"queued", "running", "pending", "processing"}
    should_auto_refresh = (discover_item_count == 0) or (status in non_terminal)
    st.session_state[f"discover_auto_refresh_needed_{project_id}"] = should_auto_refresh


# ----------------------------------------------------------------
# DISCOVER PAGE
# ----------------------------------------------------------------
if st.session_state.current_page == "Discover":
    st.header("🔍 Discover")

    project_id = st.session_state.selected_project_id

    if project_id:
        proj_name = project_id
        for p in st.session_state.projects:
            if p["id"] == project_id:
                proj_name = p["title"]
                break

        st.caption(f"Discover feed for **{proj_name}**")

        col_refresh, col_sort = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh Feed"):
                st.rerun()
        with col_sort:
            st.caption("Sorted by backend relevance score")

        raw_items = _api.get_discover_feed(project_id)
        items = [_to_discover_item(item) for item in raw_items]

        _maybe_auto_refresh_discover(project_id, len(items))

        ingest_status = _api.get_ingest_status(project_id)
        _sync_project_chunk_count_from_ingest(project_id, ingest_status)
        _render_ingest_status(ingest_status)

        if items:
            tiktok_items = [item for item in items if item.get("source") == "tiktok"]
            instagram_items = [item for item in items if item.get("source") == "instagram"]
            youtube_items = [item for item in items if item.get("source") == "youtube"]
            reddit_items = [item for item in items if item.get("source") == "reddit"]
            x_items = [item for item in items if item.get("source") == "x"]
            social_items = tiktok_items + instagram_items + youtube_items + reddit_items + x_items
            paper_items = [item for item in items if item.get("source") == "paper"]
            patent_items = [item for item in items if item.get("source") == "patent"]
            news_items = [item for item in items if item.get("source") == "news"]
            web_items = [item for item in items if item.get("source") == "search"]

            (
                tab_all,
                tab_social,
                tab_tiktok,
                tab_instagram,
                tab_youtube,
                tab_reddit,
                tab_x,
                tab_papers,
                tab_patents,
                tab_news,
                tab_web,
            ) = st.tabs(
                [
                    f"All ({len(items)})",
                    f"Social ({len(social_items)})",
                    f"TikTok ({len(tiktok_items)})",
                    f"Instagram ({len(instagram_items)})",
                    f"YouTube ({len(youtube_items)})",
                    f"Reddit ({len(reddit_items)})",
                    f"X ({len(x_items)})",
                    f"Papers ({len(paper_items)})",
                    f"Patents ({len(patent_items)})",
                    f"News ({len(news_items)})",
                    f"Web ({len(web_items)})",
                ]
            )

            with tab_all:
                _render_discover_grid(items, empty_msg="No discover items yet.")

            with tab_social:
                _render_discover_grid(
                    social_items,
                    empty_msg="No social content yet.",
                )

            with tab_tiktok:
                _render_discover_grid(tiktok_items, empty_msg="No TikTok content yet.")

            with tab_instagram:
                _render_discover_grid(instagram_items, empty_msg="No Instagram content yet.")

            with tab_youtube:
                _render_discover_grid(youtube_items, empty_msg="No YouTube content yet.")

            with tab_reddit:
                _render_discover_grid(reddit_items, empty_msg="No Reddit content yet.")

            with tab_x:
                _render_discover_grid(x_items, empty_msg="No X content yet.")

            with tab_papers:
                _render_discover_grid(
                    paper_items,
                    empty_msg="No papers found yet.",
                )

            with tab_patents:
                _render_discover_grid(
                    patent_items,
                    empty_msg="No patents found yet.",
                )

            with tab_news:
                _render_discover_grid(
                    news_items,
                    empty_msg="No news articles yet.",
                )

            with tab_web:
                _render_discover_grid(
                    web_items,
                    empty_msg="No web results yet.",
                )
        else:
            st.info(
                "No discover items yet. Ingest may still be running in the background."
            )
    else:
        st.info("Select a project from Home to open Discover.")
        if st.button("Go to Home", key="discover_go_home"):
            st.session_state.current_page = "Home"
            st.rerun()
