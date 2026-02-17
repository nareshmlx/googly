"""
Beauty Social AI - Clean Chat Interface
Simple, stable implementation with Streamlit native components.
"""

import streamlit as st
import uuid
from pathlib import Path
from agent import (
    create_agent,
    get_database,
    get_chat_history,
    get_session_state_data,
    update_session_state_data,
)
from tools.ensemble_tiktok_tool import get_discovered_videos, clear_discovered_videos

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Beauty Social AI",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ============ SESSION STATE INITIALIZATION ============
# Use URL query params to persist session_id across refreshes
query_params = st.query_params

# Initialize user_id
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

# Get session_id from URL or create new one
if "session_id" not in st.session_state:
    if "sid" in query_params:
        st.session_state.session_id = query_params["sid"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params["sid"] = st.session_state.session_id

# Track if we've loaded persisted data this session
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Initialize messages - load from database on first run
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load persisted data on first run (after refresh)
if not st.session_state.data_loaded:
    # Load chat history from database
    history = get_chat_history(
        st.session_state.session_id,
        st.session_state.user_id
    )
    if history:
        st.session_state.messages = history
    
    # Load discovered videos from session state
    state_data = get_session_state_data(
        st.session_state.session_id,
        st.session_state.user_id
    )
    if "discovered_videos" in state_data:
        st.session_state.discovered_videos = state_data["discovered_videos"]
    else:
        st.session_state.discovered_videos = []
    
    st.session_state.data_loaded = True

# Initialize remaining session state
if "sessions" not in st.session_state:
    # Don't load from database on startup (causes Agno errors with old sessions)
    # Sessions are populated locally during app usage
    st.session_state.sessions = {}

if "enable_tiktok" not in st.session_state:
    st.session_state.enable_tiktok = True
if "enable_openalex" not in st.session_state:
    st.session_state.enable_openalex = True
if "discovered_videos" not in st.session_state:
    st.session_state.discovered_videos = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"


# ============ HELPER FUNCTIONS ============
def new_chat():
    """Start a new chat session."""
    # Save discovered videos before switching
    if st.session_state.discovered_videos:
        update_session_state_data(
            st.session_state.session_id,
            st.session_state.user_id,
            {"discovered_videos": st.session_state.discovered_videos}
        )
    
    if st.session_state.messages:
        # Save current session to local state
        title = st.session_state.messages[0]["content"][:30] + "..."
        st.session_state.sessions[st.session_state.session_id] = {
            "title": title,
            "messages": st.session_state.messages.copy()
        }
    
    # Create new session
    new_sid = str(uuid.uuid4())
    st.session_state.session_id = new_sid
    st.session_state.messages = []
    st.session_state.discovered_videos = []
    st.session_state.data_loaded = True  # Mark as loaded (empty is valid)
    
    # Update URL query param
    st.query_params["sid"] = new_sid


def load_session(sid):
    """Load a previous session from database."""
    # Save current discovered videos first
    if st.session_state.discovered_videos:
        update_session_state_data(
            st.session_state.session_id,
            st.session_state.user_id,
            {"discovered_videos": st.session_state.discovered_videos}
        )
    
    # Save current session if has messages
    if st.session_state.messages:
        title = st.session_state.messages[0]["content"][:30] + "..."
        st.session_state.sessions[st.session_state.session_id] = {
            "title": title,
            "messages": st.session_state.messages.copy()
        }
    
    # Load selected session from database
    st.session_state.session_id = sid
    history = get_chat_history(sid, st.session_state.user_id)
    st.session_state.messages = history if history else []
    
    # Load discovered videos for this session
    state_data = get_session_state_data(sid, st.session_state.user_id)
    st.session_state.discovered_videos = state_data.get("discovered_videos", [])
    
    # Update URL
    st.query_params["sid"] = sid


def sync_videos():
    """Sync discovered videos from tool and persist to database."""
    new_videos_added = False
    for v in get_discovered_videos():
        if not any(x['video_id'] == v['video_id'] for x in st.session_state.discovered_videos):
            st.session_state.discovered_videos.append(v)
            new_videos_added = True
    
    # Persist to database if new videos were added
    if new_videos_added and st.session_state.discovered_videos:
        update_session_state_data(
            st.session_state.session_id,
            st.session_state.user_id,
            {"discovered_videos": st.session_state.discovered_videos}
        )


# ============ SIDEBAR ============
with st.sidebar:
    st.title("💄 Beauty Social")
    st.caption("AI-powered beauty insights")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Chat", "Discover", "History"],
        label_visibility="collapsed"
    )
    st.session_state.current_page = page
    
    st.divider()
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()
    
    st.divider()
    
    # Tool toggle
    st.caption("TOOLS")
    st.session_state.enable_tiktok = st.toggle(
        "🎵 TikTok Search",
        value=st.session_state.enable_tiktok,
        help="Enable TikTok hashtag search"
    )
    st.session_state.enable_openalex = st.toggle(
        "📚 OpenAlex Search",
        value=st.session_state.enable_openalex,
        help="Enable OpenAlex research paper search"
    )
    
    st.divider()
    
    # Database status
    db = get_database()
    if db:
        st.success("🧠 Memory Active")
    else:
        st.warning("⚠️ Memory Offline")
    
    st.caption(f"📊 {len(st.session_state.discovered_videos)} videos discovered")


# ============ MAIN CONTENT ============

# --- CHAT PAGE ---
if st.session_state.current_page == "Chat":
    st.header("💬 Chat")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about beauty trends, search TikTok..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            
            agent = create_agent(
                st.session_state.enable_tiktok,
                st.session_state.enable_openalex,
                st.session_state.user_id,
                st.session_state.session_id
            )
            
            try:
                for event in agent.run(prompt):
                    if hasattr(event, 'content') and event.content:
                        response += event.content
                        placeholder.markdown(response + "▌")
                placeholder.markdown(response)
                sync_videos()
            except Exception as e:
                response = f"⚠️ Error: {e}"
                placeholder.error(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# --- DISCOVER PAGE ---
elif st.session_state.current_page == "Discover":
    st.header("🔍 Discover")
    st.caption("TikTok videos from your searches")
    
    sync_videos()
    
    if st.session_state.discovered_videos:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.discovered_videos = []
                clear_discovered_videos()
                st.rerun()
        
        # Video grid
        cols = st.columns(3)
        for i, video in enumerate(st.session_state.discovered_videos):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"**@{video['author_username']}**")
                    if video.get('video_url'):
                        st.video(video['video_url'])
                    elif video.get('cover_url'):
                        st.image(video['cover_url'])
                    st.caption(video.get('description', '')[:60] + "...")
                    st.caption(f"Date: {video.get('upload_date', 'Unknown')}")
                    st.write(f"❤️ {video.get('likes', 0):,} | 👀 {video.get('views', 0):,}")
                    st.divider()
    else:
        st.info("💡 Enable TikTok Search and ask about beauty hashtags to discover videos!")


# --- HISTORY PAGE ---
elif st.session_state.current_page == "History":
    st.header("📚 History")
    
    if st.session_state.sessions:
        for sid, data in st.session_state.sessions.items():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.subheader(data['title'])
                st.caption(f"{len(data['messages'])} messages")
            with col2:
                if st.button("Open", key=f"open_{sid}"):
                    load_session(sid)
                    st.session_state.current_page = "Chat"
                    st.rerun()
            st.divider()
    else:
        st.info("No chat history yet. Start a conversation!")
