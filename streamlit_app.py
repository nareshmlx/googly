import streamlit as st
import settings  # Load environment variables
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIResponses
from agno.db.postgres import PostgresDb
from tools.ensemble_tiktok_tool import search_tiktok_hashtag, get_discovered_videos, clear_discovered_videos

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Beauty Social Chatbot",
    page_icon="💄",
    layout="wide"
)

# Custom CSS for larger tabs
st.markdown("""
<style>
    /* Make tab labels bigger */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Make tab panel have more padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    /* Style tab buttons */
    .stTabs [data-baseweb="tab-list"] button {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize PostgreSQL database for memory (with error handling)
@st.cache_resource
def get_database():
    """Create and cache the PostgreSQL database connection."""
    try:
        db = PostgresDb(db_url=settings.DATABASE_URL)
        return db
    except Exception as e:
        st.warning(f"⚠️ Could not connect to PostgreSQL: {e}. Memory will not be persisted.")
        return None

# Initialize the Agno agent with model and memory
def get_agent(enable_tiktok_search=False, user_id=None, session_id=None):
    """Create the Agno agent with optional TikTok search tool and memory."""
    tools = []
    if enable_tiktok_search:
        tools.append(search_tiktok_hashtag)
    
    db = get_database()
    
    return Agent(
        name="beauty-social-chatbot",
        model=OpenAIResponses(id="gpt-4o-mini"),
        instructions=(
            "You are a helpful and friendly AI assistant specialized in beauty and social media trends. "
            "Provide clear, concise, and accurate responses to user questions. "
            "Remember details about the user from previous conversations to provide personalized responses."
            + ("\n\nYou have access to a TikTok search tool. When users ask about TikTok content, "
               "trends, or want to search for videos by hashtag, use the search_tiktok_hashtag tool." 
               if enable_tiktok_search else "")
        ),
        tools=tools if tools else None,
        markdown=True,
        stream=True,
        # Memory configuration
        db=db,
        user_id=user_id,
        session_id=session_id,
        update_memory_on_run=True,  # Automatically update memory after each run
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "enable_tiktok" not in st.session_state:
    st.session_state.enable_tiktok = False
if "discovered_videos" not in st.session_state:
    st.session_state.discovered_videos = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Sync discovered videos from tool module
def sync_discovered_videos():
    """Sync discovered videos from the tool module to session state."""
    tool_videos = get_discovered_videos()
    for video in tool_videos:
        if not any(v['video_id'] == video['video_id'] for v in st.session_state.discovered_videos):
            st.session_state.discovered_videos.append(video)

# Create main tabs
tab_chat, tab_discover = st.tabs(["💬 Chat", "🔍 Discover"])

# ============ CHAT TAB ============
with tab_chat:
    st.title("💄 Beauty Social Chatbot")
    st.caption("Your AI assistant for beauty trends and social media insights")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get the agent with optional TikTok search and memory
            agent = get_agent(
                enable_tiktok_search=st.session_state.enable_tiktok,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            )
            
            # Stream the response
            try:
                response = agent.run(prompt)
                
                for event in response:
                    if hasattr(event, 'content') and event.content:
                        full_response += event.content
                        message_placeholder.markdown(full_response + "▌")
                
                # Display final response without cursor
                message_placeholder.markdown(full_response)
                
                # Sync discovered videos after response
                sync_discovered_videos()
                
            except Exception as e:
                full_response = f"⚠️ Error: {str(e)}"
                message_placeholder.error(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ============ DISCOVER TAB ============
with tab_discover:
    st.title("🔍 Discover")
    st.caption("Videos discovered from your TikTok searches")
    
    # Sync videos first
    sync_discovered_videos()
    
    # Show videos if any
    if st.session_state.discovered_videos:
        # Clear button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear All", key="clear_discover"):
                st.session_state.discovered_videos = []
                clear_discovered_videos()
                st.rerun()
        
        with col1:
            st.write(f"**{len(st.session_state.discovered_videos)} videos discovered**")
        
        st.divider()
        
        # Display videos in a grid
        cols = st.columns(2)
        for idx, video in enumerate(st.session_state.discovered_videos):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"**@{video['author_username']}** - {video['author_name']}")
                    
                    # Display video if URL exists
                    if video.get('video_url'):
                        st.video(video['video_url'])
                    elif video.get('cover_url'):
                        st.image(video['cover_url'], use_container_width=True)
                    
                    # Video info
                    st.markdown(f"_{video['description'][:100]}..._" if len(video.get('description', '')) > 100 else f"_{video.get('description', '')}_")
                    
                    # Stats
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("❤️", f"{video.get('likes', 0):,}")
                    with stat_cols[1]:
                        st.metric("💬", f"{video.get('comments', 0):,}")
                    with stat_cols[2]:
                        st.metric("📤", f"{video.get('shares', 0):,}")
                    with stat_cols[3]:
                        st.metric("👀", f"{video.get('views', 0):,}")
                    
                    # Hashtag source
                    st.caption(f"🏷️ From #{video.get('hashtag', 'unknown')}")
                    st.divider()
    else:
        st.info(
            "No videos discovered yet!\n\n"
            "💡 **How to discover videos:**\n"
            "1. Enable TikTok Search in the sidebar\n"
            "2. Go to the Chat tab\n"
            "3. Ask something like: *'Search TikTok for #skincare videos'*\n"
            "4. Videos will appear here automatically!"
        )

# ============ SIDEBAR ============
with st.sidebar:
    st.header("⚙️ Settings")
    
    # TikTok Search Toggle
    tiktok_enabled = st.toggle(
        "Enable TikTok Search",
        value=st.session_state.enable_tiktok,
        help="Enable the agent to search TikTok videos by hashtag"
    )
    
    # Update session state if toggle changed
    if tiktok_enabled != st.session_state.enable_tiktok:
        st.session_state.enable_tiktok = tiktok_enabled
        st.rerun()
    
    # Show warning if TikTok is enabled but token is missing
    if tiktok_enabled and not settings.ENSEMBLE_API_TOKEN:
        st.warning("⚠️ ENSEMBLE_API_TOKEN not set in .env")
    
    st.divider()
    
    # Memory status
    st.header("🧠 Memory")
    db = get_database()
    if db:
        st.success("✅ Connected to PostgreSQL")
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    else:
        st.warning("⚠️ Memory not available")
        st.caption("Start PostgreSQL to enable memory")
    
    # New Session button
    if st.button("🔄 New Session"):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats
    st.header("📊 Stats")
    st.metric("Videos Discovered", len(st.session_state.discovered_videos))
    st.metric("Chat Messages", len(st.session_state.messages))
    
    st.divider()
    
    st.header("About")
    st.info(
        "Your AI-powered assistant for:\n\n"
        "- 💬 Beauty advice and recommendations\n"
        "- 📱 Social media trend analysis\n"
        "- 🧠 Remembers your preferences"
        + ("\n- 🎵 TikTok video search" if tiktok_enabled else "")
    )
