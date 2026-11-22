#!/usr/bin/env python3
"""
Streamlit chat UI: upload documents, ask questions, get answers grounded in them.
Chats live only in the browser session and are gone when it closes.
Run: streamlit run project/ui/app.py
"""

import streamlit as st
import sys
import os
import json
import time
import uuid
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from rag.rag_pipeline import ask, ask_llm_only
from rag.ollama_client import list_models
from rag.pdf_indexer import build_memory_index

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chat_sessions():
    # Privacy: nothing is stored on disk; chats exist only in this session
    return []


def save_chat_sessions(sessions):
    pass


def create_new_session():
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "messages": []
    }


# ─── Page Config ───
st.set_page_config(
    page_title="Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Noto Naskh Arabic', serif;
}
/* Ensure Streamlit Icons (which use text ligatures) retain their icon font */
.material-symbols-rounded, 
[data-testid="stIconMaterial"], 
.stIcon, 
[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* ── Hide Streamlit Top Menu and Deploy options carefully ── */
#MainMenu {display: none !important;}
.stAppDeployButton {display: none !important;}
/* Target the internal settings and GitHub links of the toolbar, but NOT the toolbar itself, which sometimes holds the sidebar expander in RTL mode */
[data-testid="stToolbar"] [data-testid="stToolbarActions"] {display: none !important;}

/* ── Force Highly Visible Collapsed Sidebar Button ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 9999999 !important;
    color: #ffffff !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    width: 28px !important;
    height: 28px !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] button:hover {
    background: rgba(255, 255, 255, 0.2) !important;
}

/* ── Completely Restore Native Streamlit Sidebar Buttons ── */
/* ── Sidebar Buttons (Generic overrides for Streamlit) ── */

/* ── Delete button OVERRIDE ── */
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(2) div[class*="stButton"] button,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) div[class*="stButton"] button {
    background: transparent !important;
    border: none !important;
    color: #888 !important;
    font-size: 15px !important;
    border-radius: 6px !important;
    height: 38px !important;
    min-height: 0 !important;
    width: 100% !important;
    padding: 0 !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(2) div[class*="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) div[class*="stButton"] button:hover {
    background: rgba(255, 60, 60, 0.15) !important;
    color: #ff4a4a !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(2) div[class*="stButton"] button p,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(2) div[class*="stButton"] button p {
    text-align: center !important;
    width: 100% !important;
    margin: 0 !important;
}

/* ── New Chat button OVERRIDE (using primary kind or type) ── */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] div[class*="stButton"] button:has(p:contains("New Chat")),
[data-testid="stSidebar"] button[kind="primary"] {
    background: transparent !important;
    color: #ececec !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: background 0.2s, border-color 0.2s !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}
[data-testid="stSidebar"] button[kind="primary"] p {
    font-size: 15px !important;
    margin: 0 !important;
}

/* ── Chat session buttons OVERRIDE (Main row columns) ── */
/* Specifically target the FIRST column in horizontal blocks containing chat sessions */
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(1) div[class*="stButton"] button,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) div[class*="stButton"] button,
[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 8px 12px !important;
    text-align: left !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
    height: 38px !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(1) div[class*="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) div[class*="stButton"] button:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(1) div[class*="stButton"] button p,
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) div[class*="stButton"] button p,
[data-testid="stSidebar"] button[kind="secondary"] p {
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(1) div[class*="stButton"] button div[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="column"]:nth-of-type(1) div[class*="stButton"] button div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
}

/* ── Model combo input & Disable Search Typing ── */
.model-combo input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #ddd !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] input {
    caret-color: transparent !important;
    cursor: pointer !important;
    user-select: none !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    cursor: pointer !important;
}

.main-header {
    text-align: center;
    color: white;
    margin-top: 10vh;
    margin-bottom: 5vh;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 500;
}

.chat-message {
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    text-align: start;
    line-height: 1.8;
}

.user-msg {
    background: #2f2f2f;
    color: #e0e0e0;
    width: fit-content;
    margin-left: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

.bot-msg {
    background: transparent;
    color: #e0e0e0;
    white-space: pre-wrap;
    word-break: break-word;
}

.source-chip {
    display: inline-block;
    background: #1e1e1e;
    color: #aaa;
    padding: 2px 10px;
    border-radius: 16px;
    margin: 2px 4px;
    font-size: 0.85rem;
    border: 1px solid #444;
}

.latency-badge {
    display: inline-block;
    color: #888;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

.stTextInput > div > div > input {
    direction: ltr;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ─── Chat State ───
if "chat_sessions" not in st.session_state:
    loaded = load_chat_sessions()
    # Clean up stale empty chats and legacy English-titled chats
    cleaned = [s for s in loaded if s["messages"] and s["title"] not in ("New Chat", "محادثة جديدة")]
    if not cleaned:
        cleaned = [create_new_session()]
    else:
        # Always prepend a fresh empty chat at the top
        cleaned.insert(0, create_new_session())
    save_chat_sessions(cleaned)
    st.session_state.chat_sessions = cleaned

if "current_session" not in st.session_state:
    st.session_state.current_session = st.session_state.chat_sessions[0]["id"]

# Helper to get current session index
def get_current_session_index():
    for i, s in enumerate(st.session_state.chat_sessions):
        if s["id"] == st.session_state.current_session:
            return i
    return 0

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### Chats")

    # New Chat button — styled
    if st.button("+ New Chat", use_container_width=True, key="new_chat_btn", type="primary"):
        # Prevent spamming new empty chats
        if st.session_state.chat_sessions[0]["messages"]:
            new_sess = create_new_session()
            st.session_state.chat_sessions.insert(0, new_sess)
            st.session_state.current_session = new_sess["id"]
            save_chat_sessions(st.session_state.chat_sessions)
            st.rerun()
        else:
            # If the top chat is already empty, just switch to it
            st.session_state.current_session = st.session_state.chat_sessions[0]["id"]
            st.rerun()

    # Chat list — skip empty "New Chat" sessions to avoid duplication
    for session in st.session_state.chat_sessions:
        # Don't show empty "New Chat" in the list (the button above handles that)
        if not session["messages"] and session["title"] == "New Chat":
            continue

        sid   = session["id"]
        title = session["title"]
        col1, col2 = st.columns([7, 1])

        with col1:
            if st.button(title, key=f"sel_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()

        with col2:
            if st.button("🗑", key=f"del_{sid}", help="Delete Chat", use_container_width=True):
                st.session_state.chat_sessions = [
                    s for s in st.session_state.chat_sessions if s["id"] != sid
                ]
                if not st.session_state.chat_sessions:
                    new_sess = create_new_session()
                    st.session_state.chat_sessions = [new_sess]
                    st.session_state.current_session = new_sess["id"]
                elif st.session_state.current_session == sid:
                    st.session_state.current_session = st.session_state.chat_sessions[0]["id"]
                save_chat_sessions(st.session_state.chat_sessions)
                st.rerun()

    st.markdown("---")
    st.markdown("**Settings**")

    cfg = load_config()
    try:
        available = list_models()
    except Exception:
        available = cfg.get("models", [])

    chosen = cfg.get("chosen_model") or (available[0] if available else "qwen2.5:3b")

    # Model selector: dropdown only, no text input
    if not available:
        available = [chosen]
    default_idx = available.index(chosen) if chosen in available else 0
    selected_model = st.selectbox("Model", available, index=default_idx)

    use_rag = st.toggle("Use RAG Sources", value=True)
    show_sources = st.toggle("Show Sources", value=True)
    top_k = st.slider("Number of Sources", 1, 8, cfg.get("top_k_retrieval", 5))

    # ─── PDF Upload ───
    st.markdown("---")
    st.markdown("**📄 Documents**")

    uploaded = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Any PDFs, in any language. They stay in memory and are never saved.",
        key="pdf_uploader",
    )

    names = tuple(sorted(f.name for f in uploaded)) if uploaded else ()
    if names and names != st.session_state.get("loaded_docs"):
        with st.spinner("Reading documents..."):
            try:
                doc_index = build_memory_index([(f.name, f.getvalue()) for f in uploaded])
                st.session_state["doc_index"] = doc_index
                st.session_state["loaded_docs"] = names
            except Exception as e:
                st.session_state.pop("doc_index", None)
                st.session_state.pop("loaded_docs", None)
                st.error(f"Could not read the PDF: {e}")
    elif not names:
        st.session_state.pop("doc_index", None)
        st.session_state.pop("loaded_docs", None)

    if st.session_state.get("doc_index"):
        n_docs = len(st.session_state["loaded_docs"])
        st.success(f"{n_docs} document{'s' if n_docs > 1 else ''} loaded, "
                   f"{st.session_state['doc_index']['chunk_count']} chunks")
    else:
        st.caption("No documents yet. Upload a PDF to start.")


def render_sources(sources):
    """Show the exact passages the answer was grounded in."""
    if not sources:
        return
    st.markdown("**Sources**")
    for n, c in enumerate(sources, 1):
        with st.expander(f"[{n}] {c.get('source', 'Chunk ' + str(c.get('chunk_id')))}"):
            st.write(c.get("text", ""))


# Display chat history
current_idx = get_current_session_index()
current_messages = st.session_state.chat_sessions[current_idx]["messages"]

if not current_messages:
    st.markdown("""
    <div class="main-header">
        <h1>How can I help you today?</h1>
    </div>
    """, unsafe_allow_html=True)

for msg in current_messages:
    if msg["role"] == "user":
        st.markdown(f'<div dir="auto" class="chat-message user-msg">{msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div dir="auto" class="chat-message bot-msg">{msg["content"]}</div>',
                    unsafe_allow_html=True)
        if show_sources and msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("latency"):
            st.markdown(f'<span class="latency-badge">{msg["latency"]}</span>',
                        unsafe_allow_html=True)

# ─── Chat Input ───
user_input = st.chat_input("Ask me anything...")

if user_input:
    current_idx = get_current_session_index()
    curr_session = st.session_state.chat_sessions[current_idx]
    
    if not curr_session["messages"] or curr_session["title"] == "New Chat":
        curr_session["title"] = user_input[:25] + ("..." if len(user_input) > 25 else "")
    
    curr_session["messages"].append({"role": "user", "content": user_input})
    save_chat_sessions(st.session_state.chat_sessions)
    
    st.markdown(f'<div dir="auto" class="chat-message user-msg">{user_input}</div>',
                unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        try:
            chat_history = curr_session["messages"][:-1]
            d_index = st.session_state.get("doc_index")
            if use_rag and not d_index:
                raise ValueError("Upload a PDF in the sidebar first, or turn off Use RAG Sources.")
            if use_rag:
                from rag.rag_pipeline import ask_stream
                stream, chunks, ret_time, start_time = ask_stream(
                    user_input, model=selected_model, top_k=top_k,
                    history=chat_history,
                    dynamic_index=d_index,
                    doc_names=d_index["doc_names"],
                )
                sources = [{"chunk_id": c["chunk_id"], "source": c.get("source", ""),
                            "text": c["text"]} for c in chunks]
            else:
                from rag.rag_pipeline import ask_llm_only_stream
                stream, sources, ret_time, start_time = ask_llm_only_stream(user_input, model=selected_model)

            msg_container = st.empty()
            full_answer = ""
            for chunk in stream:
                full_answer += chunk
                msg_container.markdown(f'<div dir="auto" class="chat-message bot-msg">{full_answer}</div>', unsafe_allow_html=True)

            if use_rag:
                from rag.rag_pipeline import ensure_language
                fixed = ensure_language(full_answer, user_input, selected_model)
                if fixed != full_answer:
                    full_answer = fixed
                    msg_container.markdown(f'<div dir="auto" class="chat-message bot-msg">{full_answer}</div>', unsafe_allow_html=True)

            gen_time = round(time.time() - start_time, 3)
            latency_str = f"Retrieval: {ret_time}s | Generation: {gen_time}s"
            answer = full_answer
        except Exception as e:
            answer = str(e)
            sources = []
            latency_str = ""
            msg_container = st.empty()
            msg_container.markdown(f'<div dir="auto" class="chat-message bot-msg">{answer}</div>', unsafe_allow_html=True)

    bot_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "latency": latency_str,
    }
    
    curr_session["messages"].append(bot_msg)
    save_chat_sessions(st.session_state.chat_sessions)

    if show_sources and sources:
        render_sources(sources)
    if latency_str:
        st.markdown(f'<span class="latency-badge">{latency_str}</span>',
                    unsafe_allow_html=True)
