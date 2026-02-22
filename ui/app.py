#!/usr/bin/env python3
"""
Streamlit Chat UI for the Arabic RAG system.
Run: streamlit run project/ui/app.py
"""

import streamlit as st
import sys
import os
import json
import time

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from rag.rag_pipeline import ask, ask_llm_only
from rag.ollama_client import list_models

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Page Config ───
st.set_page_config(
    page_title="نظام RAG — أنتيخريستوس",
    page_icon="📖",
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

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
}

.main-header h1 {
    font-size: 2rem;
    margin: 0;
    background: linear-gradient(90deg, #e94560, #f5a623);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #ccc;
    margin-top: 0.3rem;
}

.chat-message {
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    direction: rtl;
    text-align: right;
    line-height: 1.8;
}

.user-msg {
    background: linear-gradient(135deg, #0f3460, #16213e);
    color: #e0e0e0;
    border-left: 4px solid #e94560;
}

.bot-msg {
    background: linear-gradient(135deg, #1a1a2e, #0d1b2a);
    color: #f0f0f0;
    border-left: 4px solid #4ecdc4;
}

.source-chip {
    display: inline-block;
    background: #16213e;
    color: #4ecdc4;
    padding: 2px 10px;
    border-radius: 16px;
    margin: 2px 4px;
    font-size: 0.85rem;
    border: 1px solid #4ecdc4;
}

.latency-badge {
    display: inline-block;
    background: rgba(233, 69, 96, 0.15);
    color: #e94560;
    padding: 3px 12px;
    border-radius: 16px;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

.stTextInput > div > div > input {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>📖 نظام RAG — أنتيخريستوس</h1>
    <p>اسأل أي سؤال عن كتاب أنتيخريستوس وسيتم الإجابة باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")

    cfg = load_config()

    # Model selector
    try:
        available = list_models()
    except Exception:
        available = cfg.get("models", [])

    chosen = cfg.get("chosen_model") or (available[0] if available else "qwen2.5:3b")
    selected_model = st.selectbox("اختر النموذج", available, index=available.index(chosen) if chosen in available else 0)

    # Mode toggle
    use_rag = st.toggle("استخدام RAG (السياق من الكتاب)", value=True)

    # Show sources toggle
    show_sources = st.toggle("إظهار المصادر", value=True)

    # Top-K
    top_k = st.slider("عدد المقاطع المسترجعة", 1, 3, 1)

    st.markdown("---")

    # Chat history
    st.markdown("## 📜 سجل المحادثة")
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

    if "messages" in st.session_state:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"**👤** {msg['content'][:50]}...")

# ─── Chat State ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-msg">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-msg">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True)
        if show_sources and msg.get("sources"):
            chips = " ".join([f'<span class="source-chip">مقطع {s}</span>' for s in msg["sources"]])
            st.markdown(f"📎 المصادر: {chips}", unsafe_allow_html=True)
        if msg.get("latency"):
            st.markdown(f'<span class="latency-badge">⏱️ {msg["latency"]}</span>',
                        unsafe_allow_html=True)

# ─── Chat Input ───
user_input = st.chat_input("اكتب سؤالك هنا...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="chat-message user-msg">👤 {user_input}</div>',
                unsafe_allow_html=True)

    with st.spinner("جاري التفكير... 🤔"):
        try:
            # Pass previous messages (excluding the current prompt we just appended)
            chat_history = st.session_state.messages[:-1]
            
            if use_rag:
                from rag.rag_pipeline import ask_stream
                stream, sources, ret_time, start_time = ask_stream(user_input, model=selected_model, top_k=top_k, history=chat_history)
            else:
                from rag.rag_pipeline import ask_llm_only_stream
                stream, sources, ret_time, start_time = ask_llm_only_stream(user_input, model=selected_model)

            # Stream output live
            msg_container = st.empty()
            full_answer = ""
            for chunk in stream:
                full_answer += chunk
                msg_container.markdown(f'<div class="chat-message bot-msg">🤖 {full_answer}</div>', unsafe_allow_html=True)

            gen_time = round(time.time() - start_time, 3)
            latency_str = f"استرجاع: {ret_time}ث | توليد: {gen_time}ث"
            answer = full_answer
        except Exception as e:
            answer = f"❌ خطأ: {str(e)}"
            sources = []
            latency_str = ""
            msg_container = st.empty()
            msg_container.markdown(f'<div class="chat-message bot-msg">{answer}</div>', unsafe_allow_html=True)

    # Add bot message to session state
    bot_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "latency": latency_str,
    }
    st.session_state.messages.append(bot_msg)

    if show_sources and sources:
        chips = " ".join([f'<span class="source-chip">مقطع {s}</span>' for s in sources])
        st.markdown(f"📎 المصادر: {chips}", unsafe_allow_html=True)
    if latency_str:
        st.markdown(f'<span class="latency-badge">⏱️ {latency_str}</span>',
                    unsafe_allow_html=True)

