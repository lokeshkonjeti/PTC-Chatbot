import os
from typing import List

import streamlit as st
from database_setup import pinecone_setup  # uses Gemini + Pinecone via LlamaIndex

FALLBACK_NOTE = (
    "Note: I am just an AI agent and I do not have full information about this, "
    "please try to contact the customer care for more information."
)

def _extract_source_urls(response, max_sources: int = 5) -> List[str]:
    """Pull website links from LlamaIndex response.source_nodes metadata."""
    urls: List[str] = []
    try:
        source_nodes = getattr(response, "source_nodes", None) or []
        for sn in source_nodes:
            node = getattr(sn, "node", sn)
            meta = getattr(node, "metadata", None) or getattr(node, "extra_info", None) or {}
            for k in ("source", "url", "URL", "link"):
                val = meta.get(k)
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    if val not in urls:
                        urls.append(val)
                        if len(urls) >= max_sources:
                            return urls
    except Exception:
        pass
    return urls[:max_sources]

def _format_response_with_fallback(response) -> str:
    """Render response text; append fallback note when the answer looks thin."""
    text = getattr(response, "response", None)
    if text is None:
        try:
            text = str(response)
        except Exception:
            text = ""
    text = (text or "").strip()

    low_info_phrases = [
        "i don't know", "i do not know", "i am not sure", "can't find", "cannot find",
        "no information", "not enough information", "unable to", "insufficient information", "sorry,"
    ]
    has_low_info_phrase = any(p in text.lower() for p in low_info_phrases)
    source_nodes = getattr(response, "source_nodes", None)
    no_sources = isinstance(source_nodes, (list, tuple)) and len(source_nodes) == 0
    too_short = len(text) < 120

    if has_low_info_phrase or no_sources or too_short:
        return f"{text}\n\n{FALLBACK_NOTE}" if text else FALLBACK_NOTE
    return text

@st.cache_resource(show_spinner=False)
def _connect_index(index_name: str, gemini_key: str, pinecone_key: str):
    index = pinecone_setup(
        chunk_size=512,
        t_dimensions=768,           # text-embedding-004 => 768
        gemini_key=gemini_key,
        api_key=pinecone_key,
        index_name=index_name,
        documents=None             # query-only (no indexing)
    )
    return index

# ---------------- UI ----------------

st.set_page_config(page_title="PTC Chatbot", page_icon="🛣️", layout="wide")
st.title("🛣️ PTC Chatbot")

with st.sidebar:
    st.header("Configuration")
    gemini_key = st.text_input("GOOGLE_API_KEY (Gemini)", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    pinecone_key = st.text_input("PINECONE_API_KEY (Pinecone)", value=os.getenv("PINECONE_API_KEY", ""), type="password")
    index_name = st.text_input("Pinecone Index Name", value=os.getenv("INDEX_NAME", "gemini-chatbot-3"))
    top_k = st.slider("Top K (similarity)", min_value=1, max_value=20, value=10, step=1)

if not gemini_key or not pinecone_key:
    st.warning("Add your **GOOGLE_API_KEY** and **PINECONE_API_KEY** in the sidebar to continue.")
    st.stop()

with st.spinner("Connecting to index…"):
    index = _connect_index(index_name=index_name, gemini_key=gemini_key, pinecone_key=pinecone_key)

# chat history
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of (role, text)

query = st.chat_input("Ask me about PA Turnpike…")
if query:
    st.session_state["history"].append(("user", query))
    try:
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        resp = query_engine.query(query)
        answer_text = _format_response_with_fallback(resp)
        sources = _extract_source_urls(resp)
        if sources:
            src_md = "\n".join(f"- [{u}]({u})" for u in sources)
            answer_text = f"{answer_text}\n\n**Sources**\n{src_md}"
        st.session_state["history"].append(("assistant", answer_text))
    except Exception as e:
        st.session_state["history"].append(("assistant", f"Sorry, I hit an error: `{e}`\n\n{FALLBACK_NOTE}"))

# render
for role, text in st.session_state["history"]:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(text)

st.caption("Model: Gemini via LlamaIndex • Vector DB: Pinecone • Site: paturnpike.com")
