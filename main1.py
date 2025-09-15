import argparse
from crawler import get_subpath_links
from loader import load_documents
from database_setup import pinecone_setup

ROOT_URL = "https://www.paturnpike.com/"
ALLOWED_PREFIX = "https://www.paturnpike.com/"

GEMINI_KEY = "AIzaSyALgWqLHIYtD0lnrMK5zEHOwOLiaVy1_bg"
PINECONE_KEY = "pcsk_582ogW_3DJHwLo1krLpWMP26dNbPkWMyUrin62LQLzqfc5D9eB5WYGB5KMXRh9cC62H5ec"
INDEX_NAME = "gemini-chatbot-3"

FALLBACK_NOTE = (
    "Note: I am just an AI agent and I do not have full information about this, "
    "please try to contact the customer care for more information."
)

def _extract_source_urls(response, max_sources=5):
    """Pull website links from LlamaIndex response.source_nodes metadata."""
    urls = []
    try:
        source_nodes = getattr(response, "source_nodes", None) or []
        for sn in source_nodes:
            node = getattr(sn, "node", sn)  # NodeWithScore.node or node itself
            meta = getattr(node, "metadata", None) or getattr(node, "extra_info", None) or {}
            # Common keys where loaders store URLs
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
    """Return response text and append a customer-care note when information is sparse."""
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
        if text:
            return f"{text}\n\n{FALLBACK_NOTE}"
        else:
            return FALLBACK_NOTE
    return text

def run_indexing():
    urls = get_subpath_links(ROOT_URL, ALLOWED_PREFIX)
    documents_parsed = load_documents(urls)
    _ = pinecone_setup(
        chunk_size=512,
        t_dimensions=768,
        gemini_key=GEMINI_KEY,
        api_key=PINECONE_KEY,
        index_name=INDEX_NAME,
        documents=documents_parsed
    )
    print("Indexing complete.")

def run_query():
    index = pinecone_setup(
        chunk_size=512,
        t_dimensions=768,
        gemini_key=GEMINI_KEY,
        api_key=PINECONE_KEY,
        index_name=INDEX_NAME,
        documents=None
    )
    query_engine = index.as_query_engine(similarity_top_k=10)
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        try:
            response = query_engine.query(q)
            answer = _format_response_with_fallback(response)

            # Attach source links (if any)
            sources = _extract_source_urls(response)
            if sources:
                answer += "\n\nSources:\n" + "\n".join(f"- {u}" for u in sources)

            print("PTC Agent:", answer)
        except Exception as e:
            print("PTC Agent:", f"Sorry, I ran into an issue: {e}\n\n{FALLBACK_NOTE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "query"], required=True)
    args = parser.parse_args()

    if args.mode == "index":
        run_indexing()
    elif args.mode == "query":
        run_query()
