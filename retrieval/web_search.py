import os
import logging
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# logging setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "summarizer.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = os.getenv("TAVILY_API_KEY", "")
_tavily: Optional[TavilyClient] = None

if API_KEY:
    try:
        _tavily = TavilyClient(api_key=API_KEY)
    except Exception as e:
        logging.error(f"Failed to init TavilyClient: {e}")
else:
    logging.warning("TAVILY_API_KEY not set; Tavily search disabled.")

def _retry(n=3, base=0.4):
    """Simple exponential backoff decorator."""
    def deco(fn):
        def wrap(*a, **k):
            last = None
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception as e:
                    last = e
                    logging.warning(f"{fn.__name__} failed (attempt {i+1}/{n}): {e}")
                    time.sleep(base * (2 ** i))
            logging.error(f"{fn.__name__} exhausted retries: {last}")
            return []
        return wrap
    return deco

@_retry(n=3, base=0.4)
def fetch_articles(query: str, k: int = 3) -> List[Dict]:
    """
    Search provider returns up to k items with {title, url, content}.
    Always returns a list, logs warnings when unavailable.
    """
    if not _tavily:
        logging.warning("Tavily client unavailable: check if API key is set correctly.")
        return []

    resp = _tavily.search(query=query, max_results=k, search_depth="basic")
    results = resp.get("results", []) if isinstance(resp, dict) else []
    items = [{
        "title": r.get("title", "") or "",
        "url": r.get("url", "") or "",
        "content": r.get("content", "") or "",
    } for r in results[:k]]

    logging.info(f"Tavily query='{query}' returned {len(items)} result(s).")
    return items