import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# logging setup
logging.basicConfig(
    filename="summarizer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = os.getenv("TAVILY_API_KEY", "")
_tavily = None

if API_KEY:
    try:
        _tavily = TavilyClient(api_key=API_KEY)
    except Exception as e:
        logging.error(f"Failed to init TavilyClient: {e}")
else:
    logging.warning("TAVILY_API_KEY not set; Tavily search disabled.")


def fetch_three_articles(query: str):
    """
    Returns up to three items with {title, url, content} using Tavily.
    Adds basic error handling so failures don't crash the app.
    """
    if not _tavily:
        logging.warning("Tavily client unavailable; returning empty list.")
        return []

    try:
        resp = _tavily.search(query=query, max_results=3, search_depth="basic")
        results = resp.get("results", []) if isinstance(resp, dict) else []
        items = [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        } for r in results[:3]]

        logging.info(f"Tavily query='{query}' returned {len(items)} results.")
        return items

    except Exception as e:
        logging.error(f"Tavily search failed for query='{query}': {e}")
        return []
