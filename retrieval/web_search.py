import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
_tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def fetch_three_articles(query: str):
    """
    Returns exactly three items with {title, url, content} using Tavily.
    """
    resp = _tavily.search(query=query, max_results=3)
    items = []
    for r in resp.get("results", [])[:3]:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    # If Tavily returns fewer than 3, just return what we have (simple v1 behavior)
    return items
