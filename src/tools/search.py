import os
from duckduckgo_search import DDGS
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

news_client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY", ""))


def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Search web for evidence using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": r.get("body", ""), "source": "web"} for r in results]
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e), "source": "web"}]


def news_search(query: str, max_results: int = 5) -> list[dict]:
    """Search recent news articles for evidence."""
    try:
        response = news_client.get_everything(
            q=query, language="en", sort_by="relevancy", page_size=max_results
        )
        articles = response.get("articles", [])
        return [{"title": a.get("title", ""), "url": a.get("url", ""),
                 "snippet": a.get("description", "") or a.get("content", "")[:300],
                 "source": "news", "published": a.get("publishedAt", "")} for a in articles]
    except Exception as e:
        return [{"title": "News search failed", "url": "", "snippet": str(e), "source": "news"}]


def search_evidence(query: str) -> list[dict]:
    """Combined search — web + news."""
    web = web_search(query, max_results=4)
    news = news_search(query, max_results=3)
    return web + news
