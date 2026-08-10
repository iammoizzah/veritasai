import httpx
from dotenv import load_dotenv

load_dotenv()

WIKI_API = "https://en.wikipedia.org/api/rest_v1"


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """Search Wikipedia for relevant articles."""
    try:
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "search", "list": "search",
                "srsearch": query, "srlimit": max_results,
                "format": "json", "utf8": 1
            },
            timeout=10
        )
        data = r.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "").replace(
                '<span class="searchmatch">', "").replace("</span>", "")
            results.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": snippet,
                "page_id": item.get("pageid"),
                "source": "wikipedia"
            })
        return results
    except Exception as e:
        return [{"title": "Wikipedia failed", "url": "", "snippet": str(e), "source": "wikipedia"}]


def get_wikipedia_summary(title: str) -> str:
    """Get full summary of a Wikipedia article."""
    try:
        r = httpx.get(
            f"{WIKI_API}/page/summary/{title.replace(' ', '_')}",
            timeout=10
        )
        return r.json().get("extract", "")[:2000]
    except Exception as e:
        return f"Could not fetch: {e}"


def search_wiki_evidence(query: str) -> list[dict]:
    """Search Wikipedia and enrich with full summaries."""
    results = search_wikipedia(query, max_results=2)
    for r in results:
        if r.get("page_id"):
            summary = get_wikipedia_summary(r["title"])
            if summary:
                r["snippet"] = summary
    return results
