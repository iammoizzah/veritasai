import json
import re
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# Source credibility tiers
CREDIBILITY_TIERS = {
    "tier_1": {
        "sources": ["reuters.com", "apnews.com", "bbc.com", "nature.com",
                    "science.org", "who.int", "un.org", "gov", "edu", "wikipedia.org"],
        "score": 90,
        "label": "High credibility"
    },
    "tier_2": {
        "sources": ["nytimes.com", "washingtonpost.com", "theguardian.com",
                    "wsj.com", "economist.com", "ft.com", "npr.org", "pbs.org"],
        "score": 75,
        "label": "Generally reliable"
    },
    "tier_3": {
        "sources": ["huffpost.com", "foxnews.com", "msnbc.com", "cnn.com",
                    "dailymail.co.uk", "nypost.com"],
        "score": 55,
        "label": "Mixed reliability"
    },
}


def score_source_credibility(url: str) -> dict:
    """Score a source URL for credibility."""
    if not url:
        return {"score": 50, "label": "Unknown", "url": url}
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        for tier_name, tier_data in CREDIBILITY_TIERS.items():
            for source in tier_data["sources"]:
                if source in domain:
                    return {"score": tier_data["score"],
                            "label": tier_data["label"],
                            "url": url, "domain": domain, "tier": tier_name}
        return {"score": 50, "label": "Unknown source", "url": url, "domain": domain, "tier": "tier_4"}
    except Exception:
        return {"score": 50, "label": "Could not assess", "url": url}


@traceable(name="credibility_scorer")
def score_evidence_credibility(sub_claim_results: list[dict]) -> dict:
    """Score the overall credibility of evidence used across all sub-claims."""
    all_sources = []
    for result in sub_claim_results:
        key_source = result.get("key_source", "")
        if key_source:
            score = score_source_credibility(key_source)
            all_sources.append(score)

    if not all_sources:
        return {
            "average_credibility": 50,
            "credibility_label": "Unknown",
            "source_scores": [],
            "high_credibility_ratio": 0,
            "credibility_adjustment": -10
        }

    avg_score = sum(s["score"] for s in all_sources) / len(all_sources)
    high_cred = sum(1 for s in all_sources if s["score"] >= 75)
    high_cred_ratio = high_cred / len(all_sources)

    # Confidence adjustment based on source quality
    if avg_score >= 80:
        adjustment = 5
        label = "High credibility sources"
    elif avg_score >= 65:
        adjustment = 0
        label = "Moderately credible sources"
    elif avg_score >= 50:
        adjustment = -10
        label = "Mixed credibility sources"
    else:
        adjustment = -20
        label = "Low credibility sources"

    return {
        "average_credibility": round(avg_score, 1),
        "credibility_label": label,
        "source_scores": all_sources,
        "high_credibility_ratio": round(high_cred_ratio, 2),
        "credibility_adjustment": adjustment
    }
