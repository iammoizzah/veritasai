import json
import re
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# Credibility tiers for known domains
CREDIBILITY_TIERS = {
    "tier_1_high": [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nature.com", "science.org", "pubmed.ncbi", "who.int",
        "un.org", "gov", "edu", "wikipedia.org", "britannica.com",
        "bloomberg.com", "ft.com", "economist.com"
    ],
    "tier_2_medium": [
        "nytimes.com", "washingtonpost.com", "guardian.com",
        "wsj.com", "cnn.com", "npr.org", "axios.com",
        "forbes.com", "businessinsider.com", "theverge.com",
        "wired.com", "techcrunch.com", "dawn.com", "geo.tv"
    ],
    "tier_3_low": [
        "reddit.com", "quora.com", "medium.com",
        "blogspot.com", "wordpress.com", "tumblr.com"
    ],
    "tier_4_unreliable": [
        "infowars.com", "naturalnews.com", "beforeitsnews.com",
        "yournewswire.com", "worldnewsdailyreport.com"
    ]
}


def score_domain(url: str) -> dict:
    """Score a domain's credibility based on known tiers."""
    url_lower = url.lower()
    for tier, domains in CREDIBILITY_TIERS.items():
        if any(d in url_lower for d in domains):
            scores = {
                "tier_1_high": 90,
                "tier_2_medium": 65,
                "tier_3_low": 35,
                "tier_4_unreliable": 10
            }
            return {"tier": tier, "score": scores[tier], "url": url}
    return {"tier": "unknown", "score": 50, "url": url}


def extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', url)
    return match.group(1) if match else url


@traceable(name="credibility_scorer")
def score_source_credibility(evidence_list: list[dict]) -> dict:
    """
    Score the overall credibility of evidence sources.
    Returns weighted credibility score and per-source breakdown.
    """
    scored_sources = []
    for e in evidence_list:
        url = e.get("url", "")
        source_type = e.get("source", "web")
        if not url and source_type == "wikipedia":
            url = "wikipedia.org"
        scored = score_domain(url)
        scored["source_type"] = source_type
        scored["title"] = e.get("title", "")[:80]
        scored_sources.append(scored)

    # Calculate weighted average (tier_1 sources count more)
    if scored_sources:
        total_weight = 0
        weighted_sum = 0
        for s in scored_sources:
            tier_weights = {
                "tier_1_high": 3,
                "tier_2_medium": 2,
                "tier_3_low": 1,
                "tier_4_unreliable": 0.5,
                "unknown": 1
            }
            w = tier_weights.get(s["tier"], 1)
            weighted_sum += s["score"] * w
            total_weight += w
        overall_score = round(
            weighted_sum / total_weight) if total_weight > 0 else 50
    else:
        overall_score = 0

    tier_1_count = sum(1 for s in scored_sources if s["tier"] == "tier_1_high")
    tier_4_count = sum(
        1 for s in scored_sources if s["tier"] == "tier_4_unreliable")

    if overall_score >= 75:
        credibility_level = "HIGH"
        credibility_note = "Evidence comes from highly credible sources."
    elif overall_score >= 55:
        credibility_level = "MEDIUM"
        credibility_note = "Evidence comes from moderately credible sources. Cross-check recommended."
    elif overall_score >= 35:
        credibility_level = "LOW"
        credibility_note = "Evidence sources have questionable credibility. Treat verdict with caution."
    else:
        credibility_level = "VERY_LOW"
        credibility_note = "Evidence sources are unreliable. Verdict should not be trusted."

    return {
        "overall_credibility_score": overall_score,
        "credibility_level": credibility_level,
        "credibility_note": credibility_note,
        "tier_1_sources": tier_1_count,
        "unreliable_sources": tier_4_count,
        "total_sources_scored": len(scored_sources),
        "source_breakdown": scored_sources,
        "has_primary_sources": tier_1_count > 0
    }


@traceable(name="credibility_analyst")
def analyze_credibility_with_llm(claim: str, credibility_data: dict,
                                 bias_data: dict) -> dict:
    """
    Use LLM to give a holistic credibility assessment combining
    source credibility and bias analysis.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        system="""You are a credibility analyst. Given source credibility scores and bias data,
provide a final credibility assessment.

Return ONLY valid JSON:
{
  "trust_score": 0-100,
  "trust_label": "HIGHLY_TRUSTWORTHY|TRUSTWORTHY|QUESTIONABLE|UNRELIABLE",
  "key_concerns": ["concern 1", "concern 2"],
  "confidence_adjustment": -20 to +10,
  "analyst_note": "2-3 sentence assessment for the end user"
}""",
        messages=[{"role": "user", "content": f"""Claim: {claim}

Source credibility:
- Overall score: {credibility_data.get('overall_credibility_score')}/100
- Level: {credibility_data.get('credibility_level')}
- Tier 1 sources: {credibility_data.get('tier_1_sources')}
- Unreliable sources: {credibility_data.get('unreliable_sources')}

Bias analysis:
- Bias risk: {bias_data.get('overall_bias_risk')}
- Source diversity: {bias_data.get('source_diversity_score')}/100
- Bias warning: {bias_data.get('bias_warning', 'None')}

Provide holistic credibility assessment."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "trust_score": credibility_data.get("overall_credibility_score", 50),
            "trust_label": "QUESTIONABLE",
            "key_concerns": [],
            "confidence_adjustment": 0,
            "analyst_note": credibility_data.get("credibility_note", "")
        }
