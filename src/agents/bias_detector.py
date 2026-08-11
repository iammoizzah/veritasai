import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

KNOWN_BIAS_DOMAINS = {
    "foxnews.com": "right",
    "breitbart.com": "far-right",
    "msnbc.com": "left",
    "motherjones.com": "left",
    "huffpost.com": "left",
    "theblaze.com": "far-right",
    "bbc.com": "center",
    "reuters.com": "center",
    "apnews.com": "center",
    "npr.org": "center-left",
    "wsj.com": "center-right",
    "nytimes.com": "center-left",
    "washingtonpost.com": "center-left",
    "theguardian.com": "center-left",
    "economist.com": "center-right",
}


def get_domain_bias(url: str) -> str:
    """Get known bias rating for a domain."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        return KNOWN_BIAS_DOMAINS.get(domain, "unknown")
    except Exception:
        return "unknown"


@traceable(name="bias_detector")
def detect_bias(claim: str, evidence_list: list[dict]) -> dict:
    """
    Analyze evidence sources for political/ideological bias.
    Returns bias assessment and adjusted confidence.
    """
    # Check domain biases
    source_biases = []
    for e in evidence_list:
        url = e.get("url", "") or e.get("key_source", "")
        if url:
            bias = get_domain_bias(url)
            source_biases.append({"url": url, "bias": bias})

    bias_summary = "\n".join([
        f"- {s['url'][:60]}: {s['bias']}" for s in source_biases if s["bias"] != "unknown"
    ])

    evidence_text = "\n".join([
        f"[{i+1}] {e.get('sub_claim_text', e.get('title', ''))}: {e.get('verdict', e.get('snippet', ''))[:200]}"
        for i, e in enumerate(evidence_list[:5])
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        system="""You are a media bias analyst. Assess whether the evidence used to verify
a claim shows systematic bias that could affect the verdict.

Return ONLY valid JSON:
{
  "overall_bias_direction": "left|right|center|mixed|unknown",
  "bias_severity": "none|low|moderate|high",
  "bias_assessment": "2 sentences about the bias pattern in sources",
  "confidence_adjustment": -20 to 0,
  "diverse_sources": true|false,
  "recommendation": "one sentence on how to improve source diversity"
}""",
        messages=[{"role": "user", "content": f"""Claim: {claim}

Source bias ratings:
{bias_summary if bias_summary else "No known bias data for these sources"}

Evidence used:
{evidence_text}

Assess bias in the evidence base."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        result["source_biases"] = source_biases
        return result
    except Exception:
        return {
            "overall_bias_direction": "unknown",
            "bias_severity": "none",
            "bias_assessment": "Could not assess bias.",
            "confidence_adjustment": 0,
            "diverse_sources": True,
            "recommendation": "Use diverse sources for better coverage.",
            "source_biases": source_biases
        }
