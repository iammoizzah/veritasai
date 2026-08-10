import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

POLITICAL_BIAS_SPECTRUM = {
    "far_left": ["jacobin", "counterpunch", "wsws.org"],
    "left": ["msnbc", "vox", "huffpost", "guardian", "buzzfeed"],
    "center_left": ["nytimes", "washingtonpost", "cnn", "bbc", "npr"],
    "center": ["reuters", "apnews", "bloomberg", "axios"],
    "center_right": ["wsj", "economist", "ft.com"],
    "right": ["foxnews", "nypost", "breitbart"],
    "far_right": ["infowars", "dailystormer", "gatewaypundit"]
}


def detect_source_bias(url: str) -> str:
    """Quick lookup for known source bias."""
    url_lower = url.lower()
    for bias_level, domains in POLITICAL_BIAS_SPECTRUM.items():
        if any(d in url_lower for d in domains):
            return bias_level
    return "unknown"


@traceable(name="bias_detector")
def analyze_bias(claim: str, evidence_list: list[dict]) -> dict:
    """
    Detect framing bias, loaded language, and source diversity
    across all evidence collected for a claim.
    """
    # Build source bias profile
    source_biases = []
    for e in evidence_list:
        url = e.get("url", "")
        bias = detect_source_bias(url)
        source_biases.append({
            "url": url,
            "source_type": e.get("source", ""),
            "detected_bias": bias
        })

    # Analyze content for framing bias
    snippets = "\n\n".join([
        f"Source: {e.get('url', e.get('source', ''))}\nContent: {e.get('snippet', '')[:300]}"
        for e in evidence_list[:6]
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system="""You are a media bias analyst. Analyze evidence sources for framing bias,
loaded language, and source diversity.

Return ONLY valid JSON:
{
  "overall_bias_risk": "low|medium|high",
  "bias_types_detected": ["political_framing", "loaded_language", "omission_bias"],
  "source_diversity_score": 0-100,
  "dominant_narrative": "one sentence describing the dominant framing",
  "alternative_perspectives_missing": ["perspective 1", "perspective 2"],
  "loaded_language_examples": ["example phrase 1", "example phrase 2"],
  "bias_warning": "one sentence warning if high bias detected, else empty string",
  "recommendation": "how to get more balanced evidence"
}""",
        messages=[{"role": "user", "content": f"""Claim being verified: {claim}

Evidence collected:
{snippets}

Source bias profile:
{json.dumps(source_biases, indent=2)}

Analyze for bias."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        result["source_bias_breakdown"] = source_biases
        return result
    except Exception:
        return {
            "overall_bias_risk": "unknown",
            "bias_types_detected": [],
            "source_diversity_score": 50,
            "dominant_narrative": "Could not analyze bias",
            "alternative_perspectives_missing": [],
            "loaded_language_examples": [],
            "bias_warning": "",
            "recommendation": "Manually verify sources",
            "source_bias_breakdown": source_biases
        }
