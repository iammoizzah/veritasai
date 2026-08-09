import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

VERDICT_WEIGHTS = {
    "TRUE": 1.0, "PARTIALLY_TRUE": 0.6,
    "MISLEADING": 0.3, "UNVERIFIABLE": 0.5, "FALSE": 0.0
}


@traceable(name="verdict_synthesizer")
def synthesize_verdict(original_claim: str, sub_claim_results: list) -> dict:
    """Synthesize individual sub-claim results into a final verdict."""

    # Calculate weighted confidence score
    if sub_claim_results:
        weighted_scores = []
        for r in sub_claim_results:
            weight = VERDICT_WEIGHTS.get(r.get("verdict", "UNVERIFIABLE"), 0.5)
            confidence = r.get("confidence", 50) / 100
            weighted_scores.append(weight * confidence)
        overall_score = sum(weighted_scores) / len(weighted_scores)
    else:
        overall_score = 0.5

    # Preliminary verdict from weighted score
    if overall_score >= 0.8:
        preliminary = "TRUE"
    elif overall_score >= 0.6:
        preliminary = "PARTIALLY_TRUE"
    elif overall_score >= 0.4:
        preliminary = "MISLEADING"
    elif overall_score >= 0.2:
        preliminary = "UNVERIFIABLE"
    else:
        preliminary = "FALSE"

    results_summary = "\n".join([
        f"- {r['sub_claim_id']}: {r['verdict']} "
        f"(confidence: {r.get('confidence', 0)}%) — {r['sub_claim_text']}"
        for r in sub_claim_results
    ])

    all_sources = list(set([
        r.get("key_source", "")
        for r in sub_claim_results
        if r.get("key_source")
    ]))

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system="""You are a senior fact-checker writing the final verdict.
Synthesize all sub-claim results into a clear, concise final assessment.
Return ONLY valid JSON:
{
  "final_verdict": "TRUE|FALSE|MISLEADING|PARTIALLY_TRUE|UNVERIFIABLE",
  "confidence_score": 0-100,
  "executive_summary": "2-3 sentences for a general audience",
  "detailed_explanation": "4-5 sentences with specific evidence and reasoning",
  "key_facts": ["important fact 1", "fact 2", "fact 3"],
  "misleading_elements": ["element 1 if any"],
  "verdict_color": "green|red|orange|yellow|gray"
}""",
        messages=[{"role": "user", "content": f"""Original claim: {original_claim}

Sub-claim results:
{results_summary}

Preliminary verdict: {preliminary} (score: {overall_score:.2f})
Sources: {', '.join(all_sources[:5])}

Synthesize the final verdict."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        result["sources"] = all_sources
        result["sub_claim_count"] = len(sub_claim_results)
        result["overall_score"] = round(overall_score, 3)
        return result
    except Exception:
        return {
            "final_verdict": preliminary,
            "confidence_score": int(overall_score * 100),
            "executive_summary": "Verification completed.",
            "detailed_explanation": results_summary,
            "key_facts": [],
            "misleading_elements": [],
            "verdict_color": "gray",
            "sources": all_sources,
            "sub_claim_count": len(sub_claim_results),
            "overall_score": round(overall_score, 3)
        }
