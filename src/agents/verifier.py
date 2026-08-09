import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv
from src.tools.search import search_evidence

load_dotenv()
client = anthropic.Anthropic()


@traceable(name="evidence_collector")
def collect_evidence(sub_claim: dict) -> dict:
    """Collect evidence for a single sub-claim."""
    all_evidence = []
    for query in sub_claim.get("search_queries", [])[:2]:
        results = search_evidence(query)
        all_evidence.extend(results)
    return {
        "sub_claim_id": sub_claim["id"],
        "sub_claim_text": sub_claim["text"],
        "evidence": all_evidence[:8],
        "evidence_count": len(all_evidence)
    }


@traceable(name="sub_claim_verifier")
def verify_sub_claim(sub_claim_with_evidence: dict) -> dict:
    """Verify a single sub-claim against collected evidence."""
    evidence_text = "\n\n".join([
        f"[{i+1}] {e.get('title', '')}\nSource: {e.get('url', '')}\n{e.get('snippet', '')}"
        for i, e in enumerate(sub_claim_with_evidence.get("evidence", [])[:6])
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system="""You are a fact-checking expert. Verify claims against evidence strictly.

Verdict options:
- TRUE: Evidence clearly supports the claim
- FALSE: Evidence clearly contradicts the claim
- MISLEADING: Claim is technically true but missing important context
- PARTIALLY_TRUE: Some aspects correct, others incorrect
- UNVERIFIABLE: Insufficient evidence to make a determination

Return ONLY valid JSON:
{
  "verdict": "TRUE|FALSE|MISLEADING|PARTIALLY_TRUE|UNVERIFIABLE",
  "confidence": 0-100,
  "reasoning": "2-3 sentences explaining the verdict",
  "supporting_evidence": ["evidence point 1", "evidence point 2"],
  "contradicting_evidence": ["contra point 1"],
  "key_source": "most important source URL"
}""",
        messages=[{"role": "user", "content": f"""Sub-claim: {sub_claim_with_evidence['sub_claim_text']}

Evidence:
{evidence_text}

Verify this sub-claim against the evidence provided."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        result["sub_claim_id"] = sub_claim_with_evidence["sub_claim_id"]
        result["sub_claim_text"] = sub_claim_with_evidence["sub_claim_text"]
        return result
    except Exception:
        return {
            "sub_claim_id": sub_claim_with_evidence["sub_claim_id"],
            "sub_claim_text": sub_claim_with_evidence["sub_claim_text"],
            "verdict": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "Could not parse verification result.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "key_source": ""
        }
