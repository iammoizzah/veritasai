import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv
from src.tools.search import search_evidence
from src.tools.vector_store import get_relevant_evidence

load_dotenv()
client = anthropic.Anthropic()


@traceable(name="evidence_collector")
def collect_evidence(sub_claim: dict) -> dict:
    """Collect evidence from web, news, Wikipedia + RAG store."""
    all_evidence = []

    # Live search
    for query in sub_claim.get("search_queries", [])[:2]:
        results = search_evidence(query, use_wikipedia=True)
        all_evidence.extend(results)

    # RAG: pull relevant chunks from past verifications
    rag_chunks = get_relevant_evidence(sub_claim["text"], n_results=3)
    for chunk in rag_chunks:
        all_evidence.append({
            "title": "Previously retrieved evidence",
            "url": "",
            "snippet": chunk,
            "source": "rag_cache"
        })

    return {
        "sub_claim_id": sub_claim["id"],
        "sub_claim_text": sub_claim["text"],
        "evidence": all_evidence[:10],
        "evidence_count": len(all_evidence),
        "sources_used": list(set(e.get("source", "") for e in all_evidence))
    }


@traceable(name="sub_claim_verifier")
def verify_sub_claim(sub_claim_with_evidence: dict) -> dict:
    """Verify a single sub-claim against collected evidence."""
    evidence_text = "\n\n".join([
        f"[{i+1}] [{e.get('source', '').upper()}] {e.get('title', '')}\n"
        f"URL: {e.get('url', '')}\n{e.get('snippet', '')[:400]}"
        for i, e in enumerate(sub_claim_with_evidence.get("evidence", [])[:8])
    ])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system="""You are a rigorous fact-checker. Verify claims against evidence only.
Do not use prior knowledge — only what the evidence shows.

Verdicts:
- TRUE: Evidence clearly supports the claim
- FALSE: Evidence clearly contradicts the claim
- MISLEADING: Technically true but missing critical context
- PARTIALLY_TRUE: Some aspects correct, others incorrect
- UNVERIFIABLE: Insufficient evidence

Return ONLY valid JSON:
{
  "verdict": "TRUE|FALSE|MISLEADING|PARTIALLY_TRUE|UNVERIFIABLE",
  "confidence": 0-100,
  "reasoning": "2-3 sentences citing specific evidence",
  "supporting_evidence": ["point 1", "point 2"],
  "contradicting_evidence": ["contra 1"],
  "key_source": "most credible source URL"
}""",
        messages=[{"role": "user", "content": f"""Sub-claim: {sub_claim_with_evidence['sub_claim_text']}

Evidence (from web, news, Wikipedia, and cached sources):
{evidence_text}

Verify strictly against this evidence only."""}]
    )

    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        result["sub_claim_id"] = sub_claim_with_evidence["sub_claim_id"]
        result["sub_claim_text"] = sub_claim_with_evidence["sub_claim_text"]
        result["sources_used"] = sub_claim_with_evidence.get(
            "sources_used", [])
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
            "key_source": "",
            "sources_used": []
        }
