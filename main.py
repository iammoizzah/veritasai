import os
import time
from dotenv import load_dotenv
from src.agents.decomposer import decompose_claim
from src.agents.verifier import collect_evidence, verify_sub_claim
from src.agents.synthesizer import synthesize_verdict
from src.agents.bias_detector import analyze_bias
from src.agents.credibility import score_source_credibility, analyze_credibility_with_llm
from src.tools.vector_store import (
    store_verified_claim, find_similar_claims, get_collection_stats
)

load_dotenv()


def run_pipeline(claim: str, verbose: bool = True) -> dict:
    """Full VeritasAI pipeline: decompose → evidence → verify → bias → credibility → synthesize."""
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERITASAI — Claim Verification Pipeline")
        print(f"{'='*60}")
        print(f"Claim: {claim}\n")

    # ── Cache check ───────────────────────────────────────────────────────────
    if verbose:
        print("Checking semantic cache...")
    similar = find_similar_claims(claim, threshold=0.88)
    if similar:
        cached = similar[0]
        if verbose:
            print(f"  CACHE HIT (similarity: {cached['similarity']})")
            print(f"  Cached verdict: {cached['verdict']}\n")
        return {
            "final_verdict": cached["verdict"],
            "confidence_score": int(cached.get("confidence", 0)),
            "executive_summary": f"Cached result (similarity {cached['similarity']}): {cached['verdict']}",
            "detailed_explanation": f"Similar claim already verified: {cached['claim']}",
            "key_facts": [], "misleading_elements": [],
            "verdict_color": "gray", "sources": [],
            "sub_claim_count": 0, "overall_score": 0,
            "original_claim": claim,
            "processing_time": round(time.time() - start_time, 2),
            "cache_hit": True, "similar_claim": cached["claim"]
        }

    if verbose:
        print("  No cache hit — running full pipeline\n")

    # ── Step 1: Decompose ─────────────────────────────────────────────────────
    if verbose:
        print("Step 1/5 — Decomposing claim...")
    decomposed = decompose_claim(claim)
    sub_claims = decomposed.get("sub_claims", [])
    if verbose:
        print(f"  Type: {decomposed.get('claim_type')} | "
              f"Entities: {decomposed.get('entities')} | "
              f"Sub-claims: {len(sub_claims)}\n")

    # ── Step 2: Collect evidence + verify ─────────────────────────────────────
    if verbose:
        print("Step 2/5 — Collecting evidence (web + news + Wikipedia + RAG)...")
    sub_claim_results = []
    all_evidence = []

    for sc in sub_claims:
        if verbose:
            print(f"  Verifying: {sc['text'][:65]}...")
        evidence = collect_evidence(sc)
        all_evidence.extend(evidence.get("evidence", []))
        result = verify_sub_claim(evidence)
        sub_claim_results.append(result)
        if verbose:
            print(f"  → {result['verdict']} "
                  f"(confidence: {result.get('confidence', 0)}%, "
                  f"sources: {result.get('sources_used', [])})")

    # ── Step 3: Bias analysis ─────────────────────────────────────────────────
    if verbose:
        print(f"\nStep 3/5 — Analyzing source bias...")
    bias_data = analyze_bias(claim, all_evidence)
    if verbose:
        print(f"  Bias risk: {bias_data.get('overall_bias_risk')}")
        print(
            f"  Source diversity: {bias_data.get('source_diversity_score')}/100")
        if bias_data.get("bias_warning"):
            print(f"  ⚠ {bias_data['bias_warning']}")

    # ── Step 4: Credibility scoring ───────────────────────────────────────────
    if verbose:
        print(f"\nStep 4/5 — Scoring source credibility...")
    credibility_data = score_source_credibility(all_evidence)
    credibility_assessment = analyze_credibility_with_llm(
        claim, credibility_data, bias_data)
    if verbose:
        print(f"  Credibility: {credibility_data.get('credibility_level')} "
              f"({credibility_data.get('overall_credibility_score')}/100)")
        print(f"  Trust: {credibility_assessment.get('trust_label')}")
        print(f"  Tier-1 sources: {credibility_data.get('tier_1_sources')}")

    # ── Step 5: Synthesize ────────────────────────────────────────────────────
    if verbose:
        print(f"\nStep 5/5 — Synthesizing final verdict...")
    final = synthesize_verdict(claim, sub_claim_results)

    # Apply credibility confidence adjustment
    confidence_adj = credibility_assessment.get("confidence_adjustment", 0)
    final["confidence_score"] = max(0, min(100,
                                           final.get("confidence_score", 0) + confidence_adj))

    # Attach all metadata
    elapsed = round(time.time() - start_time, 2)
    final.update({
        "processing_time": elapsed,
        "original_claim": claim,
        "decomposed": decomposed,
        "sub_claim_results": sub_claim_results,
        "bias_analysis": bias_data,
        "credibility_data": credibility_data,
        "credibility_assessment": credibility_assessment,
        "cache_hit": False
    })

    # Store in cache
    try:
        claim_id = store_verified_claim(claim, final)
        final["claim_id"] = claim_id
    except Exception as e:
        if verbose:
            print(f"  Cache store failed: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERDICT:     {final['final_verdict']}")
        print(f"Confidence:  {final['confidence_score']}%")
        print(f"Trust:       {credibility_assessment.get('trust_label')}")
        print(f"Bias Risk:   {bias_data.get('overall_bias_risk')}")
        print(f"Time:        {elapsed}s")
        print(f"\nSummary: {final['executive_summary']}")
        print(f"Analyst: {credibility_assessment.get('analyst_note', '')}")
        print(f"{'='*60}\n")

    return final


if __name__ == "__main__":
    stats = get_collection_stats()
    print(f"Cache: {stats['total_claims']} claims stored\n")

    test_claims = [
        "Pakistan is the 5th most populous country in the world",
        "Climate change is causing more frequent wildfires globally",
        "The Eiffel Tower was built in 1887 and is in Berlin",
    ]

    for claim in test_claims:
        result = run_pipeline(claim)
        status = "CACHED" if result.get("cache_hit") else "FRESH"
        print(f"[{status}] {result['final_verdict']} | "
              f"Trust: {result.get('credibility_assessment', {}).get('trust_label', 'N/A')} | "
              f"{result['processing_time']}s\n")

    stats = get_collection_stats()
    print(f"Cache: {stats['total_claims']} claims stored")
