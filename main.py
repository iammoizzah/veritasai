import os
import time
from dotenv import load_dotenv
from src.agents.decomposer import decompose_claim
from src.agents.verifier import collect_evidence, verify_sub_claim
from src.agents.synthesizer import synthesize_verdict
from src.tools.vector_store import (
    store_verified_claim,
    find_similar_claims,
    get_collection_stats
)

load_dotenv()


def run_pipeline(claim: str, verbose: bool = True) -> dict:
    """Run the full VeritasAI verification pipeline with RAG caching."""
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERITASAI — Claim Verification Pipeline")
        print(f"{'='*60}")
        print(f"Claim: {claim}\n")

    # ── RAG: Check cache first ────────────────────────────────────────────────
    if verbose:
        print("Checking claim cache...")
    similar = find_similar_claims(claim, threshold=0.88)
    if similar:
        cached = similar[0]
        if verbose:
            print(
                f"  CACHE HIT — Similar claim found (similarity: {cached['similarity']})")
            print(f"  Cached verdict: {cached['verdict']}")
            print(f"  Original claim: {cached['claim'][:70]}...")
        return {
            "final_verdict": cached["verdict"],
            "confidence_score": int(cached.get("confidence", 0)),
            "executive_summary": f"This claim is semantically similar to a previously verified claim (similarity: {cached['similarity']}). Cached verdict: {cached['verdict']}.",
            "detailed_explanation": f"Similar claim: {cached['claim']}",
            "key_facts": [],
            "misleading_elements": [],
            "verdict_color": "gray",
            "sources": [],
            "sub_claim_count": 0,
            "overall_score": 0,
            "original_claim": claim,
            "processing_time": round(time.time() - start_time, 2),
            "cache_hit": True,
            "similar_claim": cached["claim"]
        }

    if verbose:
        print("  No cache hit — running full pipeline\n")

    # ── Step 1: Decompose ─────────────────────────────────────────────────────
    if verbose:
        print("Step 1/3 — Decomposing claim...")
    decomposed = decompose_claim(claim)
    sub_claims = decomposed.get("sub_claims", [])
    if verbose:
        print(f"  Type: {decomposed.get('claim_type')}")
        print(f"  Entities: {decomposed.get('entities')}")
        print(f"  Sub-claims: {len(sub_claims)}\n")

    # ── Step 2: Collect evidence + verify ─────────────────────────────────────
    if verbose:
        print("Step 2/3 — Collecting evidence (web + news + Wikipedia)...")
    sub_claim_results = []
    for sc in sub_claims:
        if verbose:
            print(f"  Verifying: {sc['text'][:65]}...")
        evidence = collect_evidence(sc)
        result = verify_sub_claim(evidence)
        sub_claim_results.append(result)
        if verbose:
            print(
                f"  → {result['verdict']} (confidence: {result.get('confidence', 0)}%)")

    # ── Step 3: Synthesize ────────────────────────────────────────────────────
    if verbose:
        print(f"\nStep 3/3 — Synthesizing final verdict...")
    final = synthesize_verdict(claim, sub_claim_results)

    elapsed = round(time.time() - start_time, 2)
    final["processing_time"] = elapsed
    final["original_claim"] = claim
    final["decomposed"] = decomposed
    final["sub_claim_results"] = sub_claim_results
    final["cache_hit"] = False

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    try:
        claim_id = store_verified_claim(claim, final)
        final["claim_id"] = claim_id
        if verbose:
            print(f"  Stored in cache with ID: {claim_id}")
    except Exception as e:
        if verbose:
            print(f"  Cache store failed: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERDICT:    {final['final_verdict']}")
        print(f"Confidence: {final['confidence_score']}%")
        print(f"Time:       {elapsed}s")
        print(f"\nSummary: {final['executive_summary']}")
        print(f"{'='*60}\n")

    return final


if __name__ == "__main__":
    # Show DB stats
    stats = get_collection_stats()
    print(f"Cache: {stats['total_claims']} claims stored\n")

    # Test claims
    test_claims = [
        "Pakistan is the 5th most populous country in the world",
        "The Eiffel Tower was built in 1887 and is located in Berlin",
        "Pakistan has the world's largest salt mine",
    ]

    for claim in test_claims:
        result = run_pipeline(claim)
        cache_status = "CACHED" if result.get("cache_hit") else "FRESH"
        print(
            f"[{cache_status}] {result['final_verdict']} in {result['processing_time']}s\n")

    # Show final stats
    stats = get_collection_stats()
    print(f"\nCache now: {stats['total_claims']} claims stored")
