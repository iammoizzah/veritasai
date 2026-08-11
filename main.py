import os
import time
from dotenv import load_dotenv
from src.agents.decomposer import decompose_claim
from src.agents.verifier import collect_evidence, verify_sub_claim
from src.agents.synthesizer import synthesize_verdict
from src.agents.bias_detector import detect_bias
from src.agents.credibility import score_evidence_credibility
from src.agents.hf_classifier import run_hf_enrichment
from src.tools.vector_store import (
    store_verified_claim, find_similar_claims, get_collection_stats
)

load_dotenv()


def run_pipeline(claim: str, verbose: bool = True, use_hf: bool = True) -> dict:
    """
    Full VeritasAI pipeline:
    Cache check → Decompose → Collect Evidence → Verify →
    Bias Detect → Credibility Score → HF Enrich → Synthesize → Cache Store
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("VERITASAI — Full Pipeline (Days 1-5)")
        print(f"{'='*60}")
        print(f"Claim: {claim}\n")

    # ── RAG Cache Check ───────────────────────────────────────────────────────
    if verbose:
        print("Checking semantic cache...")
    similar = find_similar_claims(claim, threshold=0.88)
    if similar:
        cached = similar[0]
        if verbose:
            print(f"  CACHE HIT (similarity: {cached['similarity']}) "
                  f"→ {cached['verdict']}\n")
        return {
            "final_verdict": cached["verdict"],
            "confidence_score": int(cached.get("confidence", 0)),
            "executive_summary": f"Cached result (similarity: {cached['similarity']}): {cached['verdict']}",
            "detailed_explanation": f"Similar claim previously verified: {cached['claim']}",
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

    # ── Step 2: Collect Evidence + Verify ────────────────────────────────────
    if verbose:
        print("Step 2/5 — Collecting evidence (web + news + Wikipedia + RAG)...")
    sub_claim_results = []
    all_evidence_snippets = []
    for sc in sub_claims:
        if verbose:
            print(f"  Verifying: {sc['text'][:65]}...")
        evidence = collect_evidence(sc)
        result = verify_sub_claim(evidence)
        sub_claim_results.append(result)
        all_evidence_snippets.extend([
            e.get("snippet", "") for e in evidence.get("evidence", [])[:3]
        ])
        if verbose:
            srcs = result.get("sources_used", [])
            print(f"  → {result['verdict']} ({result.get('confidence', 0)}%) "
                  f"| Sources: {srcs}")

    # ── Step 3: Bias Detection ────────────────────────────────────────────────
    if verbose:
        print(f"\nStep 3/5 — Detecting source bias...")
    bias_result = detect_bias(claim, sub_claim_results)
    if verbose:
        print(f"  Bias: {bias_result.get('overall_bias_direction')} "
              f"({bias_result.get('bias_severity')}) "
              f"| Adjustment: {bias_result.get('confidence_adjustment', 0)}")

    # ── Step 4: Source Credibility ────────────────────────────────────────────
    if verbose:
        print(f"\nStep 4/5 — Scoring source credibility...")
    credibility_result = score_evidence_credibility(sub_claim_results)
    if verbose:
        print(f"  Avg credibility: {credibility_result.get('average_credibility')} "
              f"| {credibility_result.get('credibility_label')} "
              f"| Adjustment: {credibility_result.get('credibility_adjustment', 0)}")

    # ── Step 4b: HuggingFace Enrichment ──────────────────────────────────────
    hf_enrichment = {}
    if use_hf:
        if verbose:
            print(
                f"\n  Running HuggingFace enrichment (zero-shot + NER + summarization)...")
        try:
            hf_enrichment = run_hf_enrichment(claim, all_evidence_snippets)
            if verbose and hf_enrichment.get("hf_classification"):
                clf = hf_enrichment["hf_classification"]
                print(f"  HF Category: {clf.get('claim_category')} "
                      f"({clf.get('category_confidence')}%)")
                ents = hf_enrichment.get("hf_entities", [])
                if ents:
                    print(f"  HF Entities: {[e['word'] for e in ents[:5]]}")
        except Exception as e:
            if verbose:
                print(f"  HF enrichment skipped: {e}")

    # ── Step 5: Synthesize Final Verdict ──────────────────────────────────────
    if verbose:
        print(f"\nStep 5/5 — Synthesizing final verdict...")
    final = synthesize_verdict(claim, sub_claim_results)

    # Apply bias + credibility adjustments to confidence
    confidence_adj = (
        bias_result.get("confidence_adjustment", 0) +
        credibility_result.get("credibility_adjustment", 0)
    )
    adjusted_confidence = max(0, min(100,
                                     final.get("confidence_score",
                                               0) + confidence_adj
                                     ))
    final["confidence_score"] = adjusted_confidence
    final["confidence_adjustment"] = confidence_adj

    # Attach all enrichments
    elapsed = round(time.time() - start_time, 2)
    final.update({
        "processing_time": elapsed,
        "original_claim": claim,
        "decomposed": decomposed,
        "sub_claim_results": sub_claim_results,
        "bias_analysis": bias_result,
        "credibility_analysis": credibility_result,
        "hf_enrichment": hf_enrichment,
        "cache_hit": False
    })

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    try:
        claim_id = store_verified_claim(claim, final)
        final["claim_id"] = claim_id
    except Exception as e:
        if verbose:
            print(f"  Cache store warning: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERDICT:       {final['final_verdict']}")
        print(f"Confidence:    {adjusted_confidence}% "
              f"(adj: {confidence_adj:+d})")
        print(f"Bias:          {bias_result.get('overall_bias_direction')} "
              f"({bias_result.get('bias_severity')})")
        print(f"Credibility:   {credibility_result.get('credibility_label')}")
        print(f"Time:          {elapsed}s")
        print(f"\nSummary: {final['executive_summary']}")
        print(f"{'='*60}\n")

    return final


if __name__ == "__main__":
    stats = get_collection_stats()
    print(f"Cache: {stats['total_claims']} claims stored\n")

    test_claims = [
        "Pakistan is the 5th most populous country in the world",
        "The Eiffel Tower was built in 1887 and is located in Berlin",
        "Humans only use 10 percent of their brain",
    ]

    for claim in test_claims:
        result = run_pipeline(claim)
        status = "CACHED" if result.get("cache_hit") else "FRESH"
        print(f"[{status}] {result['final_verdict']} "
              f"({result['confidence_score']}%) in {result['processing_time']}s\n")

    stats = get_collection_stats()
    print(f"Cache now: {stats['total_claims']} claims stored")
