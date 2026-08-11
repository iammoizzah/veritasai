"""
Day 5 — Evaluation Harness
Uses RAGAS metrics to evaluate pipeline quality:
- Faithfulness: Does the verdict follow from the evidence?
- Answer Relevance: Does the verdict address the actual claim?
- Context Recall: Was all relevant evidence retrieved?
"""
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def evaluate_single_result(claim: str, verdict_result: dict) -> dict:
    """
    Evaluate a single verification result using heuristic RAGAS-style metrics.
    Full RAGAS requires an LLM judge — we implement lightweight versions here.
    """
    scores = {}

    # 1. Faithfulness: Does reasoning cite specific evidence?
    reasoning = verdict_result.get("detailed_explanation", "")
    supporting = verdict_result.get("key_facts", [])
    sources = verdict_result.get("sources", [])

    faithfulness_score = 0.0
    if supporting and len(supporting) >= 2:
        faithfulness_score += 0.4
    if sources and len(sources) >= 1:
        faithfulness_score += 0.3
    if reasoning and len(reasoning) > 100:
        faithfulness_score += 0.3
    scores["faithfulness"] = round(faithfulness_score, 2)

    # 2. Answer Relevance: Is verdict type appropriate to claim type?
    claim_type = verdict_result.get("decomposed", {}).get("claim_type", "")
    verdict = verdict_result.get("final_verdict", "UNVERIFIABLE")
    confidence = verdict_result.get("confidence_score", 0)

    answer_relevance = 0.5
    if verdict != "UNVERIFIABLE":
        answer_relevance += 0.25
    if confidence >= 60:
        answer_relevance += 0.25
    scores["answer_relevance"] = round(answer_relevance, 2)

    # 3. Context Precision: How many sub-claims were verified?
    sub_claim_results = verdict_result.get("sub_claim_results", [])
    sub_claim_count = len(sub_claim_results)
    verified_count = sum(1 for r in sub_claim_results
                         if r.get("verdict") not in ["UNVERIFIABLE"])

    context_precision = 0.0
    if sub_claim_count > 0:
        context_precision = verified_count / sub_claim_count
    scores["context_precision"] = round(context_precision, 2)

    # 4. Source Diversity: Multiple source types used?
    source_types = set()
    for sc in sub_claim_results:
        for src_type in sc.get("sources_used", []):
            source_types.add(src_type)
    diversity_score = min(len(source_types) / 3, 1.0)
    scores["source_diversity"] = round(diversity_score, 2)

    # 5. Overall RAGAS-style score
    overall = (
        scores["faithfulness"] * 0.35 +
        scores["answer_relevance"] * 0.30 +
        scores["context_precision"] * 0.20 +
        scores["source_diversity"] * 0.15
    )
    scores["overall_ragas"] = round(overall, 2)

    return scores


def evaluate_against_ground_truth(
    claim: str,
    predicted_verdict: str,
    ground_truth: str
) -> dict:
    """Compare predicted verdict to ground truth label."""
    # Exact match
    exact_match = predicted_verdict == ground_truth

    # Partial credit mapping
    credit_map = {
        ("TRUE", "PARTIALLY_TRUE"): 0.5,
        ("PARTIALLY_TRUE", "TRUE"): 0.5,
        ("FALSE", "MISLEADING"): 0.3,
        ("MISLEADING", "FALSE"): 0.3,
        ("MISLEADING", "PARTIALLY_TRUE"): 0.4,
        ("PARTIALLY_TRUE", "MISLEADING"): 0.4,
    }
    partial_credit = credit_map.get((predicted_verdict, ground_truth), 0.0)
    score = 1.0 if exact_match else partial_credit

    return {
        "predicted": predicted_verdict,
        "ground_truth": ground_truth,
        "exact_match": exact_match,
        "score": score,
        "correct": exact_match
    }


def run_full_evaluation(test_claims_path: str, pipeline_fn, max_claims: int = 10) -> dict:
    """
    Run the full evaluation harness on a labeled test set.
    Returns aggregate metrics and per-claim results.
    """
    with open(test_claims_path) as f:
        test_claims = json.load(f)[:max_claims]

    results = []
    ragas_scores = []
    accuracy_scores = []
    total_time = 0

    print(f"\n{'='*60}")
    print(f"VERITASAI — Evaluation Harness")
    print(f"Running {len(test_claims)} test claims")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_claims, 1):
        claim = tc["claim"]
        ground_truth = tc["ground_truth"]
        print(f"[{i}/{len(test_claims)}] {claim[:65]}...")

        start = time.time()
        try:
            result = pipeline_fn(claim, verbose=False)
            elapsed = time.time() - start
            total_time += elapsed

            predicted = result.get("final_verdict", "UNVERIFIABLE")

            # RAGAS-style metrics
            ragas = evaluate_single_result(claim, result)
            ragas_scores.append(ragas["overall_ragas"])

            # Accuracy vs ground truth
            accuracy = evaluate_against_ground_truth(
                claim, predicted, ground_truth)
            accuracy_scores.append(accuracy["score"])

            entry = {
                "id": tc["id"],
                "claim": claim,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": accuracy["correct"],
                "score": accuracy["score"],
                "confidence": result.get("confidence_score", 0),
                "ragas_scores": ragas,
                "processing_time": round(elapsed, 2),
                "cache_hit": result.get("cache_hit", False),
                "difficulty": tc.get("difficulty", "unknown")
            }
            results.append(entry)

            status = "✓" if accuracy["correct"] else "✗"
            print(f"  {status} Predicted: {predicted} | Truth: {ground_truth} "
                  f"| RAGAS: {ragas['overall_ragas']} | {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": tc["id"],
                "claim": claim,
                "ground_truth": ground_truth,
                "predicted": "ERROR",
                "correct": False,
                "score": 0,
                "error": str(e)
            })
            accuracy_scores.append(0)

    # Aggregate metrics
    exact_accuracy = sum(1 for r in results if r.get(
        "correct", False)) / len(results)
    weighted_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_ragas = sum(ragas_scores) / len(ragas_scores) if ragas_scores else 0
    avg_time = total_time / len(results)

    # Breakdown by difficulty
    by_difficulty = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "correct": 0}
        by_difficulty[diff]["total"] += 1
        if r.get("correct"):
            by_difficulty[diff]["correct"] += 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_claims": len(results),
        "exact_accuracy": round(exact_accuracy * 100, 1),
        "weighted_accuracy": round(weighted_accuracy * 100, 1),
        "avg_ragas_score": round(avg_ragas, 3),
        "avg_processing_time": round(avg_time, 2),
        "total_time": round(total_time, 2),
        "by_difficulty": {
            k: {
                "accuracy": round(v["correct"]/v["total"]*100, 1) if v["total"] > 0 else 0,
                "total": v["total"]
            }
            for k, v in by_difficulty.items()
        },
        "per_claim_results": results
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Exact Accuracy:    {summary['exact_accuracy']}%")
    print(f"Weighted Accuracy: {summary['weighted_accuracy']}%")
    print(f"Avg RAGAS Score:   {summary['avg_ragas_score']}")
    print(f"Avg Time/Claim:    {summary['avg_processing_time']}s")
    print(f"Total Time:        {summary['total_time']}s")
    print(f"\nBy Difficulty:")
    for diff, stats in summary["by_difficulty"].items():
        print(f"  {diff}: {stats['accuracy']}% ({stats['total']} claims)")
    print(f"{'='*60}\n")

    # Save results
    output_path = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}")

    return summary
