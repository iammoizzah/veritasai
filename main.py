import os
import time
from dotenv import load_dotenv
from src.agents.decomposer import decompose_claim
from src.agents.verifier import collect_evidence, verify_sub_claim
from src.agents.synthesizer import synthesize_verdict

load_dotenv()


def run_pipeline(claim: str, verbose: bool = True) -> dict:
    """Run the full VeritasAI verification pipeline."""
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERITASAI — Claim Verification Pipeline")
        print(f"{'='*60}")
        print(f"Claim: {claim}\n")

    # Step 1: Decompose claim
    if verbose:
        print("Step 1/3 — Decomposing claim...")
    decomposed = decompose_claim(claim)
    sub_claims = decomposed.get("sub_claims", [])
    if verbose:
        print(f"  Type: {decomposed.get('claim_type')}")
        print(f"  Entities: {decomposed.get('entities')}")
        print(f"  Sub-claims: {len(sub_claims)}\n")

    # Step 2: Collect evidence and verify each sub-claim
    if verbose:
        print("Step 2/3 — Collecting evidence and verifying...")
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

    # Step 3: Synthesize final verdict
    if verbose:
        print(f"\nStep 3/3 — Synthesizing final verdict...")
    final = synthesize_verdict(claim, sub_claim_results)

    elapsed = round(time.time() - start_time, 2)
    final["processing_time"] = elapsed
    final["original_claim"] = claim
    final["decomposed"] = decomposed
    final["sub_claim_results"] = sub_claim_results

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERDICT:    {final['final_verdict']}")
        print(f"Confidence: {final['confidence_score']}%")
        print(f"Time:       {elapsed}s")
        print(f"\nSummary: {final['executive_summary']}")
        print(f"{'='*60}\n")

    return final


if __name__ == "__main__":
    test_claims = [
        "Pakistan is the 5th most populous country in the world",
        "The Eiffel Tower was built in 1887 and is located in Berlin",
    ]
    for claim in test_claims:
        result = run_pipeline(claim)
        print(f"Processed in {result['processing_time']}s\n")
