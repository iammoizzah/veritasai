"""Run the full evaluation harness — Day 5"""
from main import run_pipeline
from src.eval.ragas_eval import run_full_evaluation

if __name__ == "__main__":
    print("Starting VeritasAI Evaluation Harness...")
    print("This will run 10 test claims with ground truth labels.\n")

    summary = run_full_evaluation(
        test_claims_path="data/test_claims.json",
        pipeline_fn=run_pipeline,
        max_claims=10
    )

    print(f"\nFinal Score: {summary['exact_accuracy']}% exact accuracy")
    print(f"RAGAS Score: {summary['avg_ragas_score']}")
    print("\nUse these numbers on your resume and in your README.")
