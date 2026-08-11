"""
Day 4 — HuggingFace Task Integration
Uses HF Inference API for:
- Zero-Shot Classification (claim categorization)
- Token Classification / NER (entity extraction)
- Summarization (evidence condensing)
- Sentence Similarity (claim deduplication)
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models"
# Optional — works without token but rate limited
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def _hf_query(model: str, payload: dict, timeout: int = 30) -> dict | list | None:
    """Generic HuggingFace Inference API call."""
    try:
        r = httpx.post(
            f"{HF_API_URL}/{model}",
            headers=HEADERS,
            json=payload,
            timeout=timeout
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 503:
            # Model loading — return None to fall back to Claude
            return None
        return None
    except Exception:
        return None


def classify_claim_type(claim: str) -> dict:
    """
    HF Task: Zero-Shot Classification
    Categorizes the claim without any training data.
    """
    candidate_labels = [
        "political claim", "scientific claim", "historical fact",
        "statistical claim", "economic claim", "health claim",
        "technology claim", "geographical fact", "biographical claim"
    ]
    result = _hf_query(
        "facebook/bart-large-mnli",
        {"inputs": claim, "parameters": {"candidate_labels": candidate_labels}}
    )
    if result and "labels" in result:
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return {
            "claim_category": top_label,
            "category_confidence": round(top_score * 100, 1),
            "all_labels": dict(zip(result["labels"][:3], [round(s*100, 1) for s in result["scores"][:3]])),
            "source": "huggingface_zero_shot"
        }
    # Fallback
    return {
        "claim_category": "general claim",
        "category_confidence": 0,
        "all_labels": {},
        "source": "fallback"
    }


def extract_entities_hf(text: str) -> list[dict]:
    """
    HF Task: Token Classification (NER)
    Extracts named entities from claim text.
    """
    result = _hf_query(
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        {"inputs": text}
    )
    if result and isinstance(result, list):
        entities = []
        for item in result:
            if isinstance(item, dict) and item.get("score", 0) > 0.85:
                entities.append({
                    "word": item.get("word", "").replace("##", ""),
                    "entity_type": item.get("entity_group", item.get("entity", "")),
                    "confidence": round(item.get("score", 0) * 100, 1)
                })
        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = e["word"].lower()
            if key not in seen and len(e["word"]) > 1:
                seen.add(key)
                unique.append(e)
        return unique[:10]
    return []


def summarize_evidence(evidence_text: str) -> str:
    """
    HF Task: Summarization
    Condenses long evidence into key points.
    """
    if len(evidence_text) < 200:
        return evidence_text
    result = _hf_query(
        "facebook/bart-large-cnn",
        {
            "inputs": evidence_text[:1024],
            "parameters": {"max_length": 150, "min_length": 40, "do_sample": False}
        }
    )
    if result and isinstance(result, list) and result:
        return result[0].get("summary_text", evidence_text[:300])
    return evidence_text[:300]


def check_claim_similarity_hf(claim1: str, claim2: str) -> float:
    """
    HF Task: Sentence Similarity
    Returns similarity score 0-1 between two claims.
    """
    result = _hf_query(
        "sentence-transformers/all-MiniLM-L6-v2",
        {"inputs": {"source_sentence": claim1, "sentences": [claim2]}}
    )
    if result and isinstance(result, list):
        return round(result[0], 3)
    return 0.0


def run_hf_enrichment(claim: str, evidence_snippets: list[str]) -> dict:
    """
    Run all HF enrichments on a claim.
    Returns enriched metadata for the verification report.
    """
    # Zero-shot classification
    classification = classify_claim_type(claim)

    # NER on the claim
    entities = extract_entities_hf(claim)

    # Summarize combined evidence
    combined_evidence = " ".join(evidence_snippets[:3])[:1024]
    evidence_summary = summarize_evidence(
        combined_evidence) if combined_evidence else ""

    return {
        "hf_classification": classification,
        "hf_entities": entities,
        "evidence_summary": evidence_summary,
        "hf_enriched": True
    }
