import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize ChromaDB with persistent storage
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".chroma")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use sentence-transformers for embeddings (free, local)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Two collections: verified claims + evidence chunks
claims_collection = chroma_client.get_or_create_collection(
    name="verified_claims",
    embedding_function=embedding_fn,
    metadata={"description": "Previously verified claims with verdicts"}
)

evidence_collection = chroma_client.get_or_create_collection(
    name="evidence_chunks",
    embedding_function=embedding_fn,
    metadata={"description": "Evidence chunks from past verifications"}
)


def store_verified_claim(claim: str, result: dict) -> str:
    """Store a verified claim and its verdict in ChromaDB."""
    claim_id = f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    metadata = {
        "verdict": result.get("final_verdict", "UNVERIFIABLE"),
        "confidence": str(result.get("confidence_score", 0)),
        "timestamp": datetime.now().isoformat(),
        "processing_time": str(result.get("processing_time", 0)),
        "sub_claim_count": str(result.get("sub_claim_count", 0))
    }
    claims_collection.add(
        ids=[claim_id],
        documents=[claim],
        metadatas=[metadata]
    )
    # Store evidence chunks too
    sources = result.get("sources", [])
    for i, source in enumerate(sources[:5]):
        if source:
            evidence_collection.add(
                ids=[f"{claim_id}_src_{i}"],
                documents=[source],
                metadatas={"claim_id": claim_id, "claim": claim[:200]}
            )
    return claim_id


def find_similar_claims(claim: str, threshold: float = 0.85) -> list[dict]:
    """
    Check if a similar claim was already verified.
    Returns cached results if similarity > threshold.
    """
    try:
        results = claims_collection.query(
            query_texts=[claim],
            n_results=3
        )
        similar = []
        if results and results["distances"]:
            for i, distance in enumerate(results["distances"][0]):
                # ChromaDB returns L2 distance — convert to similarity
                similarity = 1 / (1 + distance)
                if similarity >= threshold:
                    similar.append({
                        "claim": results["documents"][0][i],
                        "similarity": round(similarity, 3),
                        "verdict": results["metadatas"][0][i].get("verdict"),
                        "confidence": results["metadatas"][0][i].get("confidence"),
                        "timestamp": results["metadatas"][0][i].get("timestamp"),
                        "id": results["ids"][0][i]
                    })
        return similar
    except Exception:
        return []


def get_relevant_evidence(query: str, n_results: int = 5) -> list[str]:
    """Retrieve relevant evidence chunks for a query."""
    try:
        results = evidence_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results and results["documents"]:
            return results["documents"][0]
        return []
    except Exception:
        return []


def get_collection_stats() -> dict:
    """Get stats about stored claims."""
    try:
        return {
            "total_claims": claims_collection.count(),
            "total_evidence_chunks": evidence_collection.count()
        }
    except Exception:
        return {"total_claims": 0, "total_evidence_chunks": 0}
