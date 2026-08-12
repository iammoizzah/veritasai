from src.tools.vector_store import get_collection_stats, find_similar_claims
from main import run_pipeline
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time
from datetime import datetime

# Import pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = FastAPI(
    title="VeritasAI API",
    description="Automated claim verification pipeline with multi-agent AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=10, max_length=1000,
                       description="The claim to verify")
    use_hf: bool = Field(
        default=True, description="Use HuggingFace enrichment")
    use_cache: bool = Field(
        default=True, description="Check semantic cache first")


class VerificationResponse(BaseModel):
    request_id: str
    claim: str
    verdict: str
    confidence_score: int
    executive_summary: str
    detailed_explanation: str
    key_facts: list
    sources: list
    bias_direction: str
    credibility_label: str
    processing_time: float
    cache_hit: bool
    timestamp: str


class StatsResponse(BaseModel):
    total_claims_verified: int
    total_evidence_chunks: int
    api_version: str
    status: str


# ── In-memory job store (use Redis in prod) ───────────────────────────────────
jobs = {}

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "VeritasAI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": stats.get("total_claims", 0)
    }


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
def get_stats():
    stats = get_collection_stats()
    return StatsResponse(
        total_claims_verified=stats.get("total_claims", 0),
        total_evidence_chunks=stats.get("total_evidence_chunks", 0),
        api_version="1.0.0",
        status="running"
    )


@app.post("/verify", response_model=VerificationResponse, tags=["Verification"])
def verify_claim(request: ClaimRequest):
    """
    Verify a claim synchronously.
    Returns full verification result including verdict, confidence,
    bias analysis, source credibility, and key facts.
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        result = run_pipeline(
            claim=request.claim,
            verbose=False,
            use_hf=request.use_hf
        )

        return VerificationResponse(
            request_id=request_id,
            claim=request.claim,
            verdict=result.get("final_verdict", "UNVERIFIABLE"),
            confidence_score=result.get("confidence_score", 0),
            executive_summary=result.get("executive_summary", ""),
            detailed_explanation=result.get("detailed_explanation", ""),
            key_facts=result.get("key_facts", []),
            sources=result.get("sources", []),
            bias_direction=result.get("bias_analysis", {}).get(
                "overall_bias_direction", "unknown"),
            credibility_label=result.get("credibility_analysis", {}).get(
                "credibility_label", "unknown"),
            processing_time=result.get("processing_time", 0),
            cache_hit=result.get("cache_hit", False),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify/async", tags=["Verification"])
def verify_claim_async(request: ClaimRequest, background_tasks: BackgroundTasks):
    """
    Submit a claim for async verification.
    Returns a job_id — poll /jobs/{job_id} for the result.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending",
                    "submitted_at": datetime.now().isoformat()}

    def run_job():
        try:
            jobs[job_id]["status"] = "running"
            result = run_pipeline(claim=request.claim,
                                  verbose=False, use_hf=request.use_hf)
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = result
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "pending", "poll_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}", tags=["Verification"])
def get_job(job_id: str):
    """Poll for async verification result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/similar", tags=["Search"])
def find_similar(claim: str, threshold: float = 0.80):
    """Find semantically similar previously verified claims."""
    similar = find_similar_claims(claim, threshold=threshold)
    return {"claim": claim, "similar_claims": similar, "count": len(similar)}


@app.get("/verdicts", tags=["Reference"])
def get_verdict_types():
    """Get all possible verdict types with descriptions."""
    return {
        "verdicts": {
            "TRUE": "Evidence clearly supports the claim",
            "FALSE": "Evidence clearly contradicts the claim",
            "MISLEADING": "Technically true but missing critical context",
            "PARTIALLY_TRUE": "Some aspects correct, others incorrect",
            "UNVERIFIABLE": "Insufficient evidence to make a determination"
        }
    }
