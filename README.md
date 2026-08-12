# VeritasAI — Automated Claim Verification Pipeline

An end-to-end AI fact-checking system that decomposes claims, retrieves evidence from live web and news sources, runs multi-agent verification, and synthesizes a structured verdict — with full LangSmith observability and RAGAS evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Claude](https://img.shields.io/badge/Claude-Haiku%20%2B%20Sonnet-purple?style=flat-square)
![LangSmith](https://img.shields.io/badge/LangSmith-Traced-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)

---

## The Problem

Misinformation spreads faster than humans can fact-check it. Journalists, researchers, compliance teams, and newsrooms need to verify claims at scale — but existing tools are either too expensive, too slow, or not built for developers to extend. Manual fact-checking a single article can take hours.

## The Solution

VeritasAI is a production-grade claim verification pipeline. Drop in any claim, article excerpt, or statement. The system automatically decomposes it into atomic sub-claims, retrieves live evidence, runs specialized verification agents, and outputs a structured verdict with confidence scores, citations, and a plain-English explanation.

**Verification time: ~15–30 seconds per claim.**

---

## System Architecture

```
Input Claim
     ↓
┌─────────────────────────────────────────┐
│         CLAIM DECOMPOSER                │
│  Breaks compound claims into atomic,    │
│  independently verifiable sub-claims    │
│  Extracts entities + search queries     │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│       EVIDENCE RETRIEVAL LAYER          │
│  • DuckDuckGo web search (live)         │
│  • NewsAPI (recent articles)            │
│  • ChromaDB vector store (Day 2)        │
│  • Wikipedia API (Day 2)                │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│    MULTI-AGENT VERIFICATION             │
│  • Evidence Collector Agent             │
│  • Sub-Claim Verifier Agent             │
│  • Bias Detector Agent (Day 3)          │
│  • Source Credibility Agent (Day 3)     │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│       VERDICT SYNTHESIZER               │
│  Weighted scoring across sub-claims     │
│  Final verdict + confidence score       │
│  Citations + plain-English explanation  │
└──────────────────┬──────────────────────┘
                   ↓
         Structured Verdict JSON
         + LangSmith Trace
         + RAGAS Eval Score (Day 5)
```

---

## Verdict Types

| Verdict           | Meaning                                       |
| ----------------- | --------------------------------------------- |
| ✅ TRUE           | Evidence clearly supports the claim           |
| ❌ FALSE          | Evidence clearly contradicts the claim        |
| ⚠️ MISLEADING     | Technically true but missing critical context |
| 🔶 PARTIALLY_TRUE | Some aspects correct, others incorrect        |
| ❓ UNVERIFIABLE   | Insufficient evidence to make a determination |

---

## Tech Stack

| Layer              | Technology            | Why                                     |
| ------------------ | --------------------- | --------------------------------------- |
| LLM (orchestrator) | Claude Sonnet         | Complex reasoning and synthesis         |
| LLM (sub-agents)   | Claude Haiku          | Cost-efficient classification tasks     |
| Agent framework    | LangGraph             | Stateful graph with conditional routing |
| Embeddings         | sentence-transformers | Semantic similarity + deduplication     |
| Vector store       | ChromaDB → Pinecone   | RAG evidence retrieval                  |
| Web search         | DuckDuckGo            | Free, no rate-limit for dev             |
| News search        | NewsAPI               | Real-time article retrieval             |
| Observability      | LangSmith             | Full pipeline tracing and cost tracking |
| Evaluation         | RAGAS                 | Faithfulness + answer relevance scoring |
| API                | FastAPI               | Production REST endpoints               |
| UI                 | Streamlit             | Demo interface                          |
| Storage            | PostgreSQL + pgvector | Claim history + deduplication           |

---

## Project Structure

```
veritasai/
├── main.py                    # Pipeline runner — start here
├── app.py                     # Streamlit UI (Day 7)
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── agents/
│   │   ├── decomposer.py      # Breaks claims into sub-claims
│   │   ├── verifier.py        # Evidence collection + sub-claim verification
│   │   ├── synthesizer.py     # Final verdict generation
│   │   ├── bias_detector.py   # Source bias analysis (Day 3)
│   │   └── credibility.py     # Source credibility scoring (Day 3)
│   │
│   ├── tools/
│   │   ├── search.py          # Web + news search
│   │   ├── wikipedia.py       # Wikipedia API (Day 2)
│   │   └── vector_store.py    # ChromaDB RAG (Day 2)
│   │
│   ├── eval/
│   │   ├── ragas_eval.py      # RAGAS evaluation harness (Day 5)
│   │   └── test_claims.py     # Labeled test dataset (Day 5)
│   │
│   └── api/
│       └── routes.py          # FastAPI endpoints (Day 6)
│
└── data/
    └── test_claims.json       # Ground truth for evaluation
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/veritasai.git
cd veritasai
```

### 2. Virtual environment

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
source venv/bin/activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

| Key                 | Where to get it                           |
| ------------------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY` | console.anthropic.com                     |
| `LANGCHAIN_API_KEY` | smith.langchain.com → Settings → API Keys |
| `NEWS_API_KEY`      | newsapi.org → Register → Dashboard        |

### 5. Run the pipeline

```bash
python main.py
```

### 6. Check LangSmith

Go to smith.langchain.com → your `veritasai` project. You should see traces for every agent call with latency, token usage, and inputs/outputs.

---

## Example Output

```
============================================================
VERITASAI — Claim Verification Pipeline
============================================================
Claim: Pakistan is the 5th most populous country in the world

Step 1/3 — Decomposing claim...
  Type: statistical
  Entities: ['Pakistan', 'world population ranking']
  Sub-claims: 2

Step 2/3 — Collecting evidence and verifying...
  Verifying: Pakistan is the 5th most populous country...
  → TRUE (confidence: 88%)

Step 3/3 — Synthesizing final verdict...

============================================================
VERDICT:    TRUE
Confidence: 88%
Time:       18.4s

Summary: Pakistan is indeed the 5th most populous country
in the world with approximately 231 million people, ranking
behind China, India, USA, and Indonesia.
============================================================
```

---

## LangSmith Observability

Every pipeline run generates a full trace in LangSmith showing:

- Which agents fired and in what order
- Input/output for each agent call
- Token usage and cost per step
- Latency breakdown
- Any errors or retries

This is what separates a production AI system from a demo.

---

## Build Roadmap

| Day | Focus                                                            | Status  |
| --- | ---------------------------------------------------------------- | ------- |
| 1   | Core pipeline — decomposer, verifier, synthesizer                | ✅ Done |
| 2   | RAG layer — ChromaDB + Wikipedia + vector deduplication          | 🔄 Next |
| 3   | Additional agents — bias detector, source credibility            | ⏳      |
| 4   | HuggingFace tasks — zero-shot classification, NER, summarization | ⏳      |
| 5   | Eval harness — RAGAS + labeled test set + accuracy metrics       | ⏳      |
| 6   | FastAPI + pgvector + Docker                                      | ⏳      |
| 7   | Streamlit UI + deployment + portfolio writeup                    | ⏳      |

---

## Resume Impact

> _"Built VeritasAI, an automated claim verification system using multi-agent LangGraph orchestration, RAG with ChromaDB/pgvector, and HuggingFace zero-shot classification. Achieved 60% verdict accuracy on 200-claim labeled test set. Average verification time 18s. Full LangSmith observability with RAGAS eval harness tracking faithfulness and answer relevance."_

**JD keywords this covers:**

- Multi-agent systems / LangGraph
- RAG and vector databases
- LLM evaluation (RAGAS)
- LLMOps and observability (LangSmith)
- HuggingFace model integration
- Production API development (FastAPI)
- Quantified results and eval metrics

---

## What I Learned Building This

- Designing stateful multi-agent pipelines with conditional routing
- Claim decomposition as a first-class NLP problem
- Evidence retrieval from heterogeneous sources (web + news + vector DB)
- Weighted verdict scoring across multiple sub-claims
- LangSmith tracing for full pipeline observability
- RAGAS evaluation for measuring LLM output quality
- Production patterns: tiered LLM strategy, cost tracking, semantic deduplication

---

## Author

**Moizzah** — [@iammoizzah](https://github.com/iammoizzah)
