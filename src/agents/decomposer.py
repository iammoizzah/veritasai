import json
import anthropic
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


@traceable(name="claim_decomposer")
def decompose_claim(claim: str) -> dict:
    """
    Break a compound claim into atomic, verifiable sub-claims.
    Each sub-claim should be independently verifiable.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system="""You are a claim decomposition expert. Break complex claims into atomic,
independently verifiable sub-claims.

Rules:
- Each sub-claim must be a single, concrete, verifiable statement
- Extract named entities (people, places, dates, numbers)
- Identify the claim type (factual, statistical, causal, predictive)
- Generate targeted search queries for each sub-claim
- Return ONLY valid JSON, no markdown

JSON structure:
{
  "original_claim": "the full original claim",
  "claim_type": "factual|statistical|causal|predictive|opinion",
  "entities": ["entity1", "entity2"],
  "sub_claims": [
    {
      "id": "sc_1",
      "text": "atomic sub-claim text",
      "search_queries": ["query1", "query2"],
      "verifiability": "high|medium|low"
    }
  ],
  "complexity": "simple|compound|complex"
}""",
        messages=[{"role": "user", "content": f"Decompose this claim: {claim}"}]
    )
    raw = response.content[0].text.strip().replace(
        "```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "original_claim": claim,
            "claim_type": "factual",
            "entities": [],
            "sub_claims": [{"id": "sc_1", "text": claim,
                           "search_queries": [claim], "verifiability": "medium"}],
            "complexity": "simple"
        }
