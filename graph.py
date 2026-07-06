from typing import TypedDict

from agent_crm.classifier import classify as classify_ticket
from agent_crm.sentiment import analyze_sentiment


class RetrievedDoc(TypedDict):
    content: str
    score: float


class AgentState(TypedDict):
    original_query: str        # the customer's message, never overwritten
    query: str                  # what actually gets embedded, starts as original_query
    category: str
    sentiment: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_score: float      # top-1 score from the most recent retrieve() call
    grounded: bool               # did the most recent retrieve() clear MATCH_THRESHOLD
    retry_count: int
    response: str
    escalate: bool


def classify_node(state: AgentState) -> dict:
    return {"category": classify_ticket(state["original_query"])}


def sentiment_node(state: AgentState) -> dict:
    return {"sentiment": analyze_sentiment(state["original_query"])}