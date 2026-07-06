from typing import TypedDict


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