from typing import TypedDict

from agents.classifier import ClassifierAgent
from agents.sentiment import SentimentAgent
from agents.retriever import RetrieverAgent

MATCH_THRESHOLD = 0.30


class RetrievedDoc(TypedDict):
    id: str
    text: str
    metadata: dict
    score: float


class AgentState(TypedDict):
    ticket_id: str
    original_query: str        # the customer's message, never overwritten
    query: str                   # what actually gets embedded, starts as original_query
    category: str
    priority: str
    sentiment: str
    tone: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_score: float       # top-1 score from the most recent retrieve() call
    grounded: bool                 # did the most recent retrieve() clear MATCH_THRESHOLD
    retry_count: int
    response: str
    escalate: bool


_classifier = ClassifierAgent()
_sentiment = SentimentAgent()
_retriever = RetrieverAgent()


def classify_node(state: AgentState) -> dict:
    ticket = {"id": state["ticket_id"], "message": state["original_query"]}
    ticket = _classifier.classify(ticket)
    return {"category": ticket["category"], "priority": ticket["priority"]}


def sentiment_node(state: AgentState) -> dict:
    ticket = {"id": state["ticket_id"], "message": state["original_query"]}
    ticket = _sentiment.analyze(ticket)
    return {"sentiment": ticket["sentiment"], "tone": ticket["tone"]}


def retrieve_node(state: AgentState) -> dict:
    ticket = {"id": state["ticket_id"], "message": state["query"]}
    ticket = _retriever.retrieve(ticket, min_score=0.0)
    docs = ticket["rag_context"]
    top_score = docs[0]["score"] if docs else 0.0
    return {
        "retrieved_docs": docs,
        "retrieval_score": top_score,
        "grounded": top_score >= MATCH_THRESHOLD,
    }