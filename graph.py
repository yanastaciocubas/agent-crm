import json
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from agents.classifier import ClassifierAgent
from agents.sentiment import SentimentAgent
from agents.retriever import RetrieverAgent
from agents.responder import ResponderAgent
from agents.escalation import EscalationAgent

MATCH_THRESHOLD = 0.30
MAX_RETRIES = 1
REFORMULATE_MODEL = "gpt-4o-mini"  # swap for whatever cheap/fast chat model your account has


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
    escalation_reason: str


_classifier = ClassifierAgent()
_sentiment = SentimentAgent()
_retriever = RetrieverAgent()
_responder = ResponderAgent()
_escalation = EscalationAgent()
_openai_client = OpenAI()


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


def reformulate_node(state: AgentState) -> dict:
    """
    LLM call that reasons about why the first search missed, not
    keyword stripping. An LLM that sees what did come back, even below
    threshold, can name the actual mismatch and rewrite toward the
    KB's language.
    """
    weak_hits = state["retrieved_docs"][:3]
    context = "\n".join(
        f"- (score {doc['score']:.2f}) {doc['text'][:200]}" for doc in weak_hits
    ) or "(nothing came back)"

    prompt = f"""A customer support search returned weak matches for this query.

Original query: "{state['query']}"

Top results and similarity scores (match threshold is {MATCH_THRESHOLD}):
{context}

In one sentence, name the likely reason the search missed (vague phrasing,
wrong terminology, missing product name, too broad, etc). Then rewrite the
query so it will search the knowledge base more effectively.Thank you.

Respond as JSON only: {{"reason": "...", "reformulated_query": "..."}}"""

    completion = _openai_client.chat.completions.create(
        model=REFORMULATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(completion.choices[0].message.content)

    return {
        "query": result["reformulated_query"],
        "retry_count": state["retry_count"] + 1,
    }


def respond_node(state: AgentState) -> dict:
    ticket = {
        "id": state["ticket_id"],
        "message": state["original_query"],
        "category": state["category"],
        "rag_context": state["retrieved_docs"] if state["grounded"] else [],
    }
    ticket = _responder.respond(ticket)
    return {"response": ticket["response"]}


def escalate_node(state: AgentState) -> dict:
    ticket = {
        "id": state["ticket_id"],
        "message": state["original_query"],
        "category": state["category"],
        "priority": state["priority"],
    }
    ticket = _escalation.evaluate(ticket)
    return {"escalate": ticket["escalated"], "escalation_reason": ticket["escalation_reason"]}


def route_after_retrieve(state: AgentState) -> Literal["respond", "reformulate"]:
    if state["grounded"]:
        return "respond"
    if state["retry_count"] < MAX_RETRIES:
        return "reformulate"
    return "respond"  # retry spent, fall through and answer honestly ungrounded


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("reformulate", reformulate_node)
    builder.add_node("respond", respond_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "sentiment")
    builder.add_edge("sentiment", "retrieve")

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"respond": "respond", "reformulate": "reformulate"},
    )
    builder.add_edge("reformulate", "retrieve")

    builder.add_edge("respond", "escalate")
    builder.add_edge("escalate", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({
        "ticket_id": "demo-1",
        "original_query": "my payment didn't go through but I got charged twice",
        "query": "my payment didn't go through but I got charged twice",
        "retry_count": 0,
    })
    print(result["response"])
    print("escalate:", result["escalate"])