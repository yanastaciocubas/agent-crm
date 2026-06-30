import json
from agents.classifier import ClassifierAgent
from agents.responder import ResponderAgent
from agents.escalation import EscalationAgent
from agents.sentiment import SentimentAgent
from agents.retriever import RetrieverAgent
from rag.embedder import build_index


def run_pipeline(tickets):
    classifier = ClassifierAgent()
    sentiment = SentimentAgent()
    retriever = RetrieverAgent()
    responder = ResponderAgent()
    escalation = EscalationAgent()

    results = []
    for ticket in tickets:
        ticket = classifier.classify(ticket)
        ticket = sentiment.analyze(ticket)
        ticket = retriever.retrieve(ticket)
        ticket = responder.respond(ticket)
        ticket = escalation.evaluate(ticket)
        results.append(ticket)

    return results


def main():
    build_index()

    with open("data/sample_tickets.json") as f:
        tickets = json.load(f)

    results = run_pipeline(tickets)

    for ticket in results:
        print(f"\n{'='*50}")
        print(f"Ticket ID: {ticket['id']}")
        print(f"Customer: {ticket['customer']}")
        print(f"Category: {ticket['category']}")
        print(f"Priority: {ticket['priority']}")
        print(f"Sentiment: {ticket['sentiment']} — tone: {ticket['tone']}")
        print(f"RAG grounded: {ticket.get('response_grounded', False)}")
        print(f"Escalated: {ticket['escalated']} — {ticket['escalation_reason']}")
        print(f"Response: {ticket['response']}")


if __name__ == "__main__":
    main()