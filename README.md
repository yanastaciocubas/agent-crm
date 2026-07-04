# AgentCRM

I'm studying Computer Science at Columbia, and I've been fascinated by how
companies like Salesforce use AI to manage customer relationships at scale.
I built AgentCRM to understand that from the inside, by actually building it,
and specifically to answer a question that generic chatbot demos gloss over:
how do you make sure the response is actually grounded in real company
knowledge instead of a plausible-sounding guess?

This is a five-agent pipeline that classifies, retrieves, responds to, and
escalates customer support tickets, with a retrieval-augmented generation
layer sitting in the middle and an eval harness to measure whether that
retrieval is actually working.

---

## How it works

Five agents run in sequence on every ticket:

1. **Classifier Agent**: categorizes the ticket (billing, bug, feature
   request, support, account) using keyword matching, and assigns a priority
2. **Sentiment Agent**: reads tone (angry, frustrated, calm, neutral) so the
   response st   response st   respon k   rrd-based   response st   response st   respon k   rrd-based   response st   respobe   response st   response st   respon k   rrd-based   response st   responto   response st   response st   respon k   rrd-based   response st   respontick   response st   response st   respon k   rrd-based   response st   responssponse from a category template, then
   grounds it with the actual retrieved knowledge base snippet when the
   retriever found a relevant match, and tags whether the response was
   grounded or not
5. **Escalation Agent**: flags tickets that need a human, based on priority
   and urgent-language detection
Ticket
│
▼
Classifier Agent  ──────► category, priority
│
▼
Sentiment Agent   ──────► sentiment, tone
│
▼
Retriever Agent   ──────► OpenAI embedding → Chroma similarity search
│                       → top-3 docs (FAQ + resolved tickets)
▼
Responder Agent   ──────► template response,Responder Agent   ──────► template response,Responder Agent   ──────► template response,Responder Agent   ──────► template response,Responder Agent   ──────► template response,Responder Agent   ──────► template response,Responder Agent   ──────�to what's actually hard: making sure retrieval works, and that thResponder Agent   ──────► template response,Responder AgseResponder Agent   ──────► template response,Responder AAQRespries and past rResponder Agent   ──────► template response,Respondemilarity. Responder Agent   ────�riever emReds it, pulls the closest matches, and the responder
only claims a response is "grounded" if a match actually cleared the
relevance threshold.

To check whether that's actually working, `eval/rag_eval.py` runs a labeled
set of test queries against the index and reports two metrics:

- **Precision@K**: of the top-K retrieved docs, how many were actually the
  right ones for that query, checked against a hand-labeled expected set
- **Faithfulness**: a lexical-overlap proxy between the response and the
  retrieved context. It's a simplified stand-in, not LLM-as-judge or RAGAS,
  but it catches the most obvious failure mode: a response that ignores the
  retrieved context entirely

```bash
python -m eval.rag_eval
```

prints a per-query breakdown plus an aggregate Precision@K and faithfulness
score, with a plain-language read on whether retrieval quality is good,
moderate, or needs work.

---

## Stack

| What | How |
|------|-----|
| Orchestration | Python, sequential agent pipeline |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB, pers| Vector store | ChromaDB, pers| Vector store | ChromaDB, pers| Vecit| Vector store | C
| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| Ru| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| D| DAPI_KEY=your-key-here   # or put it in a .env file

python main.py             python main.py             python main.py             python main.py   # run the retrieval eval harness
```

---

## What I'd build next

- Replace the template responses with actual LLM-generated text grounded in
  the re  the re  the re  the re  the re  the re  the re  the re  the re  thwap the   the re  the re  the re ess proxy for  the re  the re  the or   RA  the re  the re  the re  the re  the re  the re  the re  the re  the re  be  the re  the re  the re  the re  the re  the re  the re  the val + esca  the re  the re  ttually impro  the ra result
