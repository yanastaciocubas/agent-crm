# AgentCRM

<img width="800" height="450" alt="Image" src="https://github.com/user-attachments/assets/5f352f94-51e0-470c-8581-3f5a6b39a6a6" />

I studied Computer Science at Columbia, and I've been fascinated by how
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
   response strategy can adjust, also keyword-based
3. **Retriever Agent**: embeds the ticket text with OpenAI's
   `text-embedding-3-small`, queries a persisted Chroma vector store, and
   pulls the top-3 most relevant docs from a knowledge base of FAQs and past
   resolved tickets (above a 0.30 similarity threshold)
4. **Responder Agent**: builds a response from a category template, then
   grounds it with the actual retrieved knowledge base snippet when the
   retriever found a relevant match, and tags whether the response was
   grounded or not
5. **Escalation Agent**: flags tickets that need a human, based on priority
   and urgent-language detection

```
Ticket
  |
  v
Classifier Agent  ------> category, priority
  |
  v
Sentiment Agent   ------> sentiment, tone
  |
  v
Retriever Agent   ------> OpenAI embedding -> Chroma similarity search
  |                       -> top-3 docs (FAQ + resolved tickets)
  v
Responder Agent   ------> template response, grounded with retrieved
  |                       snippet if a match was found
  v
Escalation Agent  ------> escalated: true/false + reason
  |
  v
Result
```

Classification, sentiment, and escalation are rule-based, not ML. That's on
purpose. Keeping that part simple and deterministic let me put the real
effort into what's actually hard: making sure retrieval works, and that the
grounding is real instead of just padding on top of a generic response.

---

## RAG layer + eval harness

The knowledge base is a mix of FAQ entries and past resolved tickets, each
embedded and stored in Chroma with cosine similarity. When a ticket comes
in, the retriever embeds it, pulls the closest matches, and the responder
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
| Vector store | ChromaDB, persisted locally, cosine similarity |
| Eval | Custom Precision@K + faithfulness harness |
| Data | Mock tickets + FAQ/resolved-ticket knowledge base (JSON) |

---

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here   # or put it in a .env file

python main.py              # run the full pipeline on sample tickets
python -m eval.rag_eval     # run the retrieval eval harness
```

---

## What I'd build next

- Replace the template responses with actual LLM-generated text grounded in
  the retrieved context, rather than a template plus an appended snippet
- Swap the lexical-overlap faithfulness proxy for a real LLM-as-judge (or
  RAGAS) score
- Move classification and sentiment from keyword rules to an embedding or
  small-model based approach, and see how much retrieval + escalation
  quality actually improves as a result
