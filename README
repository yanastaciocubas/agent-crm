# AgentCRM

I'm studying Computer Science at Columbia, and I've been fascinated by how 
companies like Salesforce use AI to manage customer relationships at scale. 
I built AgentCRM to understand that from the inside (by actually building it).

This is a multi-agent pipeline that automatically classifies, responds to, 
and escalates customer support tickets. No UI yet, just pure pipeline logic.

## How it works

Three agents work in sequence on every ticket:

1. **Classifier Agent**: reads the ticket and categorizes it (billing, bug, 
   feature request, support, account) based on keyword detection
2. **Responder Agent**: generates a personalized response based on the category
3. **Escalation Agent**: flags tickets that need human attention based on 
   priority and urgency signals

## What I'd build next

- **Sentiment analysis**: detect how frustrated the customer is and adjust 
  the tone of the response accordingly
- **Customer feedback loop**: let customers rate responses (👍/👎) so the 
  system can learn and improve over time (basically a lightweight RLHF system)
- **Resource suggestions**: attach relevant help articles or discount codes 
  based on the ticket category

## Stack
- Python
- JSON (mock ticket data)

## Run it!
```bash
python3 main.py
```