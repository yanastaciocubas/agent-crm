class ResponderAgent:
    RESPONSES = {
        "billing": "Hi {customer}, we're so sorry about the billing issue! We've flagged your account for an immediate review and will process any necessary refunds within 3-5 business days. Thank you for your patience!",
        "bug": "Hi {customer}, thanks for reporting this! We've logged the issue with our engineering team and are working on a fix. We'll keep you updated on the progress!",
        "feature_request": "Hi {customer}, we love the suggestion! We've passed it along to our product team for consideration. Keep the great ideas coming!",
        "support": "Hi {customer}, happy to help! Please check our help center at help.ourstore.com or reply with more details and we'll walk you through it step by step!",
        "account": "Hi {customer}, we're sorry you're having trouble accessing your account! Our team has been notified and will reach out within 24 hours to get you back in. Hang tight!",
        "general": "Hi {customer}, thanks for reaching out! A member of our support team will get back to you within 24 hours."
    }

    def respond(self, ticket):
        category = ticket.get("category", "general")
        customer = ticket.get("customer", "there")
        template = self.RESPONSES.get(category, self.RESPONSES["general"])

        # Base response (same as before)
        base_response = template.format(customer=customer)

        # RAG augmentation: if the retriever found relevant context, append it
        rag_context = ticket.get("rag_context", [])
        if rag_context:
            top_match = rag_context[0]
            source = top_match["metadata"]["source"]
            context_text = top_match["text"]

            # Extract just the answer portion depending on doc type
            if source == "faq" and "\nA: " in context_text:
                grounded_info = context_text.split("\nA: ", 1)[1]
            elif source == "resolved_ticket" and "\nResolution: " in context_text:
                grounded_info = context_text.split("\nResolution: ", 1)[1].split("\nOutcome:")[0].strip()
            else:
                grounded_info = context_text

            ticket["response"] = (
                f"{base_response}\n\n"
                f"Based on our knowledge base: {grounded_info}"
            )
            ticket["response_grounded"] = True
            ticket["rag_sources"] = [r["id"] for r in rag_context]
        else:
            ticket["response"] = base_response
            ticket["response_grounded"] = False
            ticket["rag_sources"] = []

        return ticket