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
        ticket["response"] = template.format(customer=customer)
        return ticket