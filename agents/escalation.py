class EscalationAgent:

    HIGH_PRIORITY_CATEGORIES = ["billing", "bug", "account"]
    ESCALATION_KEYWORDS = ["urgent", "losing business", "lawyer", "scam", "fraud", "immediately", "unacceptable"]

    def evaluate(self, ticket):
        message = ticket.get("message", "").lower()
        category = ticket.get("category", "general")
        priority = ticket.get("priority", "low")

        needs_escalation = False
        reason = None

        if priority == "high" and category in self.HIGH_PRIORITY_CATEGORIES:
            needs_escalation = True
            reason = f"High priority {category} issue"

        for keyword in self.ESCALATION_KEYWORDS:
            if keyword in message:
                needs_escalation = True
                reason = f"Urgent language detected: '{keyword}'"
                break

        ticket["escalated"] = needs_escalation
        ticket["escalation_reason"] = reason if needs_escalation else "N/A"
        return ticket