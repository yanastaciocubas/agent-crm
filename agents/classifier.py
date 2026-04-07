class ClassifierAgent:
    
    KEYWORDS = {
        "billing": ["charged", "refund", "payment", "invoice", "charge", "money", "subscription"],
        "bug": ["crash", "error", "broken", "not working", "issue", "bug", "failing"],
        "feature_request": ["add", "would be nice", "can you", "feature", "improve", "suggestion"],
        "support": ["how do i", "help", "can't find", "where is", "how to"],
        "account": ["locked", "can't log in", "access", "password", "account"]
    }

    def classify(self, ticket):
        message = ticket["message"].lower()
        
        for category, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                if keyword in message:
                    ticket["category"] = category
                    ticket["priority"] = self._get_priority(category)
                    return ticket
        
        ticket["category"] = "general"
        ticket["priority"] = "low"
        return ticket
    
    def _get_priority(self, category):
        priorities = {
            "billing": "high",
            "bug": "high",
            "account": "high",
            "support": "medium",
            "feature_request": "low"
        }
        return priorities.get(category, "low")