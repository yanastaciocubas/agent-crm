class SentimentAgent:

    ANGRY_KEYWORDS = ["urgent", "immediately", "unacceptable", "furious", "ridiculous", 
                      "worst", "terrible", "awful", "scam", "fraud", "lawyer", "losing business",
                      "never again", "disgusting", "outrageous"]
    
    FRUSTRATED_KEYWORDS = ["still", "again", "already", "waiting", "days", "weeks", 
                           "nothing", "useless", "seriously", "really", "come on"]
    
    CALM_KEYWORDS = ["please", "kindly", "would love", "suggestion", "wondering", 
                     "could you", "thank you", "appreciate"]

    def analyze(self, ticket):
        message = ticket.get("message", "").lower()
        
        for keyword in self.ANGRY_KEYWORDS:
            if keyword in message:
                ticket["sentiment"] = "angry"
                ticket["tone"] = "empathetic and urgent"
                return ticket
        
        for keyword in self.FRUSTRATED_KEYWORDS:
            if keyword in message:
                ticket["sentiment"] = "frustrated"
                ticket["tone"] = "calm and reassuring"
                return ticket
        
        for keyword in self.CALM_KEYWORDS:
            if keyword in message:
                ticket["sentiment"] = "calm"
                ticket["tone"] = "friendly and helpful"
                return ticket
        
        ticket["sentiment"] = "neutral"
        ticket["tone"] = "professional"
        return ticket