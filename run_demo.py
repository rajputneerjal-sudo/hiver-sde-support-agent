from src.pipeline import SupportAgent


def main():
    agent = SupportAgent()

    messages = [
        "Where is my order? It has not arrived yet.",
        "I want to return the product and get a refund.",
        "I was charged twice for the same order.",
        "I cannot log into my account.",
        "My product arrived damaged.",
    ]

    for message in messages:
        print("\n" + "=" * 70)
        print("CUSTOMER:", message)

        result = agent.process(message)

        print("INTENT:", result.intent)
        print("CONFIDENCE:", round(result.intent_confidence, 3))
        print("ESCALATE:", result.escalate)
        print("ESCALATION REASON:", result.escalation_reason)
        print("REPLY:", result.reply)
        print("REPLY SOURCE:", result.reply_source)

        if result.evidence:
            print("EVIDENCE:")
            for evidence in result.evidence:
                print(" -", evidence)


if __name__ == "__main__":
    main()