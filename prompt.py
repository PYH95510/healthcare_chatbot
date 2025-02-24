# prompt.py

class PromptGenerator:
    """
    A class to handle different prompt engineering techniques,
    including standard retrieval prompts, Chain-of-Thought (CoT) prompting,
    and additional prompts used in the chatbot project.
    """

    def __init__(self):
        pass

    def standard_prompt(self, context, question):
        """Generates a standard RAG prompt."""
        return f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    def cot_prompt(self, context, question):
        """Generates a Chain-of-Thought (CoT) enhanced prompt."""
        return (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Let's think step by step before answering."
            " What are the key aspects to consider? Explain step by step.\n\n"
            "Final Answer:"
        )

    def chatbot_prompt(self, context, user_query):
        """Generates the prompt used in chatbot.py for querying LLM."""
        return f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"

    def llm_judge_prompt(self, question, expected_answer, generated_answer):
        return (
            "You are an AI judge evaluating a chatbot response. Based on the following criteria:\n"
            "- Relevance\n"
            "- Correctness\n"
            "- Coherence\n"
            "- Completeness\n"
            "- Conciseness\n\n"
            f"Question: {question}\n"
            f"Expected Answer: {expected_answer}\n"
            f"Chatbot Response: {generated_answer}\n\n"
            "Just provide **single numeric value between 1 and 10**. "
            "Your response must include **only this number** without any additional text or explanation."
        )


# Example usage
if __name__ == "__main__":
    prompt_gen = PromptGenerator()
    context = "Diabetes affects insulin production and causes increased blood sugar levels."
    question = "What are the symptoms of diabetes?"
    expected_answer = "Increased thirst, frequent urination, fatigue."
    generated_answer = "Diabetes causes frequent urination and high blood sugar."

    print("Standard Prompt:\n", prompt_gen.standard_prompt(context, question))
    print("\nCoT Prompt:\n", prompt_gen.cot_prompt(context, question))
    print("\nChatbot Prompt:\n", prompt_gen.chatbot_prompt(context, question))
    print("\nLLM Judge Prompt:\n", prompt_gen.llm_judge_prompt(question, expected_answer, generated_answer))
