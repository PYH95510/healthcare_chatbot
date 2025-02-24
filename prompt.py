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

    def cot_prompt(self, history, context, question):
        """
        Generates a hidden Chain-of-Thought (CoT) enhanced prompt with conversation history.
        The prompt instructs the LLM to internally reason and provide only a concise final answer.
        """
        conversation_text = "\n".join(history) if history else ""
        return (
            f"Conversation History:\n{conversation_text}\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Please think through all the key aspects internally. Then, provide only a concise final answer without showing your internal reasoning.\n\n"
            "Final Answer:"
        )

    def chatbot_prompt(self, context, user_query):
        """Generates the prompt used in chatbot.py for querying LLM."""
        return f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"

    def llm_judge_prompt(self, question, expected_answer, generated_answer):
        """
        Generates a prompt for evaluating a chatbot response based on multiple criteria.
        The criteria include relevance, correctness, coherence, completeness, and conciseness.
        """
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
            "Just provide **a single numeric value between 1 and 10**. "
            "Your response must include **only this number** without any additional text or explanation."
        )


# Example usage
if __name__ == "__main__":
    prompt_gen = PromptGenerator()
    context = "Diabetes affects insulin production and causes increased blood sugar levels."
    question = "What are the symptoms of diabetes?"
    expected_answer = "Increased thirst, frequent urination, fatigue."
    generated_answer = "Diabetes causes frequent urination and high blood sugar."
    history = ["User: What are the symptoms of diabetes?"]

    print("Standard Prompt:\n", prompt_gen.standard_prompt(context, question))
    print("\nCoT Prompt:\n", prompt_gen.cot_prompt(history, context, question))
    print("\nChatbot Prompt:\n", prompt_gen.chatbot_prompt(context, question))
    print("\nLLM Judge Prompt:\n", prompt_gen.llm_judge_prompt(question, expected_answer, generated_answer))
