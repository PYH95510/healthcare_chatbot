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
        """Generates the evaluation prompt used in LLM_judge.py."""
        return (
            "You are an AI judge evaluating chatbot responses. \n"
            "Please rate the chatbot's response on a scale from 1 to 10 based on:\n"
            "- **Relevance** (Is it on-topic?)\n"
            "- **Correctness** (Is it factually accurate?)\n"
            "- **Coherence** (Is it well-structured?)\n"
            "- **Completeness** (Does it fully answer the question?)\n"
            "- **Conciseness** (Is it the right length?)\n\n"
            f"Here is the **question**: {question}\n"
            f"The **expected correct answer** is: {expected_answer}\n"
            f"The **chatbot's response** is: {generated_answer}\n\n"
            "Please provide a **single number (1-10)** as your final score."
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
