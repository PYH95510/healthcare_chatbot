import streamlit as st
import requests
from RAG import RAGRetriever
from prompt import PromptGenerator
from streamlit_chat import message  # For a chat-like UI

# ------------------------------
# Load FAISS index for context retrieval
# ------------------------------
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()

# Create an instance of PromptGenerator
prompt_gen = PromptGenerator()

# ------------------------------
# Sidebar: Chatbot style selection (Standard or Chain-of-Thought)
# ------------------------------
st.sidebar.title("Chatbot Settings")
if "chatbot_style" not in st.session_state:
    st.session_state.chatbot_style = "Standard"
selected_style = st.sidebar.radio(
    "Select Chatbot Style:",
    ("Standard", "Chain-of-Thought"),
    index=0,
    key="style_radio"
)
st.session_state.chatbot_style = selected_style

# ------------------------------
# Initialize conversation history in session state
# ------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ------------------------------
# Main UI: Title and description
# ------------------------------
st.title("🩺 AI Healthcare Chatbot")
st.write("Ask me any medical question, and I'll provide expert-backed answers!")

# ------------------------------
# Chat input form
# ------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your question:")
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    # Append the user's new question to conversation history
    st.session_state.conversation_history.append(f"User: {user_query}")

    # Retrieve additional context from the FAISS index for the new question
    retrieved_contexts = rag.retrieve_context(user_query)
    context_text = " ".join(retrieved_contexts)

    # Generate the full prompt with the entire conversation (for LLM context)
    full_prompt = prompt_gen.cot_prompt(
        st.session_state.conversation_history,
        context_text,
        user_query
    )

    try:
        # Send the request to your local LLM server
        response = requests.post(
            "http://localhost:8000/generate",
            json={"prompt": full_prompt},
            timeout=30
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not get a response from the LLM server. Details: {e}")
        answer = "Sorry, there was an error processing your request."

    # ------------------------------
    # Post-processing the final answer
    # ------------------------------
    # 1. If it contains "Final Answer:", keep only what's after it.
    lower_answer = answer.lower()
    if "final answer:" in lower_answer:
        start_index = lower_answer.index("final answer:") + len("final answer:")
        answer = answer[start_index:].strip()

    # 2. Remove leftover tokens or tags the model might produce
    unwanted_tokens = [
        "</s>", "<s>",
        "<s>assistant:", "<s>user:"
    ]
    for token in unwanted_tokens:
        answer = answer.replace(token, "")

    # 3. Sometimes the model restates "User:" or "Human:" within the final answer.
    #    We split on those to remove them, plus everything after.
    #    If you want to keep partial text, you can adjust how you do the split.
    #    For example, to remove only the "User:" line, do a simple replace.
    if "user:" in answer.lower():
        answer = answer.split("User:")[0].strip()
    if "human:" in answer.lower():
        answer = answer.split("Human:")[0].strip()
    if "ai:" in answer.lower():
        answer = answer.split("AI:")[0].strip()

    # 4. Final cleanup
    answer = answer.strip()

    # 5. Now store the cleaned answer in the conversation history (for LLM context)
    st.session_state.conversation_history.append(f"AI: {answer}")

# ------------------------------
# Display conversation
# ------------------------------
# If you DO want to show the entire conversation as a chat, you can keep this loop.
# But you only see user queries + final answers, because we appended them in that format.
if st.session_state.conversation_history:
    for msg in st.session_state.conversation_history:
        if msg.startswith("User:"):
            # Show only the user's question (without "User:" prefix)
            message(msg[6:], is_user=True)
        elif msg.startswith("AI:"):
            # Show only the AI's final answer (without "AI:" prefix)
            message(msg[4:])
