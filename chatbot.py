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

# ------------------------------
# Helper: Generate the conversation prompt for your LLM server
# ------------------------------
def generate_conversation_prompt(history, context, user_input, use_cot=False):
    conversation_text = "\n".join(history)
    prompt_header = "You are an AI healthcare chatbot conversing with a user. "
    if context:
        prompt_header += f"Reference the following context when needed:\n{context}\n\n"
    prompt_header += "Conversation so far:\n" + conversation_text + "\n"
    prompt_header += f"User: {user_input}\nAI:"
    if use_cot:
        prompt_header += "\nLet's think step by step before answering."
    return prompt_header

# ------------------------------
# On submission: Process user query and get response from your local LLM server (ollma)
# ------------------------------
if submitted and user_query:
    # Append the user's query to the conversation history
    st.session_state.conversation_history.append(f"User: {user_query}")

    # Retrieve additional context from your FAISS index
    retrieved_contexts = rag.retrieve_context(user_query)
    context_text = " ".join(retrieved_contexts)

    # Check if Chain-of-Thought style is selected
    use_cot = st.session_state.chatbot_style == "Chain-of-Thought"

    # Generate full prompt including conversation history and retrieved context
    full_prompt = generate_conversation_prompt(
        st.session_state.conversation_history, context_text, user_query, use_cot
    )

    try:
        # Send the prompt to your local LLM server (ollma) and get the AI's answer
        response = requests.post("http://localhost:8000/generate", json={"prompt": full_prompt}, timeout=30)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not get a response from the LLM server. Details: {e}")
        answer = "Sorry, there was an error processing your request."

    # Append the AI's answer to conversation history
    st.session_state.conversation_history.append(f"AI: {answer}")

# ------------------------------
# Display conversation history using a chat-like interface
# ------------------------------
if st.session_state.conversation_history:
    for msg in st.session_state.conversation_history:
        if msg.startswith("User:"):
            message(msg[6:], is_user=True)
        elif msg.startswith("AI:"):
            message(msg[4:])
