import streamlit as st
import requests
from RAG import RAGRetriever
from prompt import PromptGenerator

# Load FAISS index
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()
prompt_gen = PromptGenerator()

# Initialize conversation history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Streamlit UI
st.title("🩺 AI Healthcare Chatbot")
st.write("Ask me any medical question, and I'll provide expert-backed answers!")

# Display conversation history
for chat in st.session_state.conversation:
    st.write(chat)

# User input
user_query = st.text_input("Enter your question:")

# Let user select prompt style
prompt_style = st.radio("Select prompt style:", ("Standard", "Chain-of-Thought"))

if st.button("Ask AI") and user_query:
    # Append user's question to conversation history
    st.session_state.conversation.append(f"**User:** {user_query}")

    # Retrieve relevant context from FAISS
    retrieved_contexts = rag.retrieve_context(user_query)
    context_text = " ".join(retrieved_contexts)

    # Choose the prompt style
    if prompt_style == "Standard":
        prompt_text = prompt_gen.chatbot_prompt(context_text, user_query)
    else:
        prompt_text = prompt_gen.cot_prompt(context_text, user_query)

    # Optionally, append the prompt_text to conversation history (for debugging or context)
    # st.session_state.conversation.append(f"**Prompt:** {prompt_text}")

    # Send query to the LLM server
    try:
        response = requests.post("http://localhost:8000/generate", json={"prompt": prompt_text}, timeout=30)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not get a response from the LLM server. Details: {e}")
        answer = "Sorry, there was an error processing your request."

    # Append the bot's answer to conversation history
    st.session_state.conversation.append(f"**Bot:** {answer}")

    # Display the chatbot response
    st.subheader("Chatbot Response:")
    st.write(answer)
