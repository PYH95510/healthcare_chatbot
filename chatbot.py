import streamlit as st
import requests
from RAG import RAGRetriever
from prompt import PromptGenerator

# Load FAISS index
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()

prompt_gen = PromptGenerator()
# Streamlit UI
st.title("🩺 AI Healthcare Chatbot")
st.write("Ask me any medical question, and I'll provide expert-backed answers!")

# User input
user_query = st.text_input("Enter your question:")

if st.button("Ask AI") and user_query:
    # Retrieve relevant context from FAISS
    retrieved_contexts = rag.retrieve_context(user_query)
    context_text = " ".join(retrieved_contexts)

    prompt_text = prompt_gen.chatbot_prompt(context_text,user_query)
    #prompt = f"Context: {context_text}\n\nQuestion: {user_query}\nAnswer:"

    # Send query to the LLM server
    response = requests.post("http://localhost:8000/generate", json={"prompt": prompt_text})

    if response.status_code == 200:
        st.subheader("Chatbot Response:")
        st.write(response.json()["response"])
    else:
        st.error("Error: Could not get a response from the LLM server.")
