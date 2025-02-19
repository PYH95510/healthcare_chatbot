import pandas as pd
import requests
from RAG import RAGRetriever

# Load test dataset
test_data = pd.read_csv("dataset/test_multiple healthcare data.csv")

# Initialize RAG retriever
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()

# Evaluation metrics
total = len(test_data)
correct_retrievals = 0
correct_answers = 0

for index, row in test_data.iterrows():
    test_query = row["question"]
    ground_truth = row["cop"]  # Expected correct option

    # Retrieve context from FAISS
    retrieved_contexts = rag.retrieve_context(test_query)
    context_text = " ".join(retrieved_contexts)

    # Check if the ground truth is in retrieved context
    if any(ground_truth in ctx for ctx in retrieved_contexts):
        correct_retrievals += 1

    # Send query + retrieved context to LLM server
    prompt = f"Context: {context_text}\n\nQuestion: {test_query}\nAnswer:"
    response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})

    if response.status_code == 200:
        generated_answer = response.json()["response"]
        if str(ground_truth) in generated_answer:
            correct_answers += 1

# Compute scores
retrieval_accuracy = correct_retrievals / total * 100
answer_accuracy = correct_answers / total * 100

# Display results
print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
print(f"LLM Answer Accuracy: {answer_accuracy:.2f}%")
