import pandas as pd
import requests
from RAG import RAGRetriever
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load test datasets
test_data_1 = pd.read_csv("dataset/test_multiple healthcare data.csv")
test_data_2 = pd.read_csv("dataset/kaggle_healthcare data.csv")

# Combine both datasets for full evaluation
test_data = pd.concat([test_data_1, test_data_2])

# Initialize RAG retriever
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()

# Evaluation metrics
total = len(test_data)
correct_retrievals = 0
correct_answers = 0
bleu_scores = []
rouge1_scores = []
rougeL_scores = []

for index, row in test_data.iterrows():
    test_query = row["question"]

    # Extract actual correct answer (map cop to the correct choice)
    if "cop" in row:
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        ground_truth = choices[int(row["cop"]) - 1]  # Convert cop (1,2,3,4) to actual text
    else:
        ground_truth = row["short_answer"]  # Use short answer from Kaggle dataset

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

        # Check if generated answer contains the correct option
        if str(ground_truth) in generated_answer:
            correct_answers += 1

        # Compute BLEU Score
        bleu_score = sentence_bleu([ground_truth.split()], generated_answer.split())
        bleu_scores.append(bleu_score)

        # Compute ROUGE Score
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth, generated_answer)
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rougeL_scores.append(rouge_scores["rougeL"].fmeasure)

# Compute overall scores
retrieval_accuracy = correct_retrievals / total * 100
answer_accuracy = correct_answers / total * 100
avg_bleu = sum(bleu_scores) / total * 100
avg_rouge1 = sum(rouge1_scores) / total * 100
avg_rougeL = sum(rougeL_scores) / total * 100

# Display results
print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
print(f"LLM Answer Accuracy: {answer_accuracy:.2f}%")
print(f"Average BLEU Score: {avg_bleu:.2f}")
print(f"Average ROUGE-1 Score: {avg_rouge1:.2f}")
print(f"Average ROUGE-L Score: {avg_rougeL:.2f}")
