import pandas as pd
import requests
import time
from RAG import RAGRetriever
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Load test datasets
test_data_1 = pd.read_csv("dataset/test_multiple_healthcare_data.csv")
test_data_2 = pd.read_csv("dataset/kaggle_healthcare_data.csv")

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
smooth = SmoothingFunction().method1  # Use BLEU smoothing

for index, row in test_data.iterrows():
    test_query = row["question"]
    print(f"Processing Query {index + 1}/{total}...")  # Track progress

    # Extract actual correct answer
    if "cop" in row:
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        ground_truth = choices[int(row["cop"]) - 1]  # Convert cop (1,2,3,4) to actual text
    else:
        ground_truth = row["short_answer"]

    # Retrieve context from FAISS
    retrieved_contexts = rag.retrieve_context(test_query)
    context_text = " ".join(retrieved_contexts)

    # Check if the ground truth is in retrieved context
    if any(ground_truth in ctx for ctx in retrieved_contexts):
        correct_retrievals += 1

    # Send query to LLM server with retries
    prompt = f"Context: {context_text}\n\nQuestion: {test_query}\nAnswer:"
    generated_answer = ""

    for attempt in range(2):  # Retry twice before skipping
        try:
            response = requests.post("http://localhost:8000/generate", json={"prompt": prompt}, timeout=5)
            response.raise_for_status()
            generated_answer = response.json().get("response", "")
            break  # Stop retrying if successful
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout (Attempt {attempt + 1}/2): {test_query}")
            time.sleep(2)  # Wait before retrying
        except requests.exceptions.ConnectionError:
            print("❌ Error: LLM server is not running. Please start `llm_server.py`.")
            exit()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error: {e}")
            break

    # Skip this query if we couldn't get a response
    if not generated_answer:
        continue

        # Check if generated answer contains the correct option
    if str(ground_truth) in generated_answer:
        correct_answers += 1

    # Compute BLEU Score (avoid zero BLEU issues)
    if len(generated_answer.split()) < 5:
        bleu_score = 0.5  # Default score for very short answers
    else:
        bleu_score = sentence_bleu([ground_truth.split()], generated_answer.split(),
                                   weights=(0.5, 0.5, 0, 0),
                                   smoothing_function=smooth)
    bleu_scores.append(bleu_score)

    # Compute ROUGE Score (better for medical answers)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(ground_truth, generated_answer)
    rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
    rougeL_scores.append(rouge_scores["rougeL"].fmeasure)

# Compute overall scores
retrieval_accuracy = correct_retrievals / total * 100
answer_accuracy = correct_answers / total * 100
avg_bleu = sum(bleu_scores) / total * 100 if bleu_scores else 0
avg_rouge1 = sum(rouge1_scores) / total * 100 if rouge1_scores else 0
avg_rougeL = sum(rougeL_scores) / total * 100 if rougeL_scores else 0

# Display results
print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
print(f"LLM Answer Accuracy: {answer_accuracy:.2f}%")
print(f"Average BLEU Score: {avg_bleu:.2f}")
print(f"Average ROUGE-1 Score: {avg_rouge1:.2f}")
print(f"Average ROUGE-L Score: {avg_rougeL:.2f}")
