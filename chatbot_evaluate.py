import pandas as pd
import requests
import re
from RAG import RAGRetriever
from prompt import PromptGenerator

def extract_numeric_score(response_text):
    """
    Extract the first occurrence of a numeric value (integer or float)
    from a string. Returns None if no number is found.
    """
    match = re.search(r"(\d+(\.\d+)?)", response_text)
    if match:
        return float(match.group(1))
    return None

# Load test datasets
test_data_1 = pd.read_csv("dataset/test_multiple_healthcare_data.csv")
test_data_2 = pd.read_csv("dataset/kaggle_healthcare_data.csv")

# Combine both datasets for full evaluation
test_data = pd.concat([test_data_1, test_data_2])

# Initialize RAG retriever
rag = RAGRetriever("dataset/kaggle_healthcare data.csv", "dataset/multiple healthcare data.csv")
rag.load_faiss_index()

# Initialize prompt generator
prompt_gen = PromptGenerator()

# Evaluation metrics
total = len(test_data)
correct_retrievals = 0
llm_scores = []

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

    # Construct prompt for chatbot
    chatbot_prompt = prompt_gen.chatbot_prompt(context_text, test_query)

    try:
        # Send query + retrieved context to chatbot LLM (localhost:8000)
        chatbot_response = requests.post("http://localhost:8000/generate",
                                         json={"prompt": chatbot_prompt}, timeout=30)
        chatbot_response.raise_for_status()
        chatbot_answer = chatbot_response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error in chatbot response: {e}")
        chatbot_answer = ""

    # Construct evaluation prompt using chatbot's answer
    eval_prompt = prompt_gen.llm_judge_prompt(test_query, ground_truth, chatbot_answer)

    try:
        # Send chatbot-generated answer to evaluation LLM (localhost:8001)
        response = requests.post("http://localhost:8001/evaluate",
                                 json={"prompt": eval_prompt}, timeout=30)
        response.raise_for_status()
        response_text = response.json().get("response", "").strip()
        score = extract_numeric_score(response_text)
        if score is not None:
            llm_scores.append(score)
        else:
            print(f"⚠️ Unable to parse numeric score from LLM evaluation response: '{response_text}'")
    except requests.exceptions.ConnectionError:
        print("Error: LLM evaluation server is not running. Please start the server.")
        exit()
    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout: LLM evaluation took too long for query: {test_query}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    except ValueError:
        print(f"⚠️ Invalid score received from LLM: {response_text}")

# Compute overall scores
retrieval_accuracy = correct_retrievals / total * 100
avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0

# Display results
print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}%")
print(f"Average LLM Evaluation Score: {avg_llm_score:.2f}/10")
