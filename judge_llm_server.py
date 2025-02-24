from fastapi import FastAPI
from pydantic import BaseModel
import requests

# Initialize FastAPI for evaluation
app = FastAPI()

# Define the model name used in Ollama
OLLAMA_MODEL = "llama3.2-1B-instruct"


class EvaluationRequest(BaseModel):
    prompt: str


@app.post("/evaluate")
def evaluate_chatbot(request: EvaluationRequest):
    """Evaluates a chatbot response using the Ollama LLM as a judge."""

    # Send the evaluation prompt to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": request.prompt, "stream": False}
    )

    # Handle errors
    if response.status_code != 200:
        return {"error": "Failed to generate evaluation response from Ollama"}

    # Extract the generated score
    result = response.json()
    score = result.get("response", "").strip()

    return {"score": score}


# Run on Port 8001
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
