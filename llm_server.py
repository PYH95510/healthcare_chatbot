from fastapi import FastAPI
from pydantic import BaseModel
import requests

# Initialize FastAPI app
app = FastAPI()

# Define the Ollama model name
OLLAMA_MODEL = "llama3.2-1B-instruct"


class Query(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_response(query: Query):
    """Generate AI response using Ollama."""

    # Send the request to Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": query.prompt, "stream": False}
    )

    # Handle errors
    if response.status_code != 200:
        return {"error": "Failed to generate response from Ollama"}

    # Extract response text
    result = response.json()
    generated_text = result.get("response", "").strip()

    return {"response": generated_text}


# Run the server on Port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
