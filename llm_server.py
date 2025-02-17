import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prevent resource leaks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model name
model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"  # Run on CPU
)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_response(query: Query):
    """Generate AI response using LLaMA 3.2-1B."""
    inputs = tokenizer(query.prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)  # Fix: Use max_new_tokens

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
