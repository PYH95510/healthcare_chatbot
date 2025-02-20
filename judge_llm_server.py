import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLM Judge Model (You can use GPT-4 via OpenAI API or an open-source LLM)
model_name = "meta-llama/Llama-3-8B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# FastAPI server
app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/judge")
async def judge_response(query: Query):
    """Generate AI score based on chatbot response evaluation."""
    inputs = tokenizer(query.prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    score = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure output is just a number (1-10)
    try:
        score = float(score.strip())
        if 1 <= score <= 10:
            return {"score": score}
        else:
            return {"score": 5.0}  # Default score if LLM outputs a bad response
    except ValueError:
        return {"score": 5.0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
