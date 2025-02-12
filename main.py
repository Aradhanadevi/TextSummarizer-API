# ✅ Install necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
from huggingface_hub import snapshot_download

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Define the model directory
MODEL_DIR = "fine_tuned_t5"

# ✅ Get Hugging Face token from Render environment variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# ✅ Download the model from Hugging Face if not available
if not os.path.exists(MODEL_DIR):
    print("📥 Downloading model from Hugging Face...")
    snapshot_download(
        repo_id="ara0014/TextSummarizer-T5",
        local_dir=MODEL_DIR,
        revision="main",
        use_auth_token=hf_token  # 🔹 Authenticate with token
    )

# ✅ Load the model from the local directory
print("🔄 Loading model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define request model
class SummaryRequest(BaseModel):
    text: str

# ✅ Summarization API Endpoint
@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    input_text = "summarize: " + request.text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}

# ✅ Run the FastAPI server with the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT environment variable
    print(f"🚀 Running FastAPI on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
