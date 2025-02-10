# Install necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch



# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load the trained model from Google Drive
# ✅ Change this line:
model_path = "D:/Models/TextSummarizer-API/fine_tuned_t5"  # Use your correct local path

# ✅ Load model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

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

# ✅ Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
