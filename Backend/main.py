import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import shap
import numpy as np
import pickle
from pymongo import MongoClient
import logging
import os
import gc
from huggingface_hub import hf_hub_download, login
from transformers import AutoModelForSequenceClassification
import bitsandbytes as bnb  # 8-bit quantization

torch.cuda.empty_cache()

# ========== 🔹 Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== 🔹 Load Environment Variables ==========
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "fake_news_detection"

# Authenticate Hugging Face API
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
    logging.info("✅ Hugging Face authentication successful.")
else:
    logging.warning("⚠️ Hugging Face token not found. Ensure HUGGINGFACE_TOKEN is set.")

# ========== 🔹 Load Model from Hugging Face Hub ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"✅ Using device: {device}")

try:
    MODEL_PATH = hf_hub_download("Samaskandan/fake_news_detection", "new_bert_lstm_model_quantized.pth")
except Exception as e:
    logging.error(f"❌ Failed to download model: {e}")
    raise RuntimeError("Failed to load model from Hugging Face.")

# Define BERT-LSTM model with 8-bit quantization
class BertLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=1):
        super(BertLSTM, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            torch_dtype=torch.float16,  # Load in 16-bit precision
            load_in_8bit=True,  # Quantize to 8-bit
            device_map="auto"  # Automatically assign to GPU/CPU
        )
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(bert_output)
        output = self.fc(lstm_output[:, -1, :])
        return output

# Load tokenizer and model
try:
    model = BertLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    raise RuntimeError("Failed to initialize model.")

# ========== 🔹 Load SHAP Explainer in CPU ==========
try:
    SHAP_EXPLAINER_PATH = hf_hub_download("Samaskandan/fake_news_detection", "shap_explainer.pkl")
    SHAP_VALUES_PATH = hf_hub_download("Samaskandan/fake_news_detection", "shap_values.pkl")

    with open(SHAP_EXPLAINER_PATH, "rb") as explainer_file:
        shap_explainer = pickle.load(explainer_file)
    with open(SHAP_VALUES_PATH, "rb") as shap_file:
        shap_values = pickle.load(shap_file)

    # Move SHAP computations to CPU
    shap_explainer.model = shap_explainer.model.to("cpu")
    logging.info("✅ SHAP explainer loaded on CPU successfully!")
except Exception as e:
    logging.error(f"❌ Error loading SHAP explainer: {e}")
    raise RuntimeError("Failed to load SHAP explainer.")

# ========== 🔹 MongoDB Connection ==========
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    feedback_collection = db["feedback"]
    subscriptions_collection = db["subscriptions"]
    logging.info("✅ Connected to MongoDB!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError("MongoDB connection failed.")

# ========== 🔹 API Endpoints ==========
class NewsRequest(BaseModel):
    text: str

@app.post("/verify-news")
def verify_news(request: NewsRequest):
    """Verify news text and return the classification result with SHAP explanation."""
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs).squeeze()
            prediction = torch.sigmoid(outputs).item()

        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction, 2)

        # Move SHAP computation to CPU
        explanation = shap_explainer.shap_values([request.text], nsamples=100)

        # Free memory after request
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"label": label, "confidence": confidence, "explanation": explanation.tolist()}
    except Exception as e:
        logging.error(f"❌ Error in /verify-news: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying news: {str(e)}")

# ========== 🔹 Run FastAPI App ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use Render's assigned PORT or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
