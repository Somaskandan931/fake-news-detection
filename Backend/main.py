import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import shap
import numpy as np
from pymongo import MongoClient
import dill
import os
import logging
import gdown  # ✅ To download files from Google Drive

# ========== 🔹 Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ========== 🔹 Define Google Drive File IDs ==========
GDRIVE_MODEL_ID = "1uN2siOjOkwjEKYr9yAH2bhjdykWqzvTX"  # Replace with actual ID
GDRIVE_SHAP_EXPLAINER_ID = "1p_UPJLnYCYrPUc-HiPWtRC7G_-fmPZRq"
GDRIVE_SHAP_VALUES_ID = "1-4Hd-DXoveFyjz1TQoZ6rX7dKOGny2hm"

# ========== 🔹 Define Local File Paths ==========
MODEL_PATH = "bert_lstm_model.pth"
SHAP_EXPLAINER_PATH = "shap_explainer.pkl"
SHAP_VALUES_PATH = "shap_values.pkl"

# ========== 🔹 Function to Download Files ==========
def download_from_gdrive(file_id, output_path):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(output_path):
        logging.info(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        logging.info(f"✅ Downloaded: {output_path}")
    else:
        logging.info(f"✅ {output_path} already exists. Skipping download.")

# ✅ Download model & SHAP files if missing
os.makedirs("models", exist_ok=True)  # Ensure "models" directory exists
download_from_gdrive(GDRIVE_MODEL_ID, MODEL_PATH)
download_from_gdrive(GDRIVE_SHAP_EXPLAINER_ID, SHAP_EXPLAINER_PATH)
download_from_gdrive(GDRIVE_SHAP_VALUES_ID, SHAP_VALUES_PATH)

# ========== 🔹 Load Model and Tokenizer ==========
class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, bidirectional=False, dropout=0.3):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output.last_hidden_state[:, 0, :]
        lstm_out, _ = self.lstm(hidden_state.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]
        output = self.dropout(lstm_out)
        output = self.fc(output)
        return self.sigmoid(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ✅ Load Model
model = BertLSTMClassifier(hidden_size=256, bidirectional=False).to(device)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    logging.info("✅ Model loaded successfully!")
else:
    logging.error("❌ Model file not found!")
    raise FileNotFoundError("Model file not found. Please check the path.")

# ✅ Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ========== 🔹 MongoDB Connection ==========
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["fake_news_db"]
    feedback_collection = db["feedback"]
    subscriptions_collection = db["subscriptions"]
    logging.info("✅ Connected to MongoDB!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError("MongoDB connection failed.")

# ========== 🔹 Load SHAP Explainer ==========
shap_explainer = None
precomputed_shap_values = None

try:
    if os.path.exists(SHAP_EXPLAINER_PATH):
        with open(SHAP_EXPLAINER_PATH, "rb") as f:
            shap_explainer = dill.load(f)
        logging.info("✅ SHAP explainer loaded!")

    if os.path.exists(SHAP_VALUES_PATH):
        with open(SHAP_VALUES_PATH, "rb") as f:
            precomputed_shap_values = dill.load(f)
        logging.info("✅ Precomputed SHAP values loaded!")
except Exception as e:
    logging.error(f"❌ Error loading SHAP files: {e}")

# ========== 🔹 API Endpoints ==========

class NewsRequest(BaseModel):
    text: str

@app.post("/verify-news")
def verify_news(request: NewsRequest):
    try:
        inputs = tokenizer(request.text, padding=True, truncation=True, return_tensors="pt", max_length=256)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            prediction = model(**inputs).item()

        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round(prediction, 2)

        explanation = "SHAP explainer not available."
        if precomputed_shap_values and request.text in precomputed_shap_values:
            explanation = precomputed_shap_values[request.text]
        elif shap_explainer:
            shap_values = shap_explainer.shap_values(inputs["input_ids"].cpu().numpy())
            explanation = shap_values.tolist()

        return {"label": label, "confidence": confidence, "explanation": explanation}

    except Exception as e:
        logging.error(f"❌ Error in /verify-news: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying news: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
