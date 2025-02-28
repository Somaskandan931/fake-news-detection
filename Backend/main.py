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
import gdown  # ✅ Added for Google Drive downloads

# ========== 🔹 Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ========== 🔹 Model Paths ==========
MODEL_DIR = "Backend/models"  # ✅ Ensure the correct path
MODEL_PATH = os.path.join(MODEL_DIR, "bert_lstm_model.pth")
SHAP_EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
SHAP_VALUES_PATH = os.path.join(MODEL_DIR, "shap_values.pkl")

# ========== 🔹 Google Drive File IDs ==========
MODEL_DRIVE_ID = "1uN2siOjOkwjEKYr9yAH2bhjdykWqzvTX"
SHAP_EXPLAINER_DRIVE_ID = "1-4Hd-DXoveFyjz1TQoZ6rX7dKOGny2hm"
SHAP_VALUES_DRIVE_ID = "1p_UPJLnYCYrPUc-HiPWtRC7G_-fmPZRq"

os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ Ensure models folder exists

# ✅ Function to Download from Google Drive
def download_from_drive(file_id, save_path):
    """Downloads a file from Google Drive if it's missing."""
    if not os.path.exists(save_path):
        logging.info(f"📥 Downloading {save_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False)
        logging.info(f"✅ Downloaded {save_path}")

# ✅ Download Models If Missing
download_from_drive(MODEL_DRIVE_ID, MODEL_PATH)
download_from_drive(SHAP_EXPLAINER_DRIVE_ID, SHAP_EXPLAINER_PATH)
download_from_drive(SHAP_VALUES_DRIVE_ID, SHAP_VALUES_PATH)

# ========== 🔹 Load Model ==========
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
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()
logging.info("✅ Model loaded successfully!")

# ✅ Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ========== 🔹 MongoDB Connection ==========
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")  # ✅ Use Render env variable
DATABASE_NAME = "fake_news_db"

try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    feedback_collection = db["feedback"]
    subscriptions_collection = db["subscriptions"]
    logging.info("✅ Connected to MongoDB!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError("MongoDB connection failed.")

# ✅ Load SHAP Explainer
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

# ✅ API Endpoints
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

        # SHAP Explainability
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

@app.post("/subscribe")
def subscribe(phone_number: str = Form(...)):
    try:
        if subscriptions_collection.find_one({"phone_number": phone_number}):
            raise HTTPException(status_code=400, detail="Already subscribed.")
        subscriptions_collection.insert_one({"phone_number": phone_number})
        return {"message": "Successfully subscribed!"}
    except Exception as e:
        logging.error(f"❌ Error in /subscribe: {e}")
        raise HTTPException(status_code=500, detail=f"Error subscribing: {str(e)}")

@app.post("/feedback")
def submit_feedback(news_text: str = Form(...), feedback: str = Form(...)):
    try:
        feedback_collection.insert_one({"news_text": news_text, "feedback": feedback})
        return {"message": "Feedback submitted!"}
    except Exception as e:
        logging.error(f"❌ Error in /feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
