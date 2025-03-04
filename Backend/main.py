import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoModelForSequenceClassification
import shap
import pickle
from pymongo import MongoClient
import logging
import os
import gc
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv

# ========== Load Environment Variables from .env ==========
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
PORT = int(os.getenv("PORT", 10000))  # Default port 10000 if not specified
DATABASE_NAME = "fake_news_detection"

# Empty CUDA cache
torch.cuda.empty_cache()

# ========== Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Authenticate Hugging Face API
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
    logging.info("✅ Hugging Face authentication successful.")
else:
    logging.warning("⚠️ Hugging Face token not found.")

# ========== Load Model from Hugging Face Hub ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"✅ Using device: {device}")

try:
    MODEL_PATH = hf_hub_download("Samaskandan/fake_news_detection", "new_bert_lstm_model_quantized.pth")
except Exception as e:
    logging.error(f"❌ Failed to download model: {e}")
    raise RuntimeError("Failed to load model from Hugging Face.")

# Define Model Class
class BertLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=1):
        super(BertLSTM, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        lstm_output, _ = self.lstm(bert_output.unsqueeze(1))
        return self.fc(lstm_output[:, -1, :])

# Load Model & Tokenizer with Exception Handling
try:
    model = BertLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logging.info("✅ Model loaded successfully!")
except UnicodeDecodeError:
    logging.error("❌ Model file encoding error! Try re-downloading the model.")
    raise RuntimeError("Model file corrupted. Please re-download.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    raise RuntimeError("Failed to initialize model.")

# ========== Load SHAP Explainer in CPU ==========
try:
    SHAP_EXPLAINER_PATH = hf_hub_download("Samaskandan/fake_news_detection", "shap_explainer.pkl")
    with open(SHAP_EXPLAINER_PATH, "rb") as explainer_file:
        shap_explainer = pickle.load(explainer_file, encoding="latin1")  # Fix UnicodeDecodeError
    shap_explainer.model = shap_explainer.model.to("cpu")
    logging.info("✅ SHAP explainer loaded successfully!")
except UnicodeDecodeError:
    logging.error("❌ SHAP file encoding error! Try re-downloading.")
    shap_explainer = None
except Exception as e:
    logging.error(f"❌ Error loading SHAP explainer: {e}")
    shap_explainer = None  # Avoid crashes if SHAP fails

# ========== MongoDB Connection ==========
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    feedback_collection = db["feedback"]
    subscriptions_collection = db["subscriptions"]
    logging.info("✅ Connected to MongoDB!")
except Exception as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    raise RuntimeError("MongoDB connection failed.")

# ========== API Endpoints ==========
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

        # Compute SHAP Explanation (if available)
        explanation = None
        if shap_explainer:
            explanation = shap_explainer.shap_values([request.text], nsamples=100)

        # Free memory
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"label": label, "confidence": confidence, "explanation": explanation}
    except Exception as e:
        logging.error(f"❌ Error in /verify-news: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying news: {str(e)}")

@app.post("/subscribe")
def subscribe(phone_number: str = Form(...)):
    """Subscribe a user to fake news alerts."""
    try:
        subscriptions_collection.insert_one({"phone_number": phone_number})
        return {"message": "Successfully subscribed!"}
    except Exception as e:
        logging.error(f"❌ Subscription error: {e}")
        raise HTTPException(status_code=500, detail="Error subscribing user.")

@app.post("/feedback")
def submit_feedback(news_text: str = Form(...), feedback: str = Form(...)):
    """Submit feedback for a news article."""
    try:
        feedback_collection.insert_one({"news_text": news_text, "feedback": feedback})
        return {"message": "Feedback submitted successfully!"}
    except Exception as e:
        logging.error(f"❌ Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Error submitting feedback.")

# ========== Run FastAPI App ==========
if __name__ == "__main__":
    logging.info(f"🚀 Starting FastAPI server on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
