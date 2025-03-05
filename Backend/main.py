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

# ========== Load Environment Variables from .env (Render) ==========
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv( "HUGGINGFACE_TOKEN" )
MONGO_URI = os.getenv( "MONGO_URI" )
PORT = int( os.getenv( "PORT", 10000 ) )  # Default port 10000 if not specified
DATABASE_NAME = os.getenv( "DATABASE_NAME", "fake_news_detection" )

# Ensure essential environment variables exist
if not HUGGINGFACE_TOKEN or not MONGO_URI :
    raise RuntimeError( "❌ Missing required environment variables! Check your .env file on Render." )

# Empty CUDA cache
torch.cuda.empty_cache()

# ========== Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s" )

# Authenticate Hugging Face API
login( token=HUGGINGFACE_TOKEN )
logging.info( "✅ Hugging Face authentication successful." )

# ========== Load Model from Hugging Face Hub ==========
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
logging.info( f"✅ Using device: {device}" )

try :
    MODEL_PATH = hf_hub_download( "Samaskandan/fake_news_detection", "bert_lstm_model_quantized.pth",
                                  force_download=True )
except Exception as e :
    logging.error( f"❌ Failed to download model: {e}" )
    raise RuntimeError( "Failed to load model from Hugging Face." )


# Define Model Class
class BertLSTM( nn.Module ) :
    def __init__ ( self, hidden_dim=256, num_classes=1 ) :
        super( BertLSTM, self ).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained( "bert-base-uncased" )
        self.lstm = nn.LSTM( input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True )
        self.fc = nn.Linear( hidden_dim * 2, num_classes )

    def forward ( self, input_ids, attention_mask ) :
        with torch.no_grad() :
            bert_output = self.bert( input_ids=input_ids, attention_mask=attention_mask ).logits
        lstm_output, _ = self.lstm( bert_output.unsqueeze( 1 ) )
        return self.fc( lstm_output[:, -1, :] )


# Load Model & Tokenizer
model = BertLSTM().to( device )
model.load_state_dict( torch.load( MODEL_PATH, map_location=device ) )
model.eval()
tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased" )
logging.info( "✅ Model loaded successfully!" )

# ========== Load SHAP Explainer & SHAP Values ==========
try :
    SHAP_EXPLAINER_PATH = hf_hub_download( "Samaskandan/fake_news_detection", "shap_explainer_reduced.pkl",
                                           force_download=True )
    SHAP_VALUES_PATH = hf_hub_download( "Samaskandan/fake_news_detection", "shap_values_compressed.pbz2", force_download=True )

    with open( SHAP_EXPLAINER_PATH, "rb" ) as explainer_file :
        shap_explainer = pickle.load( explainer_file, errors="ignore" )
    shap_explainer.model = shap_explainer.model.to( "cpu" )

    with open( SHAP_VALUES_PATH, "rb" ) as shap_values_file :
        shap_values = pickle.load( shap_values_file, errors="ignore" )

    logging.info( "✅ SHAP explainer and SHAP values loaded successfully!" )
except Exception as e :
    logging.error( f"❌ Error loading SHAP explainer or SHAP values: {e}" )
    shap_explainer, shap_values = None, None

# ========== MongoDB Connection ==========
client = MongoClient( MONGO_URI )
db = client[DATABASE_NAME]
feedback_collection = db["feedback"]
subscriptions_collection = db["subscriptions"]
logging.info( "✅ Connected to MongoDB!" )


# ========== API Endpoints ==========
class NewsRequest( BaseModel ) :
    text: str


@app.post( "/verify-news" )
def verify_news ( request: NewsRequest ) :
    """Verify news text and return classification result with SHAP explanation."""
    try :
        inputs = tokenizer( request.text, return_tensors="pt", truncation=True, padding=True, max_length=256 )
        inputs = {key : value.to( device ) for key, value in inputs.items()}

        with torch.no_grad() :
            outputs = model( **inputs ).squeeze()
            prediction = torch.sigmoid( outputs ).item()

        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round( prediction, 2 )

        explanation = None
        shap_contributions = None
        if shap_explainer and shap_values :
            explanation = shap_explainer.shap_values( [request.text], nsamples=100 )
            shap_contributions = shap_values.get( request.text, None )  # Retrieve precomputed SHAP values

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"label" : label, "confidence" : confidence, "explanation" : explanation,
                "shap_values" : shap_contributions}
    except Exception as e :
        logging.error( f"❌ Error in /verify-news: {e}" )
        raise HTTPException( status_code=500, detail=f"Error verifying news: {str( e )}" )


@app.post( "/subscribe" )
def subscribe ( phone_number: str = Form( ... ) ) :
    """Subscribe a user to fake news alerts."""
    try :
        subscriptions_collection.insert_one( {"phone_number" : phone_number} )
        return {"message" : "Successfully subscribed!"}
    except Exception as e :
        logging.error( f"❌ Subscription error: {e}" )
        raise HTTPException( status_code=500, detail="Error subscribing user." )


@app.post( "/feedback" )
def submit_feedback ( news_text: str = Form( ... ), feedback: str = Form( ... ) ) :
    """Submit feedback for a news article."""
    try :
        feedback_collection.insert_one( {"news_text" : news_text, "feedback" : feedback} )
        return {"message" : "Feedback submitted successfully!"}
    except Exception as e :
        logging.error( f"❌ Feedback error: {e}" )
        raise HTTPException( status_code=500, detail="Error submitting feedback." )


# ========== Run FastAPI App ==========
if __name__ == "__main__" :
    logging.info( f"🚀 Starting FastAPI server on port {PORT}..." )
    uvicorn.run( "main:app", host="0.0.0.0", port=PORT, log_level="info" )
