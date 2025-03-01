import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import shap
import numpy as np
import pickle
from pymongo import MongoClient
import logging
import os
from huggingface_hub import hf_hub_download

# ========== 🔹 Initialize FastAPI and Logging ==========
app = FastAPI()
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s" )

# ========== 🔹 Load Model from Hugging Face Hub ==========
MODEL_PATH = hf_hub_download( repo_id="Samaskandan/fake_news_detection", filename="bert_lstm_model.pth" )
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
logging.info( f"Using device: {device}" )


# Define BERT-LSTM model
class BertLSTM( nn.Module ) :
    def __init__ ( self, bert_model, hidden_dim, num_classes ) :
        super( BertLSTM, self ).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM( input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True )
        self.fc = nn.Linear( hidden_dim * 2, num_classes )

    def forward ( self, input_ids, attention_mask ) :
        with torch.no_grad() :
            bert_output = self.bert( input_ids=input_ids, attention_mask=attention_mask ).last_hidden_state
        lstm_output, _ = self.lstm( bert_output )
        output = self.fc( lstm_output[:, -1, :] )
        return output


# Load tokenizer and model
bert_model = torch.hub.load( 'huggingface/pytorch-transformers', 'model', 'bert-base-uncased' )
model = BertLSTM( bert_model, hidden_dim=256, num_classes=1 ).to( device )
model.load_state_dict( torch.load( MODEL_PATH, map_location=device ) )
model.eval()
tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased" )
logging.info( "✅ Model loaded successfully from Hugging Face!" )

# ========== 🔹 Load SHAP Explainer ==========
try :
    SHAP_EXPLAINER_PATH = hf_hub_download( repo_id="Samaskandan/fake_news_detection", filename="shap_explainer.pkl" )
    SHAP_VALUES_PATH = hf_hub_download( repo_id="Samaskandan/fake_news_detection", filename="shap_values.pkl" )

    with open( SHAP_EXPLAINER_PATH, "rb" ) as explainer_file :
        shap_explainer = pickle.load( explainer_file )
    with open( SHAP_VALUES_PATH, "rb" ) as shap_file :
        shap_values = pickle.load( shap_file )
    logging.info( "✅ SHAP explainer and values loaded successfully!" )
except Exception as e :
    logging.error( f"❌ Error loading SHAP explainer: {e}" )
    raise RuntimeError( "Failed to load SHAP explainer." )

# ========== 🔹 MongoDB Connection ==========
MONGO_URI = os.getenv( "MONGO_URI", "mongodb://localhost:27017/" )
DATABASE_NAME = "fake_news_detection"

try :
    client = MongoClient( MONGO_URI )
    db = client[DATABASE_NAME]
    feedback_collection = db["feedback"]
    subscriptions_collection = db["subscriptions"]
    logging.info( "✅ Connected to MongoDB!" )
except Exception as e :
    logging.error( f"❌ MongoDB connection failed: {e}" )
    raise RuntimeError( "MongoDB connection failed." )


# ✅ API Endpoints
class NewsRequest( BaseModel ) :
    text: str


@app.post( "/verify-news" )
def verify_news ( request: NewsRequest ) :
    try :
        inputs = tokenizer( request.text, return_tensors="pt", truncation=True, padding=True, max_length=256 )
        inputs = {key : value.to( device ) for key, value in inputs.items()}

        with torch.no_grad() :
            outputs = model( **inputs ).squeeze()
            prediction = torch.sigmoid( outputs ).item()

        label = "Real" if prediction >= 0.5 else "Fake"
        confidence = round( prediction, 2 )

        # SHAP Explainability
        explanation = shap_explainer.shap_values( [request.text] )

        return {"label" : label, "confidence" : confidence, "explanation" : explanation.tolist()}
    except Exception as e :
        logging.error( f"❌ Error in /verify-news: {e}" )
        raise HTTPException( status_code=500, detail=f"Error verifying news: {str( e )}" )


@app.post( "/subscribe" )
def subscribe ( phone_number: str = Form( ... ) ) :
    try :
        if subscriptions_collection.find_one( {"phone_number" : phone_number} ) :
            raise HTTPException( status_code=400, detail="Already subscribed." )
        subscriptions_collection.insert_one( {"phone_number" : phone_number} )
        return {"message" : "Successfully subscribed!"}
    except Exception as e :
        logging.error( f"❌ Error in /subscribe: {e}" )
        raise HTTPException( status_code=500, detail=f"Error subscribing: {str( e )}" )


@app.post( "/feedback" )
def submit_feedback ( news_text: str = Form( ... ), feedback: str = Form( ... ) ) :
    try :
        feedback_collection.insert_one( {"news_text" : news_text, "feedback" : feedback} )
        return {"message" : "Feedback submitted!"}
    except Exception as e :
        logging.error( f"❌ Error in /feedback: {e}" )
        raise HTTPException( status_code=500, detail=f"Error submitting feedback: {str( e )}" )


if __name__ == "__main__" :
    uvicorn.run( app, host="0.0.0.0", port=8000, log_level="info" )
