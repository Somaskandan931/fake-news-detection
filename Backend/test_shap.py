import pytest
import dill as pickle  # ✅ Use dill instead of pickle for better compatibility
import os
import shap
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

# Define file paths
MODEL_PATH = r"C:\Users\somas\Desktop\bert_lstm_model.pth"
EXPLAINER_PATH = r"C:\Users\somas\Desktop\shap_explainer.pkl"
SHAP_VALUES_PATH = r"C:\Users\somas\Desktop\shap_values.pkl"

# ✅ Define Model Architecture (Must Match Your Trained Model)
class BertLSTMModel(nn.Module):
    def __init__(self):
        super(BertLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        return self.fc(lstm_output[:, -1, :])

@pytest.fixture
def trained_model():
    """Fixture to load the trained model before testing."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"❌ Trained model file '{MODEL_PATH}' not found!")

    try:
        model = BertLSTMModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")), strict=False)
        model.eval()
        return model
    except Exception as e:
        pytest.fail(f"❌ Error loading model: {e}")

@pytest.fixture
def shap_explainer():
    """Fixture to load SHAP explainer."""
    if not os.path.exists(EXPLAINER_PATH):
        pytest.fail(f"❌ SHAP explainer file '{EXPLAINER_PATH}' not found!")

    try:
        # ✅ Fix: Ensure `BertLSTMModel` is defined before loading
        with open(EXPLAINER_PATH, "rb") as f:
            explainer = pickle.load(f)  # ✅ Using `dill` to handle custom classes
        return explainer
    except AttributeError as e:
        pytest.fail(f"❌ SHAP explainer failed to load due to missing class definition: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error loading SHAP explainer: {e}")

@pytest.fixture
def shap_values():
    """Fixture to load SHAP values."""
    if not os.path.exists(SHAP_VALUES_PATH):
        pytest.fail(f"❌ SHAP values file '{SHAP_VALUES_PATH}' not found!")

    try:
        with open(SHAP_VALUES_PATH, "rb") as f:
            shap_vals = pickle.load(f)

        # ✅ Convert NumPy array to list if needed
        if isinstance(shap_vals, np.ndarray):
            shap_vals = shap_vals.tolist()

        return shap_vals
    except Exception as e:
        pytest.fail(f"❌ Error loading SHAP values: {e}")

def test_trained_model(trained_model):
    """Test if the trained model loads correctly."""
    assert trained_model is not None, "❌ Model failed to load!"
    print("✅ Model loaded successfully.")

def test_shap_explainer(shap_explainer):
    """Test if the SHAP explainer loads correctly."""
    assert shap_explainer is not None, "❌ SHAP explainer failed to load!"
    print("✅ SHAP explainer loaded successfully.")

def test_shap_values(shap_values):
    """Test if SHAP values load correctly and have valid structure."""
    assert isinstance(shap_values, list), "❌ SHAP values should be a list!"
    assert len(shap_values) > 0, "❌ SHAP values list is empty!"
    assert isinstance(shap_values[0], (np.ndarray, list)), "❌ First SHAP value should be a NumPy array or list!"
    print("✅ SHAP values loaded and validated successfully.")
