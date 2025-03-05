import torch
import torch.nn as nn
from transformers import BertModel

MODEL_PATH = r"C:\Users\somas\Desktop\bert_lstm_model.pth"
QUANTIZED_MODEL_PATH = r"C:\Users\somas\Desktop\bert_lstm_model_quantized.pth"

# ✅ Define the model architecture (must match your training script)
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

# ✅ Load model properly
model = BertLSTMModel()  # Instantiate model
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # Load weights
model.eval()  # Put model in evaluation mode

# ✅ Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)

# ✅ Save the quantized model
torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

print("✅ Quantized model saved successfully! Size reduced.")
