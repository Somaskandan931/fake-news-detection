import torch
import torch.nn as nn
import shap
import dill as pickle  # ✅ Use dill instead of pickle
from transformers import BertModel, BertTokenizer

# ✅ Define Model
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

# ✅ Load Model
model = BertLSTMModel()
model.load_state_dict(torch.load(r"C:\Users\somas\Desktop\bert_lstm_model.pth", map_location="cpu"), strict=False)
model.eval()

# ✅ Define SHAP Explainer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(model, masker=masker)

# ✅ Save SHAP Explainer using `dill`
with open(r"C:\Users\somas\Desktop\shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

print("✅ SHAP explainer re-saved successfully!")
