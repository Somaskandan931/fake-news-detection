import torch
from transformers import BertTokenizer
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
import os

MODEL_PATH = "Backend/models/bert_lstm_model.pth"

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["fake_news_db"]
collection = db["training_data"]

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = tokenizer(item["text"], padding="max_length", truncation=True, return_tensors="pt")
        return tokens["input_ids"].squeeze(0), torch.tensor(item["label"])

def retrain_model():
    print("Fetching new training data from MongoDB...")
    new_data = list(collection.find({}, {"_id": 0, "text": 1, "label": 1}))

    if len(new_data) < 10:  # Avoid retraining with too little data
        print("Not enough data to retrain. Skipping...")
        return

    dataset = FakeNewsDataset(new_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load model
    model = torch.load(MODEL_PATH)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("Retraining model with new data...")
    for epoch in range(3):  # Train for 3 epochs
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete. Loss: {loss.item()}")

    # Save updated model
    torch.save(model, MODEL_PATH)
    print("Retrained model saved.")

if __name__ == "__main__":
    retrain_model()
