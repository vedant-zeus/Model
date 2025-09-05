import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from torch.utils.data import Dataset
import torch

# ----------------------------
# Load your jargon list
# ----------------------------
from LEGAL_JARGONS import LEGAL_JARGONS

# ----------------------------
# Step 1: Prepare Dataset
# ----------------------------
class LegalJargonDataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_len=256):
        self.texts, self.labels = [], []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    
                    # Label: 1 if contains any legal jargon, else 0
                    label = 1 if any(j.lower() in text.lower() for j in LEGAL_JARGONS) else 0
                    
                    self.texts.append(text)
                    self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ----------------------------
# Step 2: Initialize tokenizer and dataset
# ----------------------------
DATASET_DIR = "data"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = LegalJargonDataset(DATASET_DIR, tokenizer)

# ----------------------------
# Step 3: Load Model
# ----------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ----------------------------
# Step 4: Training
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# ----------------------------
# Step 5: Save model
# ----------------------------
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("✅ Training completed! Model saved in ./saved_model/")
