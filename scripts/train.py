#!/usr/bin/env python3 
"""Simplified training script with real text features.""" 
import sys 
import json 
import logging 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from pathlib import Path 
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel 
 
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from src.config import load_config 
from src.data.dataset import GKHRDDataset 
from src.models.gk_hrd import GKHRD 
 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 
 
class SimpleTextModel(nn.Module): 
    def __init__(self, num_classes=4): 
        super().__init__() 
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        self.encoder = AutoModel.from_pretrained("bert-base-uncased") 
        self.classifier = nn.Linear(768, num_classes) 
 
    def forward(self, texts): 
        encoding = self.tokenizer(texts, padding=True, truncation=True, 
                                max_length=128, return_tensors="pt") 
        outputs = self.encoder(**encoding) 
        pooled = outputs.last_hidden_state[:, 0, :] 
        return self.classifier(pooled) 
 
def main(): 
    logger.info("="*60) 
    logger.info("GK-HRD Training with Real Text Features") 
    logger.info("="*60) 
 
    cfg = load_config() 
    device = torch.device("cpu") 
    logger.info(f"Using device: {device}") 
 
    with open("data/processed/gkhrd_unified_dataset.json", "r") as f: 
        data = json.load(f) 
    logger.info(f"Loaded {len(data)} records") 
 
    train_data = data[:2] 
    val_data = data[2:] 
 
    train_texts = [d["claim"] for d in train_data] 
    train_labels = torch.tensor([d["label"] for d in train_data]) 
    val_texts = [d["claim"] for d in val_data] 
    val_labels = torch.tensor([d["label"] for d in val_data]) 
 
    model = SimpleTextModel(num_classes=4) 
    model.to(device) 
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
    criterion = nn.CrossEntropyLoss() 
 
    for epoch in range(20): 
        model.train() 
        optimizer.zero_grad() 
        logits = model(train_texts) 
        loss = criterion(logits, train_labels) 
        loss.backward() 
        optimizer.step() 
 
        model.eval() 
        with torch.no_grad(): 
            val_logits = model(val_texts) 
            val_loss = criterion(val_logits, val_labels) 
            preds = torch.argmax(val_logits, dim=-1) 
            acc = (preds == val_labels).float().mean() 
 
        if (epoch + 1) % 5 == 0: 
            logger.info(f"Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc.item():.4f}") 
 
    torch.save(model.state_dict(), "best_model.pt") 
    logger.info("\n? Training complete! Model saved to best_model.pt") 
 
if __name__ == "__main__": 
    main() 
