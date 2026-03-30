import torch 
from torch.utils.data import Dataset 
from transformers import AutoTokenizer 
from typing import List, Dict 
from .preprocessor import TextPreprocessor 
 
class GKHRDDataset(Dataset): 
    def __init__(self, records: List[Dict], tokenizer_name: str, max_length: int = 512): 
        self.records = records 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) 
        self.max_length = max_length 
        self.preprocessor = TextPreprocessor() 
 
    def __len__(self): 
        return len(self.records) 
 
    def __getitem__(self, idx): 
        rec = self.records[idx] 
        claim = self.preprocessor.clean_text(rec.get("claim", "")) 
        label = self.preprocessor.normalize_label(rec.get("label", "unproven")) 
        encoding = self.tokenizer( 
            claim, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt" 
        ) 
        return { 
            "input_ids": encoding["input_ids"].squeeze(), 
            "attention_mask": encoding["attention_mask"].squeeze(), 
            "label": torch.tensor(label, dtype=torch.long), 
            "claim_text": claim, 
            "source": rec.get("source", "unknown"), 
            "evidence": rec.get("evidence", "") 
        } 
