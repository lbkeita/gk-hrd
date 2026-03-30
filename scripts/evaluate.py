#!/usr/bin/env python3 
"""Complete evaluation script for GK-HRD model.""" 
import sys 
import json 
import torch 
import logging 
from pathlib import Path 
from sklearn.metrics import accuracy_score, classification_report 
 
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from src.config import load_config 
from src.models.gk_hrd import GKHRD 
from src.data.preprocessor import TextPreprocessor, LABEL_NAMES 
 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 
 
def evaluate(): 
    logger.info("="*60) 
    logger.info("GK-HRD Evaluation") 
    logger.info("="*60) 
 
    cfg = load_config() 
    device = torch.device(cfg.project.device) 
    logger.info(f"Device: {device}") 
 
    with open("data/processed/gkhrd_unified_dataset.json", "r") as f: 
        data = json.load(f) 
    logger.info(f"Loaded {len(data)} test records") 
 
    model = GKHRD(cfg) 
    if Path("best_model.pt").exists(): 
        model.load_state_dict(torch.load("best_model.pt", map_location=device)) 
        logger.info("Loaded trained model") 
    model.to(device) 
    model.eval() 
 
    preprocessor = TextPreprocessor() 
    predictions = [] 
    true_labels = [] 
 
    logger.info("Running inference on test set...") 
    for rec in data: 
        claim = preprocessor.clean_text(rec["claim"]) 
        true_label = preprocessor.normalize_label(rec["label"]) 
        graph_data = {"claim": {"x": torch.randn(1, 768)}} 
        with torch.no_grad(): 
            logits, _ = model([claim], graph_data) 
            pred = torch.argmax(logits, dim=-1).item() 
        predictions.append(pred) 
        true_labels.append(true_label) 
 
    accuracy = accuracy_score(true_labels, predictions) 
    logger.info(f"\nAccuracy: {accuracy:.4f}") 
    print("\n" + "="*60) 
    print("Classification Report") 
    print("="*60) 
    print(classification_report(true_labels, predictions, target_names=LABEL_NAMES)) 
 
if __name__ == "__main__": 
    evaluate() 
