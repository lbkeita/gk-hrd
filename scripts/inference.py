#!/usr/bin/env python3 
"""Complete inference script for GK-HRD model.""" 
import sys 
import torch 
import logging 
from pathlib import Path 
 
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from src.config import load_config 
from src.models.gk_hrd import GKHRD 
from src.data.preprocessor import TextPreprocessor, LABEL_NAMES 
 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 
 
def load_model(model_path="best_model.pt"): 
    logger.info("Loading model...") 
    cfg = load_config() 
    device = torch.device(cfg.project.device) 
    model = GKHRD(cfg) 
    if Path(model_path).exists(): 
        model.load_state_dict(torch.load(model_path, map_location=device)) 
        logger.info(f"Model loaded from {model_path}") 
    else: 
        logger.warning(f"Model file {model_path} not found, using random weights") 
    model.to(device) 
    model.eval() 
    return model, device 
 
def predict(model, device, claim_text): 
    preprocessor = TextPreprocessor() 
    clean_claim = preprocessor.clean_text(claim_text) 
 
    with torch.no_grad(): 
        graph_data = {"claim": {"x": torch.randn(1, 768)}} 
        logits, gate_weights = model([clean_claim], graph_data) 
        probs = torch.softmax(logits, dim=-1) 
        pred_class = torch.argmax(probs, dim=-1).item() 
 
    return { 
        "claim": claim_text, 
        "prediction": LABEL_NAMES[pred_class], 
        "confidence": probs[0][pred_class].item(), 
        "probabilities": {LABEL_NAMES[i]: probs[0][i].item() for i in range(len(LABEL_NAMES))}, 
        "gate_weights": {"rag": gate_weights[0][0].item(), "gnn": gate_weights[0][1].item()} 
    } 
 
def main(): 
    import argparse 
    parser = argparse.ArgumentParser(description="GK-HRD Inference") 
    parser.add_argument("--claim", type=str, required=True, help="Claim to evaluate") 
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to model checkpoint") 
    args = parser.parse_args() 
 
    logger.info("="*60) 
    logger.info("GK-HRD Inference") 
    logger.info("="*60) 
    logger.info(f"Claim: {args.claim}") 
 
    model, device = load_model(args.model) 
    result = predict(model, device, args.claim) 
 
    print("\n" + "="*60) 
    print("Prediction Results") 
    print("="*60) 
    print(f"Claim: {result['claim']}") 
    print(f"Prediction: {result['prediction']}") 
    print(f"Confidence: {result['confidence']:.4f}") 
    print(f"\nClass Probabilities:") 
    for label, prob in result['probabilities'].items(): 
        print(f"  {label}: {prob:.4f}") 
    print(f"\nGate Weights (RAG vs GNN):") 
    print(f"  RAG: {result['gate_weights']['rag']:.3f}") 
    print(f"  GNN: {result['gate_weights']['gnn']:.3f}") 
    print("="*60) 
 
if __name__ == "__main__": 
    main() 
