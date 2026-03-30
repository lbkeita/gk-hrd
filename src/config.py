# Configuration loader for GK-HRD 
import os 
import torch 
import random 
import numpy as np 
from pathlib import Path 
from omegaconf import OmegaConf 
from dotenv import load_dotenv 
 
load_dotenv 
 
def get_device(preference="auto"): 
    if preference != "auto": 
        return torch.device(preference) 
    if torch.cuda.is_available(): 
        return torch.device("cuda") 
    return torch.device("cpu") 
 
def set_seed(seed): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
 
def load_config(config_path="config/config.yaml"): 
    cfg = OmegaConf.load(config_path) 
    cfg.project.device = str(get_device(cfg.project.device)) 
    set_seed(cfg.project.seed) 
    for d in [cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.graphs_dir, cfg.data.kb_dir]: 
        Path(d).mkdir(parents=True, exist_ok=True) 
    return cfg 
 
def get_twitter_token(): 
    token = os.getenv("TWITTER_BEARER_TOKEN") 
    if not token or token == "your_token_here": 
        raise ValueError("TWITTER_BEARER_TOKEN not set in .env file") 
    return token 
