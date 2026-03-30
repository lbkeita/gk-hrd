import json 
import logging 
from pathlib import Path 
from typing import List, Dict 
 
logger = logging.getLogger(__name__) 
 
class HealthFCLoader: 
    def __init__(self, data_dir="data/raw"): 
        self.data_dir = Path(data_dir) / "healthfc" 
 
    def load(self, split="train"): 
        records = [] 
        path = self.data_dir / f"{split}.json" 
        if not path.exists(): 
            logger.warning(f"HealthFC {split} file not found: {path}") 
            return records 
        with open(path, encoding="utf-8") as f: 
            for line in f: 
                item = json.loads(line) 
                records.append({"claim": item.get("claim", ""), "label": item.get("label", "unproven"), "evidence": item.get("explanation", ""), "source": "healthfc", "language": item.get("language", "en")}) 
        logger.info(f"Loaded {len(records)} HealthFC records ({split})") 
        return records 
