import json 
import logging 
from pathlib import Path 
from typing import List, Dict 
 
logger = logging.getLogger(__name__) 
 
class ScifactLoader: 
    def __init__(self, data_dir="data/raw"): 
        self.data_dir = Path(data_dir) / "scifact" 
        self.corpus = self._load_corpus() 
 
    def _load_corpus(self): 
        corpus = {} 
        path = self.data_dir / "corpus.json" 
        if not path.exists(): 
            return corpus 
        with open(path, encoding="utf-8") as f: 
            for line in f: 
                doc = json.loads(line) 
                doc_id = str(doc.get("doc_id", "")) 
                abstract = doc.get("abstract", []) 
                if isinstance(abstract, list): 
                    abstract_text = " ".join(abstract) 
                else: 
                    abstract_text = str(abstract) 
                corpus[doc_id] = {"title": doc.get("title", ""), "abstract": abstract_text} 
        logger.info(f"Loaded SciFact corpus: {len(corpus)} abstracts") 
        return corpus 
 
    def load(self, split="train"): 
        records = [] 
        path = self.data_dir / f"claims_{split}.json" 
        if not path.exists(): 
            logger.warning(f"SciFact {split} file not found: {path}") 
            return records 
        with open(path, encoding="utf-8") as f: 
            for line in f: 
                item = json.loads(line) 
                records.append({"claim": item.get("claim", ""), "label": item.get("label", "NOT_ENOUGH_INFO"), "evidence": "", "source": "scifact", "language": "en"}) 
        logger.info(f"Loaded {len(records)} SciFact records ({split})") 
        return records 
