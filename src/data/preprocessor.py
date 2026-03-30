import re 
import unicodedata 
 
LABEL_MAP = { 
    "true": 0, "false": 1, "mixture": 2, "unproven": 3, 
    "supported": 0, "refuted": 1, "nei": 3, 
    "SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 3, 
    "not_misinforming": 0, "misinforming": 1 
} 
 
LABEL_NAMES = ["TRUE", "FALSE", "MIXTURE", "UNPROVEN"] 
 
class TextPreprocessor: 
    @staticmethod 
    def clean_text(text): 
        if not isinstance(text, str): 
            return "" 
        text = unicodedata.normalize("NFKC", text) 
        text = re.sub(r"http\S+", "", text) 
        text = re.sub(r"@\w+", "", text) 
        text = re.sub(r"#(\w+)", r"\1", text) 
        text = re.sub(r"\s+", " ", text).strip() 
        return text 
 
    @staticmethod 
    def normalize_label(label, source=""): 
        label_str = str(label).strip().lower() 
        return LABEL_MAP.get(label_str, 3) 
