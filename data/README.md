"# Data Directory" 
 
This directory contains datasets. They are not included in the repository due to size. 
 
## Download Instructions 
 
### 1. HealthFC 
Download from: https://github.com/jvladika/HealthFC 
Run: `python scripts/prepare_data.py --healthfc` 
 
### 2. SciFact 
Load with HuggingFace: 
```python 
from datasets import load_dataset 
dataset = load_dataset(\"allenai/scifact\") 
``` 
 
### 3. MuMiN 
Install: `pip install mumin[all]` 
Run: `python scripts/prepare_mumin.py --size small` 
 
### Processed Data 
After running the preparation scripts, processed data will appear here. 
