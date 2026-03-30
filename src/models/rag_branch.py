import torch.nn as nn 
import torch 
 
class RAGBranch(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        if hasattr(config, 'rag'): 
            self.hidden_dim = config.rag.hidden_dim 
        else: 
            self.hidden_dim = 768 
        if hasattr(config, 'fusion'): 
            self.projection = nn.Linear(self.hidden_dim, config.fusion.hidden_dim) 
        else: 
            self.projection = nn.Linear(self.hidden_dim, 512) 
 
    def forward(self, claims): 
        batch_size = len(claims) if isinstance(claims, list) else claims.size(0) if hasattr(claims, 'size') else 1 
        emb = torch.randn(batch_size, self.hidden_dim) 
        return self.projection(emb) 
