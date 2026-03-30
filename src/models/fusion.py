import torch.nn as nn 
import torch 
 
class AttentionGateFusion(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        if hasattr(config, 'fusion'): 
            self.hidden_dim = config.fusion.hidden_dim 
            self.num_classes = config.fusion.num_classes 
        else: 
            self.hidden_dim = 512 
            self.num_classes = 4 
        self.gate = nn.Linear(self.hidden_dim * 2, 2) 
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes) 
 
    def forward(self, rag_emb, gnn_emb): 
        if rag_emb.dim() == 1: 
            rag_emb = rag_emb.unsqueeze(0) 
        if gnn_emb.dim() == 1: 
            gnn_emb = gnn_emb.unsqueeze(0) 
        if rag_emb.size(0) != gnn_emb.size(0): 
            min_size = min(rag_emb.size(0), gnn_emb.size(0)) 
            rag_emb = rag_emb[:min_size] 
            gnn_emb = gnn_emb[:min_size] 
        combined = torch.cat([rag_emb, gnn_emb], dim=-1) 
        gate_weights = torch.softmax(self.gate(combined), dim=-1) 
        fused = gate_weights[:, 0:1] * rag_emb + gate_weights[:, 1:2] * gnn_emb 
        logits = self.classifier(fused) 
        return logits, gate_weights 
