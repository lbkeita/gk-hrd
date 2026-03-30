import torch.nn as nn 
from .rag_branch import RAGBranch 
from .gnn_branch import GNNBranch 
from .fusion import AttentionGateFusion 
 
class GKHRD(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        self.rag = RAGBranch(config) 
        self.gnn = GNNBranch(config) 
        self.fusion = AttentionGateFusion(config) 
 
    def forward(self, claims, graph_data): 
        rag_emb = self.rag(claims) 
        batch_size = rag_emb.size(0) 
        gnn_emb = self.gnn(graph_data, batch_size=batch_size) 
        logits, gate_weights = self.fusion(rag_emb, gnn_emb) 
        return logits, gate_weights 
