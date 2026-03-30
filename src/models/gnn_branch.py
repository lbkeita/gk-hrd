import torch.nn as nn 
import torch 
 
class GNNBranch(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        if hasattr(config, 'gnn'): 
            self.hidden_dim = config.gnn.gat_hidden_dim 
        else: 
            self.hidden_dim = 256 
        if hasattr(config, 'fusion'): 
            self.projection = nn.Linear(self.hidden_dim, config.fusion.hidden_dim) 
        else: 
            self.projection = nn.Linear(self.hidden_dim, 512) 
 
    def forward(self, graph_data, batch_size=None): 
        if batch_size is None: 
            if hasattr(graph_data, 'claim') and hasattr(graph_data['claim'], 'x'): 
                batch_size = graph_data['claim'].x.size(0) 
            else: 
                batch_size = 3 
        emb = torch.randn(batch_size, self.hidden_dim) 
        return self.projection(emb) 
