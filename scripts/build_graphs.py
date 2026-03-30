#!/usr/bin/env python3 
"""Build heterogeneous graphs for GNN branch.""" 
import sys 
import json 
import logging 
import torch 
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from src.config import load_config 
 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 
 
def build_simple_graph(): 
    logger.info("Building simple graph for demonstration...") 
    graph_data = { 
        "claim": {"x": torch.randn(3, 768), "y": torch.tensor([1, 0, 0])}, 
        "entity": {"x": torch.randn(5, 768)}, 
        "source": {"x": torch.randn(2, 768)} 
    } 
    return graph_data 
 
def main(): 
    logger.info("="*60) 
    logger.info("Building Graphs for GK-HRD") 
    logger.info("="*60) 
    cfg = load_config() 
    graphs_dir = Path(cfg.data.graphs_dir) 
    graphs_dir.mkdir(parents=True, exist_ok=True) 
    logger.info(f"Graphs directory: {graphs_dir}") 
 
    graph_data = build_simple_graph() 
    torch.save(graph_data, graphs_dir / "hetero_graph.pt") 
    logger.info(f"Saved graph to {graphs_dir / 'hetero_graph.pt'}") 
 
    import json 
    metadata = { 
        "num_claim_nodes": graph_data["claim"]["x"].size(0), 
        "num_entity_nodes": graph_data["entity"]["x"].size(0), 
        "num_source_nodes": graph_data["source"]["x"].size(0), 
        "node_types": list(graph_data.keys()) 
    } 
    with open(graphs_dir / "graph_metadata.json", "w") as f: 
        json.dump(metadata, f, indent=2) 
    logger.info("Graph building complete!") 
 
if __name__ == "__main__": 
    main() 
