import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EdgeCostGNN(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, edge_attr_dim: int):
        super(EdgeCostGNN, self).__init__()
        # Node embeddings
        self.conv1 = SAGEConv(node_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Edge cost predictor
        # Inputs: source node embedding, target node embedding, edge attributes
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Generate node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Extract embeddings for source and target nodes of each edge
        src, dst = edge_index
        src_embeds = x[src]
        dst_embeds = x[dst]

        # Concatenate src, dst, and edge attributes
        edge_inputs = torch.cat([src_embeds, dst_embeds, edge_attr], dim=1)

        # Predict costs
        predicted_costs = self.edge_predictor(edge_inputs)
        return predicted_costs
