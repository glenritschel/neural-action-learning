import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNEdgeCostPredictor(nn.Module):
    def __init__(self, node_in_dim: int = 3, edge_in_dim: int = 2, hidden_dim: int = 64):
        """
        GNN to predict individual edge costs.
        Node features: [nx, ny, nt]
        Edge features: [dx, dy]
        """
        super(GNNEdgeCostPredictor, self).__init__()

        # Node feature embedding
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)

        # Message passing layers to update node representations
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Edge prediction MLP
        # Input will be concatenation of source node, target node, and edge attribute
        edge_mlp_in = hidden_dim * 2 + edge_in_dim

        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Predict scalar cost
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix [num_nodes, node_in_dim]
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge feature matrix [num_edges, edge_in_dim]
        """
        # Node embedding
        x = self.node_emb(x)
        x = torch.relu(x)

        # Message passing
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Gather source and target node representations for each edge
        src, dst = edge_index[0], edge_index[1]
        node_features_src = x[src]
        node_features_dst = x[dst]

        # Concatenate src, dst, and edge features
        edge_inputs = torch.cat([node_features_src, node_features_dst, edge_attr], dim=-1)

        # Predict edge costs
        predicted_costs = self.edge_predictor(edge_inputs)

        return predicted_costs.squeeze(-1) # [num_edges]
