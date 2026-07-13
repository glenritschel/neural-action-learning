import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Tuple, List, Dict
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator

class GraphBuilder:
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator):
        self.world = world
        self.calculator = calculator

    def build_networkx_graph(self) -> nx.DiGraph:
        """
        Builds a NetworkX directed graph where nodes are valid states (x, y, t)
        and edges represent valid actions with their associated costs.
        """
        G = nx.DiGraph()

        # Add nodes
        for t in range(self.world.time_steps):
            for x in range(self.world.grid_size_x):
                for y in range(self.world.grid_size_y):
                    if self.world.is_valid_state(x, y, t):
                        # Node features: normalized x, y, t
                        nx_val = x / float(self.world.grid_size_x)
                        ny_val = y / float(self.world.grid_size_y)
                        nt_val = t / float(self.world.time_steps)
                        G.add_node((x, y, t), x=nx_val, y=ny_val, t=nt_val)

        # Add edges
        for node in G.nodes():
            x, y, t = node
            next_t = t + 1
            if next_t >= self.world.time_steps:
                continue

            for action in Action:
                dx, dy = action.value
                nx_state, ny_state = x + dx, y + dy
                next_state = (nx_state, ny_state, next_t)

                if next_state in G.nodes():
                    # Calculate cost. Note: accurate cost requires previous state context for acceleration,
                    # but for GNN edge prediction we might approximate it as the step cost from static.
                    # Or we calculate it for a simple trajectory [node, next_state]
                    path = [node, next_state]
                    cost = self.calculator.calculate_cost(path)

                    G.add_edge(node, next_state, cost=cost, action_dx=dx, action_dy=dy)

        return G

    def to_torch_geometric(self, G: nx.DiGraph) -> Data:
        """
        Converts the NetworkX graph to a PyTorch Geometric Data object.
        """
        # Create a mapping from node tuple to integer index
        node_mapping = {node: i for i, node in enumerate(G.nodes())}

        # Prepare node features
        x = []
        for node in G.nodes():
            attrs = G.nodes[node]
            x.append([attrs['x'], attrs['y'], attrs['t']])
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Prepare edge indices and features (costs)
        edge_indices = []
        edge_attrs = []
        edge_costs = [] # target values

        for u, v, data in G.edges(data=True):
            edge_indices.append([node_mapping[u], node_mapping[v]])
            edge_attrs.append([data['action_dx'], data['action_dy']])
            edge_costs.append([data['cost']])

        if not edge_indices:
            # Handle empty graph case
            return Data(x=x_tensor, edge_index=torch.empty((2, 0), dtype=torch.long))

        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)
        edge_costs_tensor = torch.tensor(edge_costs, dtype=torch.float32)

        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, edge_costs=edge_costs_tensor)
        return data
