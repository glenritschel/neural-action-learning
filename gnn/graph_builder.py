import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import Tuple, List, Dict

from environment.discrete_world import DiscreteWorld, Action

class GraphBuilder:
    def __init__(self, world: DiscreteWorld):
        self.world = world

    def build_grid_graph(self) -> nx.DiGraph:
        """
        Builds a directed graph representing the valid states and transitions in DiscreteWorld.
        Nodes represent (x, y, t) states.
        Edges represent valid actions between states.
        """
        G = nx.DiGraph()

        # Add all valid nodes and their features
        for t in range(self.world.time_steps):
            for x in range(self.world.grid_size_x):
                for y in range(self.world.grid_size_y):
                    if self.world.is_valid_state(x, y, t):
                        # Node features: normalized coordinates
                        nx_val = x / self.world.grid_size_x
                        ny_val = y / self.world.grid_size_y
                        nt_val = t / self.world.time_steps

                        node_id = (x, y, t)
                        G.add_node(node_id, x=[nx_val, ny_val, nt_val])

        # Add all valid edges (transitions)
        for t in range(self.world.time_steps - 1):
            next_t = t + 1
            for x in range(self.world.grid_size_x):
                for y in range(self.world.grid_size_y):
                    current_node = (x, y, t)
                    if G.has_node(current_node):
                        for action in Action:
                            dx, dy = action.value
                            nx_c, ny_c = x + dx, y + dy
                            next_node = (nx_c, ny_c, next_t)

                            if G.has_node(next_node):
                                # Edge features: normalized action vector (dx, dy)
                                edge_feat = [float(dx), float(dy)]
                                G.add_edge(current_node, next_node, edge_attr=edge_feat)

        return G

    def to_torch_geometric(self, graph: nx.DiGraph) -> Data:
        """
        Converts the networkx graph to a PyTorch Geometric Data object.
        """
        # from_networkx handles converting 'x' and 'edge_attr' attributes if they exist
        data = from_networkx(graph)

        # from_networkx converts node labels to a generic mapping.
        # Node indices will be integers 0 to N-1.

        # Ensure tensors are float32
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.to(torch.float32)

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(torch.float32)

        return data
