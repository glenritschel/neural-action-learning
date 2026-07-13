import pytest
import torch
from environment.discrete_world import DiscreteWorld
from gnn.graph_builder import GraphBuilder
from gnn.gnn_model import GNNEdgeCostPredictor

def test_graph_builder():
    world = DiscreteWorld(grid_size_x=5, grid_size_y=5, time_steps=3)
    builder = GraphBuilder(world)

    graph = builder.build_grid_graph()
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    data = builder.to_torch_geometric(graph)
    assert data is not None
    assert data.x.shape[1] == 3 # [nx, ny, nt]
    assert data.edge_attr.shape[1] == 2 # [dx, dy]

def test_gnn_model():
    world = DiscreteWorld(grid_size_x=5, grid_size_y=5, time_steps=3)
    builder = GraphBuilder(world)
    graph = builder.build_grid_graph()
    data = builder.to_torch_geometric(graph)

    model = GNNEdgeCostPredictor(node_in_dim=3, edge_in_dim=2, hidden_dim=16)

    out = model(data.x, data.edge_index, data.edge_attr)

    assert out is not None
    assert out.shape[0] == data.edge_attr.shape[0]
