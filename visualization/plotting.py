import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import umap

def plot_trajectory(trajectory, grid_size=(20, 20), title="Trajectory", figsize=(8, 8)):
    """
    Plots a trajectory on a 2D grid.

    Args:
        trajectory: List of (x, y) tuples representing the path.
        grid_size: Tuple representing (width, height) of the grid.
        title: Title of the plot.
        figsize: Figure size.
    """
    if not trajectory:
        return

    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    plt.figure(figsize=figsize)
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', markersize=4)
    plt.plot(x_coords[0], y_coords[0], marker='s', color='g', markersize=8, label="Start")
    plt.plot(x_coords[-1], y_coords[-1], marker='*', color='r', markersize=10, label="End")

    plt.xlim(-0.5, grid_size[0] - 0.5)
    plt.ylim(-0.5, grid_size[1] - 0.5)
    plt.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, grid_size[0], 1))
    plt.yticks(np.arange(0, grid_size[1], 1))
    plt.title(title)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

def plot_heatmap(visited_cells, grid_size=(20, 20), title="Visited Cells Heatmap", figsize=(8, 8)):
    """
    Plots a heatmap of visited cells.

    Args:
        visited_cells: List of (x, y) tuples or a list of counts.
        grid_size: Tuple representing (width, height) of the grid.
        title: Title of the plot.
        figsize: Figure size.
    """
    heatmap = np.zeros((grid_size[1], grid_size[0]))
    for x, y in visited_cells:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            heatmap[y, x] += 1

    plt.figure(figsize=figsize)
    plt.imshow(heatmap, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_action_histogram(actions, title="Action Histogram"):
    """
    Plots a histogram of taken actions using Plotly.

    Args:
        actions: List of action labels or indices.
        title: Title of the plot.
    """
    fig = px.histogram(x=actions, title=title, labels={'x': 'Action', 'count': 'Frequency'})
    fig.update_layout(bargap=0.2)
    fig.show()

def plot_latent_space(latent_vectors, labels=None, method='pca', title="Latent Space Projection"):
    """
    Plots a 2D projection of the latent space using PCA or UMAP.

    Args:
        latent_vectors: Numpy array of shape (n_samples, n_features).
        labels: Optional labels for coloring the points.
        method: 'pca' or 'umap'.
        title: Title of the plot.
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("method must be 'pca' or 'umap'")

    projections = reducer.fit_transform(latent_vectors)

    fig = px.scatter(
        x=projections[:, 0],
        y=projections[:, 1],
        color=labels,
        title=f"{title} ({method.upper()})",
        labels={'x': 'Component 1', 'y': 'Component 2'}
    )
    fig.show()
