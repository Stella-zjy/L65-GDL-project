import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch_geometric.utils as pyg_utils


# Data loader for train test split
def prepare_data(dataset, train_split, batch_size):
    dataset = dataset.shuffle()

    # Train test split
    train_idx = int(len(dataset) * train_split)
    train_set = dataset[:train_idx]
    test_set = dataset[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    visual_data_loader = DataLoader(test_set, batch_size=200, shuffle=True)

    train_zeros = 0
    train_ones = 0
    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0
    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    print()
    print(f"Class split - Training 0: {train_zeros} 1: {train_ones}, Test 0: {test_zeros} 1: {test_ones}")

    return train_loader, test_loader, visual_data_loader



# To make node feature vector in the size of [B, N, F], to be compatible with input tensor size of DenseGCNConv
# B: batch size
# N: number of nodes
# F: feature dimension
def pad_features(batch_data):
    # Check if the input is already a batched data object
    if isinstance(batch_data, Batch):
        data = batch_data
    else:
        # Convert the list of data objects to a batched data object
        data = Batch.from_data_list(batch_data)

    # Find the maximum number of nodes in any graph in the batch
    max_nodes = 0
    for i in range(data.num_graphs):
        max_nodes = max(max_nodes, (data.batch == i).sum().item())

    # Pad each graph's node feature matrix
    padded_features = []
    for i in range(data.num_graphs):
        # Extract the features of the i-th graph in the batch
        x = data.x[data.batch == i]
        num_nodes = x.size(0)
        num_features = x.size(1)
        padding = torch.zeros(max_nodes - num_nodes, num_features, device=x.device)
        padded_x = torch.cat([x, padding], dim=0)
        padded_features.append(padded_x)

    # Stack all padded matrices to create a batched tensor
    batched_x = torch.stack(padded_features, dim=0)
    return batched_x

def visualize_graphs_with_pooled_nodes_concept(data, cluster_top_features, node_info, labels, real_graph, clustering_type, reduction_type, layer_num, k, num_of_diffpool):
    """
    Visualizes the graphs with specified nodes highlighted and labeled by concepts.
    """
    num_rows = len(cluster_top_features)
    num_cols = max(len(v) for v in cluster_top_features.values())

    # Pre-compute the starting index for each graph
    starting_indices = [0]  # The first graph starts at index 0
    for num_nodes in real_graph[:-1]:  # Exclude the last graph as its starting index is not needed
        starting_indices.append(starting_indices[-1] + num_nodes)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 3 * num_rows + 2), squeeze=False)
    fig.suptitle(f'Nearest Instances to {clustering_type}, with k = {k}, number of diffpool clusters {num_of_diffpool}, Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    axes = axes.flatten()

    for cluster_idx, (cluster, feature_indexes) in enumerate(cluster_top_features.items()):
        for feature_idx, feature_index in enumerate(feature_indexes):
            batch_index, node_indexes = node_info[feature_index]

            G = to_networkx(data[batch_index], to_undirected=True)

            # Use the starting index for this graph to map labels
            start_index = starting_indices[batch_index]
            node_labels = {i: labels[start_index + i] for i in G.nodes()}

            node_colors = ["gray" if i not in node_indexes else "red" for i in G.nodes()]

            ax = axes[cluster_idx * num_cols + feature_idx]
            nx.draw(G, ax=ax, node_color=node_colors, labels=node_labels, with_labels=True, node_size=50)
            ax.set_title(f"Cluster {cluster}, Feature {feature_index}")

    plt.tight_layout()
    plt.show()

def find_top_closest(features_np, labels_km, centroids, top_n=5):
    # Dictionary to hold the indices of the top closest points for each cluster
    top_closest_indices = {i: [] for i in range(len(centroids))}

    # Calculate the distances and find top closest for each cluster
    for i, centroid in enumerate(centroids):
        # Find indices of points belonging to the current cluster
        indices = np.where(labels_km == i)[0]
        cluster_points = features_np[indices]

        # Calculate distances from the centroid to each point in the cluster
        distances = np.linalg.norm(cluster_points - centroid, axis=1)

        # Get indices of top N closest points
        closest_points_indices = np.argsort(distances)[:top_n]

        # Store the global indices of these points
        top_closest_indices[i] = indices[closest_points_indices]

    return top_closest_indices

def to_networkx(data):
    # Convert a PyG graph to a networkx graph
    G = pyg_utils.to_networkx(data, to_undirected=True)
    return G