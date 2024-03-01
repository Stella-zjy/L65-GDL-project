import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans


# Get the actual number of nodes in each graph of the visual_data_batch
# len(graph_node) = batch size of visual data loader
def real_graph_node(visual_data_batch):
    graph_nodes = []
    batch_size = len(visual_data_batch)
    for i in range(batch_size):
        graph_nodes.append(visual_data_batch[i].num_nodes)
    return graph_nodes


def kmeans_clustering(features, k):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids


# Dimension Reduction for 2D visualization
def dimension_reduction(features_np, DR_method):
    if DR_method == 'PCA':
        pca = PCA(n_components=2)
        features_DR = pca.fit_transform(features_np)
    elif DR_method == 'TSNE':
        tsne = TSNE(n_components=2)
        features_DR = tsne.fit_transform(features_np)
    elif DR_method == 'UMAP':
        reducer = umap.UMAP(n_components=2)
        features_DR = reducer.fit_transform(features_np)
    return features_DR



# Visualization of the activation space before the first DiffPool layer
# Coloring according to K-means clustering of the raw activation space
def before_first_diffpool_plot(activation_space_before, graph_nodes, DR_method, k):
    """
    Input:
        activation_space_before: B * N * num_hidden_units
        graph_nodes: len() = B, containing the real number of nodes in each graph
        DR_method: 'PCA' or 'TSNE' or 'UMAP'
        k: cluster number for K-means clustering
    """
    relevant_features = []
    for idx, real_node_number in enumerate(graph_nodes):
        # Extract the features for the relevant nodes in the graph
        relevant_features.append(activation_space_before[idx, :real_node_number, :])
    # Concatenate all the relevant features along the first dimension
    features = torch.cat(relevant_features, dim=0)

    # Convert to numpy for dimensionality reduction
    features_np = features.detach().cpu().numpy()

    # K-means clustering of the raw activation space to get clustering labels for each node
    labels, centroids = kmeans_clustering(features_np, k)

    # Apply Dimension Reduction
    features_DR = dimension_reduction(features_np, DR_method)


    # ========================== Plotting Activation Space ==========================

    # Generate a colormap with distinct colors
    colors = plt.cm.jet(np.linspace(0, 1, k))

    # plt.figure(figsize=(8, 6))

    for i, label in enumerate(range(k)):
        idx = labels == label
        plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors[i], label=f'Cluster {label}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before First DiffPool')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()            
        
    return labels, centroids




def after_first_diffpool_plot(activation_space_after, graph_nodes, cluster_assignments, DR_method, k):

    batch_size, num_dp_clusters, _ = activation_space_after.shape

    relevant_cluster_assignments = []
    for idx, real_node_number in enumerate(graph_nodes):
        # Extract diffpooled cluster labels for each graph
        relevant_cluster_assignments.append(torch.unique(cluster_assignments[idx, :real_node_number]))


    relevant_features = []
    info_graph_node = []
    for b in range(batch_size):
        for c in relevant_cluster_assignments[b]:
            node_indexes = torch.where(cluster_assignments[b, :graph_nodes[b]] == c)[0]
            relevant_features.append(activation_space_after[b, int(c), :])
            info_graph_node.append([b, node_indexes.tolist()])
    
    
    # Stack the tensors
    features = torch.stack(relevant_features)
    # Convert to numpy array
    features_np = features.detach().cpu().numpy()

    # K-means clustering of the raw activation space to get clustering labels for each node
    labels_km, centroids = kmeans_clustering(features_np, k)

    # Apply Dimension Reduction
    features_DR = dimension_reduction(features_np, DR_method)


    # Deal with diffpooled cluster labels
    labels_dp = torch.cat(relevant_cluster_assignments).detach().cpu().numpy()
    

    # ========================== Plotting Activation Space (no clustering) ==========================
    plt.figure()
    plt.scatter(features_DR[:, 0], features_DR[:, 1])
    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space after First DiffPool')
    plt.show()            

    # ========================== Plotting Activation Space (K-Means colored) ==========================
    
    # Generate a colormap with distinct colors according to k-means clusters
    colors_km = plt.cm.jet(np.linspace(0, 1, k))

    plt.figure()
    for i, label in enumerate(range(k)):
        idx = labels_km == label
        plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors_km[i], label=f'Cluster {label}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space after First DiffPool (K-Means colored)')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()   

    # ========================== Plotting Activation Space (Diff-Pooling colored) ==========================  
           
    # Generate a colormap with distinct colors according to diffpool clusters
    colors_dp = plt.cm.jet(np.linspace(0, 1, num_dp_clusters)) 

    plt.figure()
    for i, label in enumerate(range(num_dp_clusters)):
        idx = labels_dp == label
        plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors_dp[i], label=f'Cluster {label}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space after First DiffPool (DiffPool colored)')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()  

    return labels_km, centroids, colors_km, colors_dp, labels_dp, features_np, info_graph_node




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


def get_cluster_assignments_for_graph(graph_index, all_nodes_diffpool_assignments, real_nodes):
    """
    Return the cluster assignments for each node in the specified graph.

    Parameters:
    - graph_index: Integer, the index of the graph in the batch.
    - all_nodes_diffpool_assignments: List of integers, cluster assignments for each node across all graphs in the batch.
    - real_nodes: List of integers, the number of real nodes in each graph.

    Returns:
    - List of integers, the cluster assignments for each node in the specified graph.
    """
    # Calculate the start index in all_nodes_diffpool_assignments for the specified graph
    start_index = sum(real_nodes[:graph_index])

    # Calculate the end index using the number of nodes in the specified graph
    end_index = start_index + real_nodes[graph_index]

    # Extract and return the cluster assignments for the specified graph
    return all_nodes_diffpool_assignments[start_index:end_index]


def get_cluster_assignments(cluster_assignments, nodes_per_graph):
    """
    Create a list of cluster assignments for each node across all graphs, given direct cluster indices.

    Parameters:
    - cluster_assignments: Tensor of shape (num_graphs, max_nodes) containing direct cluster indices for each node.
    - nodes_per_graph: List of integers indicating the number of true nodes in each graph.

    Returns:
    - List of integers representing the cluster assignment for each node.
    """
    all_nodes_assignments = []

    # Iterate over each graph
    for graph_idx, num_nodes in enumerate(nodes_per_graph):
        # Extract the cluster assignments for the actual number of nodes in the current graph
        graph_assignments = cluster_assignments[graph_idx][:num_nodes]

        # Since cluster_assignments already contain direct cluster indices, append them directly
        all_nodes_assignments.extend(graph_assignments.tolist())

    return all_nodes_assignments



def calculate_centroids_with_padding(data, labels):

    max_label = np.max(labels)
    centroids = np.zeros((max_label+1, data.shape[1]))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_data = data[labels == label]
        centroids[labels] = cluster_data.mean(axis = 0)

    return centroids



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


def visualize_graphs_with_diffpool_clusters(data, cluster_top_features, node_info, labels, clustering_type,
                                            reduction_type, layer_num, k, num_of_diffpool, cluster_assignments1, graph_nodes):
    """
    Visualizes the graphs with specified nodes highlighted and labeled by concepts, along with their diff-pool cluster visualizations.

    Parameters:
    - diffpool_cluster_assignments: A list or dictionary containing the diff-pool cluster assignments for nodes in each graph.
    """
    num_rows = 2 * len(cluster_top_features)  # Double the number of rows for original and diff-pool visualizations
    num_cols = max(len(v) for v in cluster_top_features.values())

    # Pre-compute the starting index for each graph
    starting_indices = [0]  # The first graph starts at index 0
    for num_nodes in graph_nodes[:-1]:  # Exclude the last graph as its starting index is not needed
        starting_indices.append(starting_indices[-1] + num_nodes)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * len(cluster_top_features) + 2), squeeze=False)
    fig.suptitle(
        f'Nearest Instances to {clustering_type}, with k = {k}, number of diffpool clusters {num_of_diffpool}, Cluster Centroid for {reduction_type} Activations of Layer {layer_num}',
        y=1.005)

    axes = axes.flatten()

    for cluster_idx, (cluster, feature_indexes) in enumerate(cluster_top_features.items()):
        for feature_idx, feature_index in enumerate(feature_indexes):
            batch_index, node_indexes = node_info[feature_index]

            # Original Graph Visualization
            G = to_networkx(data[batch_index], to_undirected=True)
            start_index = starting_indices[batch_index]
            node_labels = {i: labels[start_index + i] for i in G.nodes()}
            node_colors = ["gray" if i not in node_indexes else "red" for i in G.nodes()]

            ax = axes[2 * (cluster_idx * num_cols + feature_idx)]
            nx.draw(G, ax=ax, node_color=node_colors, labels=node_labels, with_labels=True, node_size=50)
            ax.set_title(f"Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

            # Diff-Pool Cluster Visualization
            # Assuming diffpool_cluster_assignments[batch_index] gives you the cluster assignment for each node in the graph
            # You might need to adjust this part based on your data structure
            all_nodes_diffpool_assignments = get_cluster_assignments(cluster_assignments1, graph_nodes)
            cluster_assignments = get_cluster_assignments_for_graph(batch_index, all_nodes_diffpool_assignments,
                                                                    graph_nodes)
            # You need to determine how to use cluster_assignments to color nodes or otherwise visualize the diff-pool clusters.
            # This is a placeholder for the actual visualization logic.
            # For simplicity, let's assume each node's color is determined by its cluster assignment:
            diffpool_node_colors = [cluster_assignments[i] for i in G.nodes()]

            ax = axes[2 * (cluster_idx * num_cols + feature_idx) + 1]
            nx.draw(G, ax=ax, node_color=diffpool_node_colors, labels=node_labels, with_labels=True,
                    cmap=plt.get_cmap('viridis'), node_size=50)
            ax.set_title(f"Diff-Pool of Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

    plt.tight_layout()
    plt.show()