import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
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


# Binarize the s matrix in DiffPool and return clustering based on binarized assignment strings
def s_clustering(s):
    s = F.softmax(s, dim=1)
    # Binarize 
    binarized_s = (s > 0.5).float()
    # Find unique rows
    unique_rows, indices = torch.unique(binarized_s, dim=0, return_inverse=True)
    # Number of unique rows
    num_unique_rows = unique_rows.size(0)

    return num_unique_rows, indices


def cem_clustering(features):
    features_softmax = F.softmax(features, dim=1)
    features_div = torch.div(features_softmax, torch.max(features_softmax, dim=-1)[0].unsqueeze(1))
    features_binarized = (features_div > 0.5).float()
    
    # Find unique rows
    unique_rows, indices = torch.unique(features_binarized, dim=0, return_inverse=True)
    # Number of unique rows
    num_unique_rows = unique_rows.size(0)
    
    return num_unique_rows, indices



# Visualization of the activation space before the first DiffPool layer
# Coloring according to K-means clustering of the raw activation space
def before_diffpool_plot(activation_space_before, s_gnn_pool, y_labels, graph_nodes_enumerate, DR_method, k, layer_num):
    """
    Input:
        activation_space_before: B * N * num_hidden_units
        graph_nodes_enumerate: containing the real nodes in each graph
        DR_method: 'PCA' or 'TSNE' or 'UMAP'
        k: cluster number for K-means clustering
    """
    relevant_features = []
    relevant_s = []
    for idx, real_node_number in enumerate(graph_nodes_enumerate):
        # Extract the features for the relevant nodes in the graph
        relevant_features.append(activation_space_before[idx, real_node_number, :])
        # To the same for the s matrix
        relevant_s.append(s_gnn_pool[idx, real_node_number, :])
    # Concatenate all the relevant features along the first dimension
    features = torch.cat(relevant_features, dim=0)
    # To the same for the revavent s
    s = torch.cat(relevant_s, dim=0)

    # s clustering
    num_unique_rows_s, indices_s = s_clustering(s)

    # CEM-clustering
    num_unique_rows_cem, indices_cem = cem_clustering(features)
    

    # Convert to numpy for dimensionality reduction
    features_np = features.detach().cpu().numpy()


    # K-means clustering of the raw activation space to get clustering labels for each node
    labels, centroids = kmeans_clustering(features_np, k)

    # Apply Dimension Reduction
    features_DR = dimension_reduction(features_np, DR_method)


    # ========================== Plotting Activation Space (Binarized-S colored) ==========================
    colors_s = plt.cm.jet(np.linspace(0, 1, num_unique_rows_s))

    for i, label in enumerate(indices_s):
        plt.scatter(features_DR[i, 0], features_DR[i, 1], color=colors_s[int(label)], label=f'Cluster {int(label)}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before {layer_num} (Binarised-S colored with {num_unique_rows_s} clusters)')
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()    

    # ========================== Plotting Activation Space (CEM colored) ==========================
    colors_cem = plt.cm.jet(np.linspace(0, 1, num_unique_rows_cem))

    for i, label in enumerate(indices_cem):
        plt.scatter(features_DR[i, 0], features_DR[i, 1], color=colors_cem[int(label)], label=f'Cluster {int(label)}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before {layer_num} (CEM colored with {num_unique_rows_cem} clusters)')
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()        


    # ========================== Plotting Activation Space (K-Means colored) ==========================

    # Generate a colormap with distinct colors
    colors = plt.cm.jet(np.linspace(0, 1, k))

    # plt.figure(figsize=(8, 6))

    for i, label in enumerate(range(k)):
        idx = labels == label
        plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors[i], label=f'Cluster {label}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before {layer_num} (K-Means colored)')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()    

    # ========================== Plotting Activation Space (Ground Truth Label colored) ========================= 
    
    colors2 = plt.cm.jet(np.linspace(0, 1, 2))

    # Track whether the label has been used
    labels_added = {0: False, 1: False}
    
    count = 0
    for i, label in enumerate(y_labels):
        idx = range(count, count + len(graph_nodes_enumerate[i]))
        if not labels_added[label]:
            plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors2[label], label=f'y = {label}')
            labels_added[label] = True
        else:
            plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors2[label])
        count += len(graph_nodes_enumerate[i])

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before {layer_num} (Ground Truth Label colored)')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()

    return labels, centroids



def after_diffpool_plot(activation_space_after, y_labels, graph_nodes_enumerate, cluster_assignments, DR_method, k, layer_num):

    batch_size, num_dp_clusters, _ = activation_space_after.shape

    relevant_cluster_assignments = []
    for idx, real_node_number in enumerate(graph_nodes_enumerate):
        # Extract diffpooled cluster labels for each graph
        relevant_cluster_assignments.append(torch.unique(cluster_assignments[idx, real_node_number]))


    relevant_features = []
    info_graph_node = []
    for b in range(batch_size):
        for c in relevant_cluster_assignments[b]:
            node_indexes = torch.where(cluster_assignments[b, graph_nodes_enumerate[b]] == c)[0]
            relevant_features.append(activation_space_after[b, int(c), :])
            info_graph_node.append([b, node_indexes.tolist()])
    
    # Stack the tensors
    features = torch.stack(relevant_features)

    # CEM clustering
    num_unique_rows, indices = cem_clustering(features)
    

    # Convert to numpy array
    features_np = features.detach().cpu().numpy()

    # K-means clustering of the raw activation space to get clustering labels for each node
    labels_km, centroids = kmeans_clustering(features_np, k)

    # Apply Dimension Reduction
    features_DR = dimension_reduction(features_np, DR_method)

    # Deal with diffpooled cluster labels
    labels_dp = torch.cat(relevant_cluster_assignments).detach().cpu().numpy()
    
    
    # ========================== Plotting Activation Space (CEM colored) ==========================
    colors0 = plt.cm.jet(np.linspace(0, 1, num_unique_rows))

    for i, label in enumerate(indices):
        plt.scatter(features_DR[i, 0], features_DR[i, 1], color=colors0[int(label)], label=f'Cluster {int(label)}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space after {layer_num} (CEM colored with {num_unique_rows} clusters)')
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
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
    plt.title(f'{DR_method} Visualization of Activation Space after {layer_num} (K-Means colored)')
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
    plt.title(f'{DR_method} Visualization of Activation Space after {layer_num} (DiffPool colored)')
    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust subplot parameters to give some space for the legend
    plt.subplots_adjust(right=0.75)
    plt.show()  

    # ========================== Plotting Activation Space (Ground Truth Label colored) ========================== 

    colors_label = plt.cm.jet(np.linspace(0, 1, 2)) 
    
    labels_added = {0: False, 1: False}
    
    count = 0
    plt.figure()
    for i, label in enumerate(y_labels):
        idx = range(count, count + len(relevant_cluster_assignments[i]))
        if not labels_added[label]:
            plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors_label[label], label=f'y = {label}')
            labels_added[label] = True
        else:
            plt.scatter(features_DR[idx, 0], features_DR[idx, 1], color=colors_label[label])
        count += len(relevant_cluster_assignments[i])

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space after {layer_num} (Ground Truth Label colored)')
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
                                            layer_num, k, num_of_diffpool, cluster_assignments1, graph_nodes):
    """
    Visualizes the graphs with specified nodes highlighted and labeled by concepts, along with their diff-pool cluster visualizations.
    """
    num_rows = 3 * len(cluster_top_features) 
    num_cols = max(len(v) for v in cluster_top_features.values())

    # Pre-compute the starting index for each graph
    starting_indices = [0]  # The first graph starts at index 0
    for num_nodes in graph_nodes[:-1]:  # Exclude the last graph as its starting index is not needed
        starting_indices.append(starting_indices[-1] + num_nodes)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12 * len(cluster_top_features) + 2), squeeze=False)
    fig.suptitle(
        f'{clustering_type} Central Instances, K-Means node concept with k={k}, {num_of_diffpool} DiffPool clusters, {layer_num}',
        y=1.005)

    axes = axes.flatten()

    for cluster_idx, (cluster, feature_indexes) in enumerate(cluster_top_features.items()):
        for feature_idx, feature_index in enumerate(feature_indexes):
            batch_index, node_indexes = node_info[feature_index]

            # Original Graph Visualization
            G = to_networkx(data[batch_index], to_undirected=True)
            
            # ======================== Highlighted Pooled Instances ======================== 
            start_index = starting_indices[batch_index]
            node_labels = {i: labels[start_index + i] for i in G.nodes()}
            node_colors = ["gray" if i not in node_indexes else "red" for i in G.nodes()]

            ax = axes[3 * (cluster_idx * num_cols + feature_idx)]
            nx.draw(G, ax=ax, node_color=node_colors, labels=node_labels, with_labels=True, node_size=50)
            ax.set_title(f"Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

            # ======================== Complete Pooled Instances ========================         
            all_nodes_diffpool_assignments = get_cluster_assignments(cluster_assignments1, graph_nodes)
            cluster_assignments = get_cluster_assignments_for_graph(batch_index, all_nodes_diffpool_assignments,
                                                                    graph_nodes)
            
            diffpool_node_colors = [cluster_assignments[i] for i in G.nodes()]
            
            ax = axes[3 * (cluster_idx * num_cols + feature_idx) + 1]
            nx.draw(G, ax=ax, node_color=diffpool_node_colors, labels=node_labels, with_labels=True,
                    cmap=plt.get_cmap('viridis'), node_size=50)
            ax.set_title(f"Diff-Pool of Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

            # ======================== Graph visualization with Atom Types ======================== 
            atom_mapping = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            atom_type = np.argmax(data[batch_index].x.detach().numpy(), axis = 1)
            
            atom_labels = {i: atom_mapping[atom_type[i]] for i in G.nodes()}
            ax = axes[3 * (cluster_idx * num_cols + feature_idx) + 2]
            nx.draw(G, ax=ax, node_color=atom_type, labels=atom_labels, with_labels=True,
                    cmap=plt.get_cmap('viridis'), node_size=50)
            ax.set_title(f"(Atoms) Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")
                        

    plt.tight_layout()
    plt.savefig(f"graph_visualization_{clustering_type}_central_instances.png")
    plt.show()


def get_cluster_assignments_for_graph_2nddiffpool(graph_index, all_nodes_diffpool_assignments, real_nodes):
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


def get_cluster_assignments_2nddiffpool(cluster_assignments, nodes_per_graph, graph_nodes_enumerate_2):
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

        # graph_assignments = cluster_assignments[graph_idx][:num_nodes]
        graph_assignments = []
        for i in range(num_nodes):
            graph_assignments.append(cluster_assignments[graph_idx, graph_nodes_enumerate_2[graph_idx][i]])

        # Since cluster_assignments already contain direct cluster indices, append them directly
        all_nodes_assignments.extend(graph_assignments)

    return all_nodes_assignments

def visualize_graphs_with_2nddiffpool_clusters(data, cluster_top_features, node_info, labels, clustering_type,
                                               layer_num, k, num_of_diffpool, cluster_assignments1, graph_nodes,
                                               graph_nodes_enumerate_2):
    """
    Visualizes the graphs with specified nodes highlighted and labeled by concepts, along with their diff-pool cluster visualizations.
    """
    num_rows = 2 * len(cluster_top_features)
    num_cols = max(len(v) for v in cluster_top_features.values())

    # Pre-compute the starting index for each graph
    starting_indices = [0]  # The first graph starts at index 0
    for num_nodes in graph_nodes[:-1]:  # Exclude the last graph as its starting index is not needed
        starting_indices.append(starting_indices[-1] + num_nodes)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12 * len(cluster_top_features) + 2), squeeze=False)
    fig.suptitle(
        f'{clustering_type} Central Instances, K-Means node concept with k={k}, {num_of_diffpool} DiffPool clusters, {layer_num}',
        y=1.005)

    axes = axes.flatten()
    all_nodes_diffpool_assignments = get_cluster_assignments_2nddiffpool(cluster_assignments1, graph_nodes,
                                                                         graph_nodes_enumerate_2)

    for cluster_idx, (cluster, feature_indexes) in enumerate(cluster_top_features.items()):
        for feature_idx, feature_index in enumerate(feature_indexes):
            batch_index, node_indexes = node_info[feature_index]

            # Original Graph Visualization
            G = nx.Graph()
            nodes_in_this_graph = graph_nodes_enumerate_2[batch_index]
            for node in nodes_in_this_graph:
                G.add_node(node)
            # G = to_networkx(data[batch_index], to_undirected=True)
            adjacencymatrix = data[batch_index, :, :]

            for i in nodes_in_this_graph:
                for j in nodes_in_this_graph:
                    if adjacencymatrix[i][j] > 1 and i != j:
                        G.add_edge(i, j)

            # ======================== Highlighted Pooled Instances ========================
            start_index = starting_indices[batch_index]

            node_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, node_mapping)
            pos = nx.spring_layout(G)

            node_labels = {i: labels[start_index + i] for i in list(range(len(G.nodes())))}
            node_colors = ["gray" if i not in node_indexes else "red" for i in G.nodes()]

            ax = axes[2 * (cluster_idx * num_cols + feature_idx)]

            nx.draw(G, ax=ax, node_color=node_colors, labels=node_labels, with_labels=True, node_size=50)
            ax.set_title(f"Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

            # ======================== Complete Pooled Instances ========================

            cluster_assignments = get_cluster_assignments_for_graph_2nddiffpool(batch_index,
                                                                                all_nodes_diffpool_assignments,
                                                                                graph_nodes)

            diffpool_node_colors = [cluster_assignments[i] for i in G.nodes()]

            ax = axes[2 * (cluster_idx * num_cols + feature_idx) + 1]
            nx.draw(G, ax=ax, node_color=diffpool_node_colors, labels=node_labels, with_labels=True,
                    cmap=plt.get_cmap('viridis'), node_size=50)
            ax.set_title(f"Diff-Pool of Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

            # # ======================== Graph visualization with Atom Types ========================
            # atom_mapping = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            # atom_type = np.argmax(data[batch_index].x.detach().numpy(), axis = 1)

            # atom_labels = {i: atom_mapping[atom_type[i]] for i in G.nodes()}
            # ax = axes[3 * (cluster_idx * num_cols + feature_idx) + 2]
            # nx.draw(G, ax=ax, node_color=atom_type, labels=atom_labels, with_labels=True,
            #         cmap=plt.get_cmap('viridis'), node_size=50)
            # ax.set_title(f"(Atoms) Cluster {cluster}, Feature {feature_index}, Graph{batch_index}")

    plt.tight_layout()
    plt.savefig(f"graph_visualization_{clustering_type}_central_instances.png")
    plt.show()