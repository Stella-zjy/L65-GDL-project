from collections import Counter
import torch
from utils import *

def cem_cluster_dict(activation_space_before, graph_nodes_enumerate):
    relevant_features = []
    for idx, real_node_number in enumerate(graph_nodes_enumerate):
        # Extract the features for the relevant nodes in the graph
        relevant_features.append(activation_space_before[idx, real_node_number, :])
    # Concatenate all the relevant features along the first dimension
    features = torch.cat(relevant_features, dim=0)
    
    # CEM-clustering
    _, indices = cem_clustering(features)
    

    # Use Counter to count occurrences
    occurrences = Counter(indices.tolist())

    # Sort the occurrences dictionary by values
    sorted_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)

    # Optionally, convert back to dictionary if needed
    sorted_occurrences_dict = dict(sorted_occurrences)
    # print(len(indices))
    # print(sorted_occurrences_dict)

    return sorted_occurrences_dict


def cem_plot(activation_space_before, graph_nodes_enumerate, DR_method, layer_num):
    relevant_features = []
    for idx, real_node_number in enumerate(graph_nodes_enumerate):
        # Extract the features for the relevant nodes in the graph
        relevant_features.append(activation_space_before[idx, real_node_number, :])
    # Concatenate all the relevant features along the first dimension
    features = torch.cat(relevant_features, dim=0)
    
    # CEM-clustering
    num_unique_rows, indices = cem_clustering(features)

    # Convert to numpy for dimensionality reduction
    features_np = features.detach().cpu().numpy()

    # Apply Dimension Reduction
    features_DR = dimension_reduction(features_np, DR_method)

    # ========================== Plotting Activation Space ==========================
    colors0 = plt.cm.jet(np.linspace(0, 1, num_unique_rows))

    for i, label in enumerate(indices):
        plt.scatter(features_DR[i, 0], features_DR[i, 1], color=colors0[int(label)], label=f'Cluster {int(label)}')

    plt.xlabel(f'{DR_method} Component 1')
    plt.ylabel(f'{DR_method} Component 2')
    plt.title(f'{DR_method} Visualization of Activation Space before {layer_num} (CEM colored with {num_unique_rows} clusters)')
    plt.show()  


def cem_guidance(sorted_cluster_dict, threshold = 0.02):
    total_nodes = sum(sorted_cluster_dict.values())
    boundary = total_nodes * threshold
    selected_clusters = dict()
    for key, value in sorted_cluster_dict.items():
        if value > boundary:
            selected_clusters[key] = value
    
    return selected_clusters, len(selected_clusters)