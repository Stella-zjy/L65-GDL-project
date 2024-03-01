import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


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