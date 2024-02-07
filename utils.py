import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


def prepare_data(dataset, train_split, batch_size):
    dataset = dataset.shuffle()

    # Train test split
    train_idx = int(len(dataset) * train_split)
    train_set = dataset[:train_idx]
    test_set = dataset[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

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

    return train_loader, test_loader