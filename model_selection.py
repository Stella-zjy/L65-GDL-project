import torch

from torch_geometric.datasets import TUDataset

from preprocess import *
from utils import *
from model import *
from train import *

import json


SEEDS = [8, 88, 888, 8888, 88888]
NUM_HIDDEN_UNIT = [40, 64]
NUM_NODES1 = [8, 12]
NUM_NODES2 = [2, 4]


def main():

    dataset = TUDataset(root='data/TUDataset', name='Mutagenicity')
    train_split = 0.8
    batch_size = 16
    visual_batch_size = 200

    model_stats = dict()

    for num_hidden_unit in NUM_HIDDEN_UNIT:
        for num_nodes1 in NUM_NODES1:
            for num_nodes2 in NUM_NODES2:

                model_name = f'model_hidden_unit_{num_hidden_unit}_diffpool_clusters_{num_nodes1}_{num_nodes2}'
                model_stats[model_name] = []

                for seed in SEEDS:

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    train_loader, val_loader, test_loader, _ = prepare_data(dataset, train_split, batch_size, visual_batch_size)

                    # Define the model
                    model = DiffPoolGNN(dataset.num_features, num_hidden_unit, dataset.num_classes, num_nodes1, num_nodes2)

                    lr = 0.001
                    epochs = 100
                    checkpoint_path = './checkpoints/' + model_name + f'_seed_{seed}.pth'
                    
                    # Training
                    experiment_runner(model, train_loader, val_loader, lr, epochs, checkpoint_path)

                    # Testing
                    model.eval() 
                    criterion = torch.nn.CrossEntropyLoss()
                    _, test_acc = test(model, test_loader, criterion)

                    model_stats[model_name].append(test_acc)
    

    file_path = "model_stats.json"
    with open(file_path, 'w') as file:
        json.dump(model_stats, file)


if __name__ == "__main__":
    main()