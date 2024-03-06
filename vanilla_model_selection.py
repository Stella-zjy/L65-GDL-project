import torch

from torch_geometric.datasets import TUDataset

from preprocess import *
from utils import *
from model import *
from train import *

import json


SEEDS = [123, 1234]
NUM_HIDDEN_UNIT = [32, 40, 64]


def main():

    dataset = TUDataset(root='data/TUDataset', name='Mutagenicity')

    train_split = 0.8
    batch_size = 16
    visual_batch_size = 200

    train_loader, val_loader, test_loader, _ = prepare_data(dataset, train_split, batch_size, visual_batch_size)

    model_stats = dict()

    for num_hidden_unit in NUM_HIDDEN_UNIT:

        model_vanilla_name = f'vanilla_model_hidden_unit_{num_hidden_unit}'
        model_stats[model_vanilla_name] = []

        for seed in SEEDS:

            torch.manual_seed(seed)

            # Define the model
            model_vanilla = Vanilla_GNN(dataset.num_features, num_hidden_unit, dataset.num_classes)

            lr = 0.001
            epochs = 20
            
            checkpoint_path = './checkpoints/' + model_vanilla_name + f'_seed_{seed}.pth'
            

            # Training
            experiment_runner(model_vanilla, train_loader, val_loader, lr, epochs, checkpoint_path)
            print()

            # Testing
            model_vanilla.eval() 
            criterion = torch.nn.CrossEntropyLoss()
            _, test_acc = test(model_vanilla, test_loader, criterion)

            model_stats[model_vanilla_name].append(test_acc)
    

    file_path =  f"vanilla_model_stats.json"
    with open(file_path, 'w') as file:
        json.dump(model_stats, file)


if __name__ == "__main__":
    main()
