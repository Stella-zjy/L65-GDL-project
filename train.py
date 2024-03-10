import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj

from preprocess import *
from model import *


def train(model, loader, optimizer, criterion, model_type = 'diffpool'):
    model.train()
    total_loss = 0

    for data in loader:
        adj = to_dense_adj(data.edge_index, data.batch)
        batched_x = pad_features(data)
        
        optimizer.zero_grad()

        if model_type == 'diffpool':
            out, l, e, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)
            loss = criterion(out, data.y) + l + e

        elif model_type == '3diffpool_noembed':
            out, l, e, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)
            loss = criterion(out, data.y) + l + e

        elif model_type == '3diffpool_withembed':
            out, l, e, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)
            loss = criterion(out, data.y) + l + e

        # Used for training Vanilla GNN
        elif model_type == 'vanilla':
            out, _ = model(batched_x, adj)
            loss = criterion(out, data.y)
        
        elif model_type == 'vanilla2':
            out, l, e, _, _ = model(batched_x, adj)
            loss = criterion(out, data.y) + l + e
        
        elif model_type =='chocolate':
            out, l, e = model(batched_x, adj)
            loss = criterion(out, data.y) + l + e

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, criterion, model_type = 'diffpool'):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data in loader:
            adj = to_dense_adj(data.edge_index, data.batch)
            batched_x = pad_features(data)

            if model_type == 'diffpool':
                out, _, _, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)

            elif model_type == '3diffpool_noembed':
                out, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)

            elif model_type == '3diffpool_withembed':
                out, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(batched_x, adj)

            # Used for training Vanilla GNN
            elif model_type == 'vanilla':
                out, _ = model(batched_x, adj)
            
            elif model_type == 'vanilla2':
                out, _, _, _, _ = model(batched_x, adj)
            
            elif model_type =='chocolate':
                out, _, _ = model(batched_x, adj)
            
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

 

# Model training
def experiment_runner(model, train_loader, val_loader, lr, epochs, model_checkpoint, plot = False, model_type = 'diffpool'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, model_type)
        val_loss, val_acc = test(model, val_loader, criterion, model_type)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            # Save model checkpoint
            torch.save(model.state_dict(), model_checkpoint)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        # plt.savefig('model_train_val_loss.png')
        plt.legend()
        plt.show()