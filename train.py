import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj

from preprocess import *
from model import *


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        adj = to_dense_adj(data.edge_index, data.batch)
        batched_x = pad_features(data)
        
        optimizer.zero_grad()

        out, l, e, _, _, _, _, _, _, _, _ = model(batched_x, adj)
        
        loss = criterion(out, data.y) + l + e
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data in loader:
            adj = to_dense_adj(data.edge_index, data.batch)
            batched_x = pad_features(data)

            out, _, _, _, _, _, _, _, _, _, _= model(batched_x, adj)
            
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

 

# Model training
def experiment_runner(model, train_loader, test_loader, lr, epochs, model_checkpoint):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        if test_acc > best_acc:
            best_acc = test_acc
            # Save model checkpoint
            torch.save(model.state_dict(), model_checkpoint)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.savefig('model_train_test_loss.png')
    plt.legend()
    plt.show()