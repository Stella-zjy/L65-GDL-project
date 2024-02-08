import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool


class DiffPool(torch.nn.Module):
    def __init__(self, num_channels, num_clusters):
        super(DiffPool, self).__init__()
        self.s = torch.nn.Linear(num_channels, num_clusters)

    def forward(self, x, adj, mask=None):
        s = F.softmax(self.s(x), dim=1)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=mask)
        return x, adj, l1, e1


class DiffPoolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(DiffPoolGNN, self).__init__()

        num_hidden_units = 64
        num_clusters1 = 8
        num_clusters2 = 2

        self.conv1 = DenseGCNConv(num_node_features, num_hidden_units)
        self.conv2 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv4 = DenseGCNConv(num_hidden_units, num_hidden_units)
        # self.conv5 = DenseGCNConv(num_hidden_units, num_hidden_units)
        # self.conv6 = DenseGCNConv(num_hidden_units, num_hidden_units)

        self.pool1 = DiffPool(num_hidden_units, num_clusters1)
        self.pool2 = DiffPool(num_hidden_units, num_clusters2)
        # self.pool3 = DiffPool(num_hidden_units, 1)

        self.lin1 = Linear(num_hidden_units, num_hidden_units)
        self.lin2 = Linear(num_hidden_units, num_classes)


    def forward(self, x, adj, mask=None):
        # Two GCN Layers
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))

        # First Diff-Pooling Layer
        x, adj, l1, e1 = self.pool1(x, adj)

        # Two GCN Layers
        x = F.relu(self.conv3(x, adj))
        x = F.relu(self.conv4(x, adj))

        # Second Diff-Pooling Layer
        x, adj, l2, e2 = self.pool2(x, adj)

        # Classifier
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1), l1 + l2, e1 + e2