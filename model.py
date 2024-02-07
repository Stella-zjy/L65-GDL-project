import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool, global_mean_pool



# class DiffPool(torch.nn.Module):
#     def __init__(self, num_channels, num_clusters):
#         super(DiffPool, self).__init__()
#         self.s = torch.nn.Linear(num_channels, num_clusters)

#     def forward(self, x, adj, mask=None):
#         s = F.softmax(self.s(x), dim=1)
#         x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=mask)
#         return x, adj, l1, e1
    

class DiffPool(torch.nn.Module):
    def __init__(self, num_channels, pooling_ratio):
        super(DiffPool, self).__init__()
        self.num_channels = num_channels
        self.pooling_ratio = pooling_ratio
        self.s = None  # Now defined dynamically in forward

    def forward(self, x, adj):
        num_nodes = x.size(0)
        num_clusters = max(int(num_nodes * self.pooling_ratio), 1)  # Calculate clusters dynamically

        # Initialize the linear layer here, with proper device assignment
        if self.s is None or self.s.out_features != num_clusters:
            self.s = torch.nn.Linear(self.num_channels, num_clusters).to(x.device)

        s = F.softmax(self.s(x), dim=1)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        return x, adj, l1, e1




class DiffPoolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, pooling_ratio):
        super(DiffPoolGNN, self).__init__()

        num_hidden_units = 16

        self.conv1 = DenseGCNConv(num_node_features, num_hidden_units)
        self.conv2 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv4 = DenseGCNConv(num_hidden_units, num_hidden_units)

        self.pool1 = DiffPool(num_hidden_units, pooling_ratio)
        self.pool2 = DiffPool(num_hidden_units, pooling_ratio)

        self.lin = Linear(num_hidden_units, num_classes)


    def forward(self, x, adj, batch):
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
        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return F.log_softmax(x, dim=1), l1 + l2, e1 + e2