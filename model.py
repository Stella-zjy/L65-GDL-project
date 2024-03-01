import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool

from typing import Optional, Tuple
from torch import Tensor


def my_dense_diff_pool(x: Tensor, adj: Tensor, s: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()
    
    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    cluster_assignments = torch.argmax(s, dim=-1)

    return out, out_adj, link_loss, ent_loss, cluster_assignments


class GNN_fromtutorial(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN_fromtutorial, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = self.convs[step](x, adj, mask)
            x = x.permute(0, 2, 1)  # Reshape x to (N, C, L) for BatchNorm1d
            x = F.relu(self.bns[step](x))
            x = x.permute(0, 2, 1)  # Reshape back to (N, L, C)

        return x


class DiffPool_fromtutorial(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_nodes1, num_nodes2):
        super(DiffPool_fromtutorial, self).__init__()

        # we now set num_nodes1 = 8
        self.gnn1_pool = GNN_fromtutorial(num_features, 64, num_nodes1)
        self.gnn1_embed = GNN_fromtutorial(num_features, 64, 64)

        # we now set num_nodes2 = 2
        self.gnn2_pool = GNN_fromtutorial(64, 64, num_nodes2)
        self.gnn2_embed = GNN_fromtutorial(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        s_gnn1_pool = s
        x = self.gnn1_embed(x, adj, mask)
        x_gnn1_embed = x

        x, adj, l1, e1, cluster_assignment1 = my_dense_diff_pool(x, adj, s, mask)
        out1 = x
        # x_1 = s_0.t() @ z_0
        # adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        s_gnn2_pool = s
        x = self.gnn2_embed(x, adj)
        x_gnn2_embed = x

        x, adj, l2, e2, cluster_assignment2 = my_dense_diff_pool(x, adj, s)
        out2 = x

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x,dim=-1), l1 + l2, e1 + e2, s_gnn1_pool, x_gnn1_embed, s_gnn2_pool, x_gnn2_embed, cluster_assignment1, cluster_assignment2, out1, out2
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        #
        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        #
        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))
        # self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            # x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
        return x


class DiffPool(torch.nn.Module):
    def __init__(self, num_channels, num_clusters):
        super(DiffPool, self).__init__()
        self.gnn1_pool = GNN(num_channels, 32, num_clusters)
        self.gnn1_embed = GNN(num_channels, 32, 32)


    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        s = F.softmax(s, dim=1)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1, cluster_assignments = my_dense_diff_pool(x, adj, s, mask)
        return   x, adj, l1, e1, cluster_assignments 

# class DiffPool(torch.nn.Module):
#     def __init__(self, num_channels, num_clusters):
#         super(DiffPool, self).__init__()
#         self.s = torch.nn.Linear(num_channels, num_clusters)

#     def forward(self, x, adj, mask=None):
#         s = F.softmax(self.s(x), dim=1)
#         x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=mask)
#         return x, adj, l1, e1


class DiffPoolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(DiffPoolGNN, self).__init__()

        num_hidden_units = 32
        num_clusters1 = 8
        num_clusters2 = 2

        self.conv1 = DenseGCNConv(num_node_features, num_hidden_units)
        self.conv2 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv4 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv5 = DenseGCNConv(num_hidden_units, num_hidden_units)
        self.conv6 = DenseGCNConv(num_hidden_units, num_hidden_units)

        self.pool1 = DiffPool(num_hidden_units, num_clusters1)
        self.pool2 = DiffPool(num_hidden_units, num_clusters2)
        # self.pool3 = DiffPool(num_hidden_units, 1)

        self.lin1 = Linear(num_hidden_units, num_hidden_units)
        self.lin2 = Linear(num_hidden_units, num_classes)


    def forward(self, x, adj, mask=None):
        # Four GCN Layers
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = F.relu(self.conv3(x, adj))
        x = F.relu(self.conv4(x, adj))

        out1before = x

        # First Diff-Pooling Layer
        x, adj, l1, e1, cluster_assignments1 = self.pool1(x, adj)
        
        out1 = x

        # Two GCN Layers
        x = F.relu(self.conv5(x, adj))
        x = F.relu(self.conv6(x, adj))

        out2before = x

        # Second Diff-Pooling Layer
        x, adj, l2, e2, cluster_assignments2 = self.pool2(x, adj)
        
        out2 = x

        # Classifier
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1), l1 + l2, e1 + e2, out1, cluster_assignments1, out2, cluster_assignments2, out1before, out2before
    