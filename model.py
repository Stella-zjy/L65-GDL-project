import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, global_mean_pool
from math import ceil
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


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False):
        super(GNN, self).__init__()

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


class DiffPoolGNN(torch.nn.Module):
    def __init__(self, num_features, num_hidden_unit, num_classes, num_nodes1, num_nodes2):
        super(DiffPoolGNN, self).__init__()

        self.gnn1_pool = GNN(num_features, num_hidden_unit, num_nodes1)
        self.gnn1_embed = GNN(num_features, num_hidden_unit, num_hidden_unit)

        self.gnn2_pool = GNN(num_hidden_unit, num_hidden_unit, num_nodes2)
        self.gnn2_embed = GNN(num_hidden_unit, num_hidden_unit, num_hidden_unit)

        self.gnn3_embed = GNN(num_hidden_unit, num_hidden_unit, num_hidden_unit)

        self.lin1 = torch.nn.Linear(num_hidden_unit, num_hidden_unit)
        self.lin2 = torch.nn.Linear(num_hidden_unit, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        s_gnn1_pool = s
        x = self.gnn1_embed(x, adj, mask)
        x_gnn1_embed = x

        x, adj, l1, e1, cluster_assignment1 = my_dense_diff_pool(x, adj, s, mask)
        out1 = x
        new_graph_adj = adj
        
        s = self.gnn2_pool(x, adj)
        s_gnn2_pool = s
        x = self.gnn2_embed(x, adj)
        x_gnn2_embed = x

        x, adj, l2, e2, cluster_assignment2 = my_dense_diff_pool(x, adj, s)
        out2 = x

        x = self.gnn3_embed(x, adj)
        x_gnn3_embed = x

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x,dim=-1), l1 + l2, e1 + e2, s_gnn1_pool, x_gnn1_embed, s_gnn2_pool, x_gnn2_embed, x_gnn3_embed, new_graph_adj, cluster_assignment1, cluster_assignment2, out1, out2

    


class VanillaGNN(torch.nn.Module):
    def __init__(self, num_features, num_hidden_unit, num_classes):
        super(VanillaGNN, self).__init__()

        self.gnn1_embed = GNN(num_features, num_hidden_unit, num_hidden_unit)
        
        self.lin1 = torch.nn.Linear(num_hidden_unit, num_hidden_unit)
        self.lin2 = torch.nn.Linear(num_hidden_unit, num_classes)

    def forward(self, x, adj, mask=None):
        x = self.gnn1_embed(x, adj, mask)
        x_gnn1_embed = x

        x = global_mean_pool(x, mask)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), x_gnn1_embed
    


class VanillaGNN2(torch.nn.Module):
    def __init__(self, num_features, num_hidden_unit, num_classes, num_nodes1):
        super(VanillaGNN2, self).__init__()

        self.gnn1_pool = GNN(num_features, num_hidden_unit, num_nodes1)
        self.gnn1_embed = GNN(num_features, num_hidden_unit, num_hidden_unit)

        self.gnn2_embed = GNN(num_hidden_unit, num_hidden_unit, num_hidden_unit)

        self.lin1 = torch.nn.Linear(num_hidden_unit, num_hidden_unit)
        self.lin2 = torch.nn.Linear(num_hidden_unit, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l, e, cluster_assignment1 = my_dense_diff_pool(x, adj, s, mask)

        x = self.gnn2_embed(x, adj)
        x_gnn2_embed = x

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), l, e, x_gnn2_embed, cluster_assignment1



class ChocolateGNN(torch.nn.Module):
    def __init__(self, num_features, num_hidden_unit, num_classes, num_nodes1):
        super(ChocolateGNN, self).__init__()

        self.gnn1_pool = GNN(num_features, num_hidden_unit, num_nodes1)
        self.gnn1_embed = GNN(num_features, num_hidden_unit, num_hidden_unit)

        self.gnn2_embed = GNN(num_hidden_unit, num_hidden_unit, num_hidden_unit)

        self.gnn3_embed = GNN(num_hidden_unit, num_hidden_unit, num_hidden_unit)

        self.lin1 = torch.nn.Linear(num_hidden_unit, num_hidden_unit)
        self.lin2 = torch.nn.Linear(num_hidden_unit, num_classes)
    

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l, e, _ = my_dense_diff_pool(x, adj, s, mask)

        x = self.gnn2_embed(x, adj)

        x = global_mean_pool(x, mask)
        
        x = x.unsqueeze(1).expand(-1, 12, -1)
        
        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), l, e