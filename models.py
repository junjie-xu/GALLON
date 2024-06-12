import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.nn import global_mean_pool



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.L = num_layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if self.L == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(self.L - 1):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            # self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, batch, edge_index=None):
        xs = []  # layer0, layer1, layer2, mean_pool
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = self.bns[i](x)
            xs.append(x) # intermediate results
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        xs.append(x) # intermediate results
        
        x = global_mean_pool(x, batch)
        xs.append(x) # intermediate results
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x, xs


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_bn=True):
        super(GCN, self).__init__()
        self.L = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(self.L - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(GCNConv(hidden_channels, out_channels))
        self.use_bn = use_bn
        self.lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch):
        xs = []  # layer0, layer1, layer2, mean_pool
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x) # intermediate results
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x) # intermediate results
        
        x = global_mean_pool(x, batch)
        xs.append(x) # intermediate results
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x, xs


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, gat_heads):
        super(GAT, self).__init__()

        self.L = num_layers
        self.dropout = dropout
        heads = gat_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(self.L - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False))

        self.activation = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, K):
        super(ChebNet, self).__init__()
        self.K = K
        self.L = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(ChebConv(in_channels, hidden_channels, self.K))
        for _ in range(self.L - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, self.K))
        self.convs.append(ChebConv(hidden_channels, out_channels, self.K))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
