import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def subgraph_to_tensor_ppgn(G, illicit_set, device="cpu"):
    node_ids = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_ids)}
    n = len(node_ids)

    X = torch.zeros((n, n, 1), dtype=torch.float32)

    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        X[i, j, 0] = 1.0
        X[j, i, 0] = 1.0

    Y = torch.tensor(
        [1.0 if node in illicit_set else 0.0 for node in node_ids],
        dtype=torch.float32
    )

    mask = torch.ones(n, dtype=torch.bool)

    return X.to(device), Y.to(device), mask.to(device)


class PPGNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, X):
        X2 = torch.einsum("ikd,kjd->ijd", X, X)
        H = torch.cat([X, X2], dim=-1)
        return self.mlp(H)


class NodePPGN_3WL(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.block1 = PPGNBlock(in_channels, hidden_dim)
        self.block2 = PPGNBlock(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def aggregate(self, X):
        return X.mean(dim=1)

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        H = self.aggregate(X)
        return self.classifier(H).squeeze(-1)

    def get_embeddings(self, X):
        X = self.block1(X)
        X = self.block2(X)
        return self.aggregate(X)


class NodePairGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.block = PPGNBlock(in_channels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def aggregate(self, X):
        return X.mean(dim=1)

    def forward(self, X):
        X = self.block(X)
        H = self.aggregate(X)
        return self.classifier(H).squeeze(-1)

    def get_embeddings(self, X):
        X = self.block(X)
        return self.aggregate(X)


class NodeGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.classifier(x).squeeze(-1)

    def get_embeddings(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))
    
