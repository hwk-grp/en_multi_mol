import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.nn import global_add_pool


# Thomas N. Kipf and Max Welling,
# Semi-Supervised Classification with Graph Convolutional Networks
# International Conference on Learning Representations (ICLR) 2017
class GCN(nn.Module):
    # graph conovolutional network (GCN)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers):
        super(GCN, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(num_node_feats, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GCNConv(dims[0], dims[0]))
        self.gc.append(GCNConv(dims[0], dims[1]))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def prt_emb(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg

    def extract_node_feat(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))

        return h, g.atom_types


class GCN_EN(nn.Module):
    # graph conovolutional network (GCN)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers, en_dim):
        super(GCN_EN, self).__init__()
        self.n_layers = n_layers
        self.fc_en = nn.Linear(num_node_feats, en_dim)
        # self.fc_en2 = nn.Linear(en_dim, en_dim) for non-linear transformation in generating EN

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(en_dim, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GCNConv(dims[0], dims[0]))
        self.gc.append(GCNConv(dims[0], dims[1]))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        en = self.calc_en(g)

        h = F.silu(self.bn_gc(self.gc[0](en, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def calc_en(self, g):
        # generate embedding vector at given dimension
        fc_out = self.fc_en(g.x)
        # For non-linear transformation.
        # fc_out = F.silu(fc_out) 
        # fc_out = self.fc_en2(fc_out)

        # make the norm to unity
        fc_out = F.normalize(fc_out, p=2, dim=1)
        # multiply the Pauling electronegativity
        en = g.atom_ens.view(-1, 1) * fc_out

        return en

    def prt_emb(self, g):
        en = self.calc_en(g)

        h = F.silu(self.bn_gc(self.gc[0](en, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg

    def save_en(self, g):
        en = self.calc_en(g)

        return en, g.atom_types


class RGCN(nn.Module):
    def __init__(self, num_node_feats, n_mol_feats, num_rel, dims, n_layers):
        super(RGCN, self).__init__()
        self.n_layers = n_layers

        # relational graph convolutional layers
        self.gc = nn.ModuleList()
        self.gc.append(RGCNConv(num_node_feats, dims[0], num_rel, is_sorted=True))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(RGCNConv(dims[0], dims[0], num_rel, is_sorted=True))
        self.gc.append(RGCNConv(dims[0], dims[1], num_rel, is_sorted=True))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index, g.edge_type)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index, g.edge_type))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out
