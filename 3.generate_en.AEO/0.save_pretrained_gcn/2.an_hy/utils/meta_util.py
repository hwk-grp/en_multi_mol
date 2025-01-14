import mealpy
import torch

from utils.ml import train
from torch_geometric.loader import DataLoader


class Problem(mealpy.utils.problem.Problem):
    def __init__(self, model, data, optimizer, criterion, batch_size, n_atom_types, lb, ub, minmax='min', **kwargs):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.n_atom_types = n_atom_types
        self.minmax = minmax
        super().__init__(lb, ub, minmax=minmax, **kwargs)

    def fit_func(self, sol):

        feat_at = torch.Tensor(sol.reshape(self.n_atom_types, -1))
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i].atom_types)):
                self.data[i].x[j] = feat_at[self.data[i].atom_types[j]]

        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i].edge_index[0])):
                j0 = self.data[i].edge_index[0, j]
                j1 = self.data[i].edge_index[1, j]
                self.data[i].edge_attr[j] = feat_at[self.data[i].atom_types[j0]] - feat_at[self.data[i].atom_types[j1]]

        train_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        train_loss = train(self.model, self.optimizer, train_loader, self.criterion)

        return train_loss

def update_at_feat(data, opt_feat_at):
   for i in range(0, len(data)):
        for j in range(0, len(data[i].atom_types)):
            data[i].x[j] = opt_feat_at[data[i].atom_types[j]]

   for i in range(0, len(data)):
        for j in range(0, len(data[i].edge_index[0])):
            j0 = data[i].edge_index[0, j]
            j1 = data[i].edge_index[1, j]
            data[i].edge_attr[j] = opt_feat_at[data[i].atom_types[j0]] - opt_feat_at[data[i].atom_types[j1]]
