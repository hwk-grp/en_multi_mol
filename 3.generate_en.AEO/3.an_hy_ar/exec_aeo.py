import random
import numpy as np
import pandas as pd
import torch
import utils.ml
import utils.chem_rand_an_hy_ar as chem
from torch_geometric.loader import DataLoader
from utils.models import GCN
import mealpy
from mealpy.physics_based import EO
from sklearn.metrics import r2_score


class GNNProblem(mealpy.utils.problem.Problem):
    def __init__(self, gcn_model, id_target, list_atom_types, elem_angles, lb, ub, batch_size, n_fp, n_radius, minmax='min', **kwargs):
        self.bounds = None
        self.gcn_model = gcn_model  # pre-trained
        self.id_target = id_target  # id_target for optimization
        self.list_atom_types = list_atom_types
        self.elem_angles = elem_angles
        self.lb = lb
        self.ub = ub
        self.batch_size = batch_size
        self.n_fp = n_fp
        self.n_radius = n_radius
        self.pre_data = chem.load_dataset(self.id_target, self.list_atom_types, self.n_fp, self.n_radius, self.elem_angles)
        super().__init__(lb=self.lb, ub=self.ub, minmax=minmax, **kwargs)

    def pnt_func(self, sol):
        penalty = 0.0
        for i, angle in enumerate(sol):
            if angle < self.lb[i]:
                penalty += 1e4
            elif angle > self.ub[i]:
                penalty += 1e4
        return penalty

    def fit_func(self, sol):  # sol = angle
        angles = sol.reshape(-1, 1)
        data = chem.update_node_features(self.pre_data, angles)
        mol = [x[1] for x in data]
        target = np.array([x.y.item() for x in mol]).reshape(-1, 1)
        loader = DataLoader(mol, batch_size=self.batch_size, shuffle=False)
        preds = utils.ml.test(self.gcn_model, loader)
        fitness = np.sqrt(np.mean(target - preds) ** 2)
        pnt_penalty = self.pnt_func(sol)
        fitness += pnt_penalty
        return fitness

# Experiment setting
dataset = '../../data/updated_qm9_u0_atom.xlsx'
model_path = '../../pretrained_gcn/gcn_an_hy_ar_2024.pth'# from 2021 to 2030
trained_gcn_state_dict = torch.load(model_path)
batch_size, n_epochs, init_lr, l2_coeff, n_fp, n_radius = 32, 150, 0.001, 1.61050628437249E-06, 256, 4
dims, n_layers = [256, 32, 96, 128, 1], [2, 4, 3]
n_mol_feats = n_fp + 188
seed_feat = 2024 # from 2021 to 2030
seed_train = 2024 # from 2021 to 2030
opt_epochs = 500

# Load dataset
print('Load molecular structures...')
list_atom_types, elem_angles = chem.get_elem_angles_rand(dataset, seed_feat)
id_target = np.array(pd.read_excel(dataset))
random.seed(seed_train)
ndx = [x for x in range(0, id_target.shape[0])]
random.shuffle(ndx)
id_target_all = id_target[ndx]

# use 10 % of total data for training
n_opt = int(0.1 * len(ndx))
id_target = id_target_all[:n_opt]

# meta-heuristic search
num_atom_types = len(list_atom_types)  # |angle| = 1, so n_dim = n_atom_types
lbs, ubs = [0] * num_atom_types, [2*np.pi]*num_atom_types
model = GCN(2, n_mol_feats, dims, n_layers)
model.load_state_dict(trained_gcn_state_dict)

problem = GNNProblem(gcn_model=model, id_target=id_target, list_atom_types=list_atom_types, elem_angles=elem_angles,
                     lb=lbs, ub=ubs, l2_coeff=l2_coeff, n_fp=n_fp, n_radius=n_radius)

EO_model = EO.AdaptiveEO(epoch=opt_epochs)
best_sol, best_val = EO_model.solve(problem)

print("Global Best Solution:", best_sol)
print("Global Best Fitness:", best_val)

result = []
for i in range(len(list_atom_types)):
    angle = best_sol[i]
    cos, sin = np.cos(angle), np.sin(angle)
    result.append([list_atom_types[i], cos, sin])
df = pd.DataFrame(result)
df.columns = ['Atom type', 'cos', 'sin']
df.to_excel(f'../../feats/aeo/3.an_hy_ar/en_an_hy_ar_gcn_aeo_2_{seed_train}.xlsx', index=False)

# Generate training and test datasets
best_sol = best_sol.reshape(-1, 1)
train_test_id = id_target_all[n_opt:]
train_test_data = chem.load_dataset(train_test_id, list_atom_types, n_fp, n_radius, best_sol)
smiles, mols = [x[0] for x in train_test_data], [x[1] for x in train_test_data]

n_train = int(0.8 * len(id_target_all))
n_test = len(id_target_all) - n_train - n_opt

train_data, test_data, test_smiles = mols[:n_train], mols[n_train:], smiles[n_train:]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size)
train_targets = np.array([x.y.item() for x in train_data])
test_targets = np.array([x.y.item() for x in test_data]).reshape(-1, 1)

train_test_model = GCN(2, n_mol_feats, dims, n_layers)
train_test_criterion = torch.nn.MSELoss()
train_test_optimizer = torch.optim.Adam(train_test_model.parameters(), lr=init_lr, weight_decay=l2_coeff)

# Train graph neural network (GNN)
print('Train the GNN-based predictor...')
for i in range(n_epochs):
    train_loss = utils.ml.train(train_test_model, train_test_optimizer, train_loader, train_test_criterion)
    print(f'Epoch [{i+1}/{n_epochs}]\tTrain loss: {train_loss:.4f}')

# Test the trained GNN
preds = utils.ml.test(train_test_model, test_loader)
test_mae = np.mean(np.abs(test_targets - preds))
test_rmse = np.sqrt(np.mean(test_targets - preds) ** 2)
test_r2 = r2_score(test_targets, preds)
print(f'Test MAE: {test_mae:.4f}\tTest RMSE: {test_rmse:.4f}\t Test R2 score: {test_r2:.4f}')

# Save prediction results (LR)
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pd.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel(f'./preds/preds_en_an_hy_ar_gcn_aeo_2_{seed_train}.xlsx', index=False)


