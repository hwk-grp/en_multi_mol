import random
import pandas as pd
import numpy as np
import torch
import utils.ml
import utils.chem_en_an as chem
from torch_geometric.loader import DataLoader
from utils.models import GCN
from sklearn.metrics import r2_score

# Experiment settings
en_dim = 2
dataset_name = 'updated_qm9_u0_atom'
data_dir = '../../../data/'
dataset = data_dir + dataset_name + '.xlsx'
batch_size, n_epochs, init_lr, l2_coeff, n_fp, n_radius = 32, 150, 0.001, 1E-07, 128, 6
dims, n_layers = [256, 64, 160, 32, 1], [2, 4, 4]
n_mol_feats = n_fp + 188
task = 'reg'  # clf or 'reg'
rand_seed = 2024 # from 2021 2030
opt = '../../../feats/gcn_en/en_an_2_2024.xlsx'# from 2021 2030

# Load dataset
print('Load molecular structures...')
list_bond_types = chem.generate_bond_types(dataset)
data = chem.load_dataset(dataset, list_bond_types, n_fp, n_radius, task, opt)

random.seed(rand_seed)
random.shuffle(data)
smiles, mols = [x[0] for x in data], [x[1] for x in data]

# Generate training and test datasets
n_train = int(len(data) * 0.9)
n_test = len(data) - n_train

train_data = mols[:n_train]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_smiles, test_data = smiles[n_train:], mols[n_train:]
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_targets = np.array([x.y.item() for x in train_data]).reshape(-1, 1)
test_targets = np.array([x.y.item() for x in test_data]).reshape(-1, 1)

# Model configuration
model = GCN(en_dim, n_mol_feats, dims, n_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)

# Train GNN model(s) using all training data
for i in range(0, n_epochs):
    train_loss = utils.ml.train(model, optimizer, train_loader, criterion)
    print(f'Epoch {i+1}/{n_epochs}, Train loss: {train_loss:.4f}')

# Test the trained GNN
preds = utils.ml.test(model, test_loader)
test_mae = np.mean(np.abs(test_targets - preds))
test_rmse = np.sqrt(np.mean((test_targets - preds) ** 2))
r2 = r2_score(test_targets, preds)
print(f'Test: MAE(gnn): {test_mae:.4f}\tTest RMSE(gnn): {test_rmse:.4f}\tTest R2 score: {r2:.4f}')

# Save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pd.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel(f'preds/gcn_{dataset_name}_{rand_seed}_an_{en_dim}.xlsx', index=False)

# Save the GCN model
torch.save(model.state_dict(), f'../../../pretrained_gcn/gcn_an_{rand_seed}.pth')

