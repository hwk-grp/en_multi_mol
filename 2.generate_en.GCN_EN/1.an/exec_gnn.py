import numpy
import random
import pandas
import torch
import utils.ml
import utils.chem_en_an as chem
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
from utils.models import GCN_EN

# Experiment settings
dataset_name = 'updated_qm9_u0_atom'
batch_size = 32
n_epochs = 150
init_lr = 0.001
l2_coeff = 1E-07
n_fp = 128
n_radius = 6
dims = [256, 64, 160, 32, 1]
n_layers = [2, 4, 4]
n_mol_feats = n_fp + 188
task = 'reg' # clf or 'reg'
rand_seed = 2024 # from 2021 to 2030
en_dim = 2 # can be replaced by arbitrary integer

# Load dataset
print('Load molecular structures...')
list_bond_types = chem.generate_bond_types('../../data/' + dataset_name + '.xlsx')
data = chem.load_dataset('../../data/' + dataset_name + '.xlsx', list_bond_types, n_fp, n_radius, task)

random.seed(rand_seed)
random.shuffle(data)
smiles = [x[0] for x in data]
mols = [x[1] for x in data]

# Generate training and test datasets
n_train = len(data)
train_data = mols[:n_train]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# Model configuration
if task == 'reg':
    criterion = torch.nn.MSELoss()
elif task == 'clf':
    criterion = torch.nn.CrossEntropyLoss()
else:
    print('task {} is not available'.format(task))
    exit()

# Train GNN model(s) using all training data
model = GCN_EN(chem.n_atom_feats, n_mol_feats, dims, n_layers, en_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)

for i in range(0, n_epochs):
    train_loss = utils.ml.train(model, optimizer, train_loader, criterion)
    print(f'Epoch {i}/{n_epochs}, Train loss: {train_loss:.4f}')
# Extract electronegativity the trained GNN
if task == 'reg':
    en_all, atom_type_all = utils.ml.extract_en(model, train_loader)

elif task == 'clf':
    en_all, atom_type_all = utils.ml.extract_en(model, train_loader)
else:
    print('Error: task not found')
    exit()

# save en and atom type
en_list, atom_type_list, atom_type_num = list(), list(), list()
list_elem_types = [1, 6, 7, 8, 9]
for i in range(0, en_all.shape[0]):
    atom_type_i = atom_type_all[i]
    if atom_type_i not in atom_type_num:
        atom_type_num.append(atom_type_i)
        atom_type_list.append(list_elem_types[atom_type_i])
        en_list.append(en_all[i])

en_results = list()
for i in range(len(atom_type_list)):
    en_results.append([atom_type_list[i], *en_list[i].reshape(-1,)])

df_en = pandas.DataFrame(en_results)

en_columns = ['Atom type']
for i in range(en_dim):
    en_columns.append(f'EN {i}')
df_en.columns = en_columns
df_en.to_excel('../../feats/gcn_en/1.an/en_an_' + str(en_dim) + '_' + str(rand_seed) + '.xlsx', index=False)
print(f'AN length: {str(en_dim)} Seed: {str(rand_seed)} done')


