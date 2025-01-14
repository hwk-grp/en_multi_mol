import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys


list_elem_types = [1, 6, 7, 8, 9]
n_atom_feats = len(list_elem_types)
en_pauling = [2.2,  0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0]


def get_mol_feats(mol, n_radius, n_fp):
    mol_feats = [ExactMolWt(mol), mol.GetRingInfo().NumRings(), Descriptors.MolLogP(mol), Descriptors.MolMR(mol),
                 Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), Descriptors.NumHeteroatoms(mol),
                 Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol), Descriptors.qed(mol),
                 rdMolDescriptors.CalcLabuteASA(mol), rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
                 rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumAmideBonds(mol),
                 rdMolDescriptors.CalcNumAromaticCarbocycles(mol), rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
                 rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcNumHeterocycles(mol),
                 rdMolDescriptors.CalcNumSaturatedCarbocycles(mol), rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
                 rdMolDescriptors.CalcNumSaturatedRings(mol)]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=n_radius, nBits=n_fp)
    morgan_array = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, morgan_array)
    mol_feats = np.append(mol_feats, morgan_array)
    mol_feats = np.append(mol_feats, MACCSkeys.GenMACCSKeys(mol))
    mol_feats = torch.tensor(np.array(mol_feats), dtype=torch.float).view(1, n_fp + 188)
    return mol_feats


def get_en_ref():
    tb_elements = fetch_table('elements')
    en_ref = np.nan_to_num(np.array(tb_elements['en_pauling']))
    return en_ref


def generate_bond_types(path_user_dataset):
    id_target = np.array(pd.read_excel(path_user_dataset))
    list_bond_types = []
    for i in tqdm(range(0, id_target.shape[0])):
        find_bt_mol(smiles=id_target[i, 0], list_bond_types=list_bond_types)
    return list_bond_types


def load_dataset(path_user_dataset, list_bond_types, n_fp, n_radius, task):
    list_mols = []
    id_target = np.array(pd.read_excel(path_user_dataset))
    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 1],
                                  list_bond_types=list_bond_types, task=task)
        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))
    return list_mols


def find_bt_mol(smiles, list_bond_types):
    # Smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # mol = Chem.MolFromSmiles(smiles)
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    n_atoms = mol.GetNumAtoms()
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    # Generate bond type set
    bond_types_set = set()
    for i in range(0, n_atoms):
        ndx_i = list_elem_types.index(atom_nums[i])
        bond_types_set.add((ndx_i, ndx_i))
        for j in range(i + 1, n_atoms):
            if adj_mat[i, j] == 1:
                ndx_j = list_elem_types.index(atom_nums[j])
                bond_types_set.add((min(ndx_i, ndx_j), max(ndx_i, ndx_j)))
    # Update list_bond_types with unique bond types
    list_bond_types.extend(bond_types_set - set(list_bond_types))
    return list_bond_types


def smiles_to_mol_graph(n_fp, n_radius, smiles, idx, target, list_bond_types, task):
    # Smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # mol = Chem.MolFromSmiles(smiles)
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    en_ref = en_pauling
    n_atoms = mol.GetNumAtoms()
    # Precompute atom nums
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    # Number of atom types and bond types
    n_at = len(list_elem_types)
    n_bt = len(list_bond_types)
    # Initialize lists
    atom_feats, bonds, list_nbrs, bond_feats, atom_types, atom_ens = [], [], [], [], [], []
    # atomic features
    for i in range(0, n_atoms):
        tmp_feats = np.zeros([n_at])
        ndx_i = list_elem_types.index(atom_nums[i])
        atom_types.append(ndx_i)
        tmp_feats[ndx_i] = 1
        atom_feats.append(tmp_feats)
        # electronegativity
        atom_ens.append(en_ref[atom_nums[i]-1])
        # neighbor list
        list_nbrs.append([])
        list_nbrs[i].append(i)
        bonds.append([i, i])
        # bond features including self-loop
        ndx_bt = list_bond_types.index((ndx_i, ndx_i))
        tmp_feats = np.zeros([n_bt])
        tmp_feats[ndx_bt] = 1
        bond_feats.append(tmp_feats)
        for j in range(i+1, n_atoms):
            if adj_mat[i, j] == 1:
                bonds.append([i, j])
                list_nbrs[i].append(j)
                ndx_j = list_elem_types.index(atom_nums[j])
                ndx_bt = list_bond_types.index((min(ndx_i, ndx_j), max(ndx_i, ndx_j)))
                tmp_feats = np.zeros([n_bt])
                tmp_feats[ndx_bt] = 1
                bond_feats.append(tmp_feats)
    if len(bonds) == 0:
        return None
    # list -> np array -> pytorch tensor
    atom_feats = torch.tensor(np.array(atom_feats), dtype=torch.float)
    atom_ens = torch.tensor(np.array(atom_ens), dtype=torch.float)
    atom_types = torch.tensor(np.array(atom_types), dtype=torch.int)
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    bond_feats = torch.tensor(np.array(bond_feats), dtype=torch.float)
    if task == 'reg':
        y = torch.tensor(target, dtype=torch.float).view(-1, 1)
    elif task == 'clf':
        y = torch.tensor(target, dtype=torch.long).view(1)
    else:
        print('task {} is not available'.format(task))
        exit()
    mol_feats = get_mol_feats(mol, n_radius=n_radius, n_fp=n_fp)
    return Data(x=atom_feats, y=y, edge_index=bonds, edge_attr=bond_feats, idx=idx, mol_feats=mol_feats,
                n_atoms=n_atoms, atom_types=atom_types, atom_ens=atom_ens)



