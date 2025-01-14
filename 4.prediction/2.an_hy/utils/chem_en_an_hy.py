import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys

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


def generate_atom_types(path_user_dataset):
    id_target = np.array(pd.read_excel(path_user_dataset))
    list_atom_types = []
    for i in tqdm(range(0, id_target.shape[0])):
        list_atom_types = find_at_mol(smiles=id_target[i, 0], list_atom_types=list_atom_types)
    return list_atom_types


def generate_bond_types(path_user_dataset, list_atom_types):
    id_target = np.array(pd.read_excel(path_user_dataset))
    list_bond_types = []

    for i in tqdm(range(0, id_target.shape[0])):
        find_bt_mol(smiles=id_target[i, 0], list_atom_types=list_atom_types, list_bond_types=list_bond_types)

    return list_bond_types


def load_dataset(path_user_dataset, list_atom_types, list_bond_types, n_fp, n_radius, task, path_opt_feat):
    list_mols = []
    id_target = np.array(pd.read_excel(path_user_dataset))
    new_en = np.array(pd.read_excel(path_opt_feat))
    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 1], new_en=new_en,
                                  list_atom_types=list_atom_types, list_bond_types=list_bond_types, task=task)

        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))

    return list_mols


def find_at_mol(smiles, list_atom_types):
    # smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    #mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    atom_types_set = set()
    # precompute atom properties
    atom_properties = [(atom.GetAtomicNum(), atom.GetHybridization()) for atom in mol.GetAtoms()]

    for i in range(0, n_atoms):
        atom_property = atom_properties[i]
        atom_types_set.add(atom_property)

    # Update list_atom_types with unique atom types
    list_atom_types.extend(atom_types_set - set(list_atom_types))

    return list_atom_types


def find_bt_mol(smiles, list_atom_types, list_bond_types):
    # smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    #mol = Chem.MolFromSmiles(smiles)
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    n_atoms = mol.GetNumAtoms()
    bond_types_set = set()
    # precompute atom properties
    atom_properties = [(atom.GetAtomicNum(), atom.GetHybridization()) for atom in mol.GetAtoms()]

    for i in range(0, n_atoms):
        atom_type_i = atom_properties[i]
        ndx_i = list_atom_types.index(atom_type_i)
        bond_types_set.add((ndx_i, ndx_i))

        for j in range(i + 1, n_atoms):
            if adj_mat[i, j] == 1:
                atom_type_i = atom_properties[j]
                ndx_j = list_atom_types.index(atom_type_i)
                bond_types_set.add((min(ndx_i, ndx_j), max(ndx_i, ndx_j)))

    # Update list_bond_types with unique bond types
    list_bond_types.extend(bond_types_set - set(list_bond_types))
    return list_bond_types


def smiles_to_mol_graph(n_fp, n_radius, smiles, idx, target, new_en, list_atom_types, list_bond_types, task):
    # smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    #mol = Chem.MolFromSmiles(smiles)

    # number of atom types and bond types
    n_at, n_bt = len(list_atom_types), len(list_bond_types)

    # initialize lists
    atom_feats, bonds, list_nbrs, bond_feats, atom_types, atom_ens = [], [], [], [], [], []
    en_ref = en_pauling
    n_atoms = mol.GetNumAtoms()

    # precompute atom properties
    atom_properties = [(atom.GetAtomicNum(), atom.GetHybridization()) for atom in mol.GetAtoms()]

    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # adjacency matrix
    adj_mat = Chem.GetAdjacencyMatrix(mol)

    # atomic features
    en_atom_type = new_en[:, 0].astype(str).tolist()
    en_feat = new_en[:, 1:].astype(float)
    for i in range(0, n_atoms):
        tmp_atom_type = str(atom_properties[i])
        atom_feats.append(en_feat[en_atom_type.index(tmp_atom_type)])
        ndx_i = list_atom_types.index(atom_properties[i])
        atom_types.append(ndx_i)

        # electronegativity
        atom_ens.append(en_ref[atom_nums[i] - 1])

        # neighbor list
        list_nbrs.append([])
        list_nbrs[i].append(i)
        bonds.append([i, i])

        # self-loop
        ndx_bt = list_bond_types.index((ndx_i, ndx_i))
        tmp_feats = np.zeros([n_bt])
        tmp_feats[ndx_bt] = 1
        bond_feats.append(tmp_feats)

        # bond features
        for j in range(i + 1, n_atoms):
            if adj_mat[i, j] == 1:
                bonds.append([i, j])
                list_nbrs[i].append(j)
                ndx_j = list_atom_types.index(atom_properties[j])
                ndx_bt = list_bond_types.index((min(ndx_i, ndx_j), max(ndx_i, ndx_j)))

                tmp_feats = np.zeros([n_bt])
                tmp_feats[ndx_bt] = 1
                bond_feats.append(tmp_feats)

    if len(bonds) == 0:
        return None

    # list -> numpy array -> pytorch tensor
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
