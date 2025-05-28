import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from sklearn.cluster import KMeans


def find_non_zero_bt(id_target, new_en):
    at_bt = []
    at = sorted(new_en[:, 0].tolist())
    for i in range(len(at)):
        for j in range(i, len(at)):
            at_bt.append((at[i], at[j]))
    at_bond_count = np.zeros(len(at_bt), dtype=int)

    for molecule in tqdm(range(0, id_target.shape[0])):
        smiles = id_target[molecule, 0]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        n_atoms = mol.GetNumAtoms()
        atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atom_hybrids = [atom.GetHybridization() for atom in mol.GetAtoms()]
        for i in range(n_atoms):
            at1_i = (atom_nums[i], atom_hybrids[i])
            at1_i = str(at1_i)
            for j in range(n_atoms):
                if adj_mat[i, j] == 1:
                    at1_j = (atom_nums[j], atom_hybrids[j])
                    at1_j = str(at1_j)
                    bt = at_bt.index((min(at1_i, at1_j), max(at1_i, at1_j)))
                    at_bond_count[bt] += 1
    nz_bts = [at_bt[i] for i in range(len(at_bond_count)) if at_bond_count[i] != 0]

    return nz_bts

def generate_bt(new_en, n_rel, non_zero_bt):
    kmeans = KMeans(n_clusters=n_rel, random_state=42, n_init=10)
    atom_type_list = new_en[:, 0].tolist()
    # If en_dim>2, dimensionality reduction method like UMAP should be applied
    en_feat_x, en_feat_y = new_en[:, 1], new_en[:, 2]
    en_bond_type, dist_angle = [], []
    for at_i in range(len(atom_type_list)):
        num_i = atom_type_list[at_i]
        for at_j in range(at_i, len(atom_type_list)):
            num_j = atom_type_list[at_j]
            if (min(num_i, num_j), max(num_i, num_j)) in non_zero_bt:
                en_bond_type.append((at_i, at_j))
                x_diff, y_diff = (en_feat_x[at_i] - en_feat_x[at_j]), (en_feat_y[at_i] - en_feat_y[at_j])
                # Any other method can be applied to distinguish the difference of \chi_{ML}s
                dist = np.sqrt(x_diff**2 + y_diff**2)
                angle = np.arctan2(abs(y_diff), abs(x_diff))
                dist_angle.append([dist, angle])
    x = np.array(dist_angle)
    list_bond_types = kmeans.fit_predict(x)
    return atom_type_list, en_bond_type, list_bond_types


def load_dataset(path_user_dataset, n_fp, n_radius, path_opt_feat, n_rel):
    list_mols = []
    id_target = np.array(pd.read_excel(path_user_dataset))
    new_en = np.array(pd.read_excel(path_opt_feat))
    non_zero_bt = find_non_zero_bt(id_target, new_en)
    print(f'non_zero bond type with length: {len(non_zero_bt)}\n{non_zero_bt}')
    atom_type_list, en_bond_type, list_bond_types = generate_bt(new_en, n_rel, non_zero_bt)
    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 1], new_en=new_en,
                                  n_rel=n_rel, atom_type_list=atom_type_list, en_bond_type=en_bond_type,
                                  list_bond_types=list_bond_types)
        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))
    return list_mols


def smiles_to_mol_graph(n_fp, n_radius, smiles, idx, target, new_en, n_rel, atom_type_list, en_bond_type,
                        list_bond_types):
    # Smiles to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    n_atoms = mol.GetNumAtoms()
    mol.GetAtoms()

    # Initialize lists
    atom_feats, bond_index, bond_types, bond_feats, atom_ens = [], [], [], [], []

    # atomic features
    en_atom_type = new_en[:, 0].tolist()
    en_feat = new_en[:, 1:].astype(float)

    for i in range(0, n_atoms):
        atom_i = mol.GetAtoms()[i]
        atom_feat_i = str((atom_i.GetAtomicNum(), atom_i.GetHybridization()))

        # atomic feature
        atom_feats.append(en_feat[en_atom_type.index(atom_feat_i)])

        # bond index(int, int), bond_type(int), bond_feature(one-hot encoding)
        for j in range(i+1, n_atoms):
            if adj_mat[i, j] == 1:
                atom_j = mol.GetAtoms()[j]
                atom_feat_j = str((atom_j.GetAtomicNum(), atom_j.GetHybridization()))

                bond_index.append([i, j])
                bond_type_idx = en_bond_type.index((min(atom_type_list.index(atom_feat_i),
                                                        atom_type_list.index(atom_feat_j)),
                                                    max(atom_type_list.index(atom_feat_i),
                                                        atom_type_list.index(atom_feat_j))))
                bond_types.append(list_bond_types[bond_type_idx])
                tmp_feats = np.zeros([n_rel])
                tmp_feats[list_bond_types[bond_type_idx]] = 1
                bond_feats.append(tmp_feats)

    if len(bond_index) == 0:
        return None
    # list -> np array -> pytorch tensor
    atom_feats = torch.tensor(np.array(atom_feats), dtype=torch.float)
    bond_index = torch.tensor(bond_index, dtype=torch.long).t().contiguous()
    bond_types = torch.tensor(bond_types, dtype=torch.long).t().contiguous()
    bond_feats = torch.tensor(np.array(bond_feats), dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float).view(-1, 1)
    mol_feats = get_mol_feats(mol, n_radius=n_radius, n_fp=n_fp)
    return Data(x=atom_feats, y=y, edge_index=bond_index, edge_type=bond_types, edge_attr=bond_feats, idx=idx,
                mol_feats=mol_feats, n_atoms=n_atoms)


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



