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


def get_elem_angles_rand(path_user_dataset, seed):
    id_target = np.array(pd.read_excel(path_user_dataset))
    np.random.seed(seed)
    list_atom_types = []
    for i in tqdm(range(id_target.shape[0])):
        smiles = id_target[i, 0]
        list_atom_types = find_at_mol(smiles=smiles, list_atom_types=list_atom_types)
    elem_angles = np.random.uniform(0, 2*np.pi, size=len(list_atom_types))
    elem_angles = elem_angles.reshape(-1, 1)
    return list_atom_types, elem_angles


def generate_atom_types(path_user_dataset):
    id_target = np.array(pd.read_excel(path_user_dataset))
    list_atom_types = []
    for i in tqdm(range(0, id_target.shape[0])):
        list_atom_types = find_at_mol(smiles=id_target[i, 0], list_atom_types=list_atom_types)
    return list_atom_types


def find_at_mol(smiles, list_atom_types):
    # SMILES to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    n_atoms = mol.GetNumAtoms()
    atom_types_set = set()
    # precompute atom properties
    atom_properties = [(atom.GetAtomicNum(), int(atom.GetIsAromatic()), atom.GetHybridization()) for atom in mol.GetAtoms()]

    for i in range(0, n_atoms):
        atom_property = atom_properties[i]
        atom_types_set.add(atom_property)

    # Update list_atom_types with unique atom types
    list_atom_types.extend(atom_types_set - set(list_atom_types))
    list_atom_types = sorted(list_atom_types)
    return list_atom_types


def load_dataset(id_target, list_atom_types, n_fp, n_radius, elem_angle):
    list_mols = []
    for i in tqdm(range(0, id_target.shape[0])):
    #for i in range(0, id_target.shape[0]):
        mol = smiles_to_mol_graph(n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 1],
                                  list_atom_types=list_atom_types, elem_angle=elem_angle)

        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))

    return list_mols


def smiles_to_mol_graph(n_fp, n_radius, smiles, idx, target, list_atom_types, elem_angle):
    # SMILES to a RDKit object
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # initialize lists
    atom_feats, bonds, atom_types, atom_ens = [], [], [], []
    en_ref = en_pauling
    n_atoms = mol.GetNumAtoms()
    # precompute atom properties
    atom_properties = [(atom.GetAtomicNum(), int(atom.GetIsAromatic()), atom.GetHybridization()) for atom in mol.GetAtoms()]
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # adjacency matrix
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    # atomic features
    for i in range(0, n_atoms):
        tmp_atom_type = atom_properties[i]
        angle_i = elem_angle[list_atom_types.index(tmp_atom_type)]
        atom_i_en = en_ref[atom_nums[i] - 1]
        cos_, sin_ = np.cos(angle_i), np.sin(angle_i)
        tmp_atom_feats = np.concatenate((atom_i_en*cos_, atom_i_en*sin_))
        atom_feats.append(tmp_atom_feats)
        ndx_i = list_atom_types.index(atom_properties[i])
        atom_types.append(ndx_i)

        # electronegativity
        atom_ens.append(en_ref[atom_nums[i] - 1])

        # bond features
        for j in range(i + 1, n_atoms):
            if adj_mat[i, j] == 1:
                bonds.append([i, j])

    if len(bonds) == 0:
        return None

    # list -> numpy array -> pytorch tensor
    atom_feats = torch.tensor(np.array(atom_feats), dtype=torch.float)
    atom_ens = torch.tensor(np.array(atom_ens), dtype=torch.float)
    atom_types = torch.tensor(np.array(atom_types), dtype=torch.int)
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    y = torch.tensor(target, dtype=torch.float).view(-1, 1)
    mol_feats = get_mol_feats(mol, n_radius=n_radius, n_fp=n_fp)
    return Data(x=atom_feats, y=y, edge_index=bonds, idx=idx, mol_feats=mol_feats, n_atoms=n_atoms,
                atom_types=atom_types, atom_ens=atom_ens)


def update_node_features(data, angles):
    updated_graphs = []
    for graphs in data:
        graph = graphs[1]
        update_graph = graph.clone()
        atom_types = graph.atom_types
        atom_ens = graph.atom_ens
        updated_atom_feats = []
        for i, atom_type in enumerate(atom_types):
            angle = angles[atom_type]
            en = atom_ens[i]
            cos_, sin_ = np.cos(angle), np.sin(angle)
            tmp_atom_feats = np.concatenate((en*cos_, en*sin_))
            updated_atom_feats.append(tmp_atom_feats)
        updated_atom_feats = torch.tensor(np.array(updated_atom_feats), dtype=torch.float)
        update_graph.x = updated_atom_feats
        new_graph_tup = (graphs[0], update_graph, graphs[1])
        updated_graphs.append(new_graph_tup)
    return updated_graphs


