import numpy
import pandas
import torch
from tqdm import tqdm
from rdkit import Chem

D_AA = {1: 102, 6: 83.0, 7: 33.2, 8: 34.4, 9: 64.6}

def save_updated_id_target(path_user_dataset):
    id_target = numpy.array(pandas.read_excel(path_user_dataset))

    for i in tqdm(range(0, id_target.shape[0])):
        id_target[i, 1] = update_target(id_target[i, 0], id_target[i, 1])

    return id_target

def update_target(smiles, target):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    adj_mat = Chem.GetAdjacencyMatrix(mol)

    mol_atom_num_list = list()
    for atom in mol.GetAtoms():
        mol_atom_num_list.append(atom.GetAtomicNum())
    n_atoms = mol.GetNumAtoms()

    # atomization energy > 0
    target *= -1

    # atomization energy - average of bond dissociation energy
    for i in range(0, mol.GetNumAtoms()):
        for j in range(i+1, mol.GetNumAtoms()):
            if adj_mat[i, j] == 1:
                target -= 0.5 * (D_AA[mol_atom_num_list[i]] + D_AA[mol_atom_num_list[j]])

    # target should be non-negative
    if target < 0:
        target = 0

    # unit of target = eV^(-1/2)
    target = numpy.sqrt(target / 23.06)

    return target
