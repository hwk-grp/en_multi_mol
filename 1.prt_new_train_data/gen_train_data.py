import numpy
import pandas
import chem_rand as chem
import torch
en_ref = torch.tensor([2.20, 0.0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98])
dataset_name = 'qm9_u0_atom'

# Load dataset
id_target = chem.save_updated_id_target('../data/' + dataset_name + '.xlsx')

df1 = pandas.DataFrame(id_target, columns=['smiles', 'E_noncov'])
df1.to_excel("../data/updated_qm9_u0_atom.xlsx", index=False)
