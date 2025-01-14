# The Nature of Chemical Bonds in the Age of Artificial Intelligence: Revisiting Electronegativity of Organic Molecules

<img src="Figure 1.png" width=90% height=90%>

## Table 1: Atomic information for the generation of one-hot encoding vectors and corresponding $\chi_{ML}$
| **Symbol**|**Atomic information**|
|---|---|
|$\chi_{ML}^{AN}$|Atomic number|
|$\chi_{ML}^{AN+HY}$|Atomic number, Hybridization|
|$\chi_{ML}^{AN+HY+AR}$|Atomic number, Hybridization, Aromaticity|

This repository contains code to generate machine learning (ML) based multidimensional electronegativity, $\chi_{ML}$, associated with the research paper **[The Nature of Chemical Bonds in the Age of Artificial Intelligence: Revisiting Electronegativity of Organic Molecules](Future link) published in (Future Journal).
This project provides the novel method to generate $\chi_{ML}$ inspired by Pauling's electronegativity using linear regression (LR) and graph convolutional networks (GCN). Alternative version by using one of the physics-based metaheuristric algorithm, adaptive equilibrium optimizer (AEO) is also provided. 
The $\chi_{ML}$ values generated during this study can be found in the `feats` directory, and the pre-trained GCN model used for global optimization is available in the `pretrained_GCN` directory. If you wish to generate new version of $\chi_{ML}$, you can execute the Python scripts located in the `2.generate_en.GCN_EN` and `3.generate_en.AEO` folders to obtain the results (**exec_gnn.py** and **exec_aeo.py**).  Please note that due to the randomness inherent in ML models, executing the same code may produce slightly different outcomes in the detailed results. If you want to verify the performance of $\chi_{ML}$ as atomic feature in molecular ML, please execute Python scripts in `4.predictions`, **exec_gcn.py** and **exec_rgcn.py**. 

## Directories

### `1.prt_new_train_data`
This directory contains the code used to generate train data from QM9 dataset.
- **gen_train_data.py**: Main Python script to update target value. As mentioned in the paper, the atomization energy at 0K is updated.
- **chem_rand.py**: Python script contains the functions used in **gen_train_data.py**.

### `2.generate_en.GCN_EN`
This directory contains the code used to generate $\chi_{ML}$ using GCN and LR layer and named the model as **GCN_EN**. The detail structure of the **GCN_EN** model can be found in **models.py**. There exist three sub-directories, **1.an**, **2.an_hy**, and **3.an_hy_ar**, which means atomic information for the generation of one-hot encoding vectors (Table 1).  Each subdirectory contains same code except for the part to create one-hot encoding vectors. 
- **exec_gnn.py**: Main Python script to generate $\chi_{ML}$ by using GCN model and LR layer.
- **chem_en.py**: Python script contains useful functions used in **exec_gnn.py**.
- **ml.py**: Python script contains three functions; **train** and **test** to train and test ML models and **extract_en** to extract generated $\chi_{ML}$ and save it.
- **models.py**: Python script contains ML model class used in this work. 

### `3.generate_en.AEO`
This directory contains the code used to generate $chi_{ML}$ using AEO algorithm. There exist four sub-directories, **0.save_pretrained_gcn**, **1.an**, **2.an_hy**, and **3.an_hy_ar**. 
- **0.save_pretrained_gcn**: Subdirectory to save the pre-trained GCN model to use it as the fitness function in AEO algorithm. This subdirectory contains Python scripts similar to those in `2.generate.GCN_EN`.
- **exec_aeo.py**: Main Python script to generate $\chi_{ML}$ by using pre-train GCN model and AEO algorithm. 
- **chem_rand.py**: Python script contains useful functions used in **exec_aeo.py**.
- **ml.py**: ython script contains three functions; **train** and **test** to train and test ML models and **extract_en** to extract generated $\chi_{ML}$ and save it.
- **models.py**: Python script contains ML model class used in this work. 

### `4.prediction`
This directory contains the scripts to predict molecular properties using $\chi_{ML}$ as atomic features. GCN and relational GCN(RGCN) are used to predict molecular properties.

### `data`
This directory contains the datasets in **.xlsx** format used in this work. The datasets used in **4.prediction** are preprocessed.

### `en_sample`
This directory contains sample files of $\chi_{ML}$ from `2.generate_en.GCN_EN` and `3.generate_en.AEO`.

### `pretrained_gcn`
This directory contains the pre-trained GCN model from `3.generate_en.AEO/0.save_pretrained_gcn`.

### `environment.yaml`
To replicate the virtual environment used in this study.
