# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

#constants
n_features = 768

# select mode:
# -1: pre-train with all haplotypes
# 0 - 5: train with selected haplotype (keep training)
hp_types = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
MODE = -1
ALL = not (MODE >= 0 and MODE <=5)
if ALL: 
  print('Pretraining with all haplotypes')
else: 
  hp_types = [hp_types[MODE]]
  print('Training haplotype',hp_types)


# load data (features and labels)
X = []
Y = []
for h in hp_types:
 X.append(np.load('synthetic_data/'+h+'_features.npy'))
 Y.append(np.load('synthetic_data/'+h+'_labels.npy'))
 

tensor_x = torch.Tensor(np.array(X)) # transform to torch tensor
tensor_y = torch.Tensor(np.array(Y))

# split data into training, validation and test 

#dataloaders

my_dataset = TensorDataset(tensor_x.reshape(-1,768*2),tensor_y.reshape(-1,1)) 
my_dataloader = DataLoader(my_dataset) # create your dataloader





