# -*- coding: utf-8 -*-
'''
Generate synthetic (toy) data, 
of the same expected shape as the actual data
'''

import numpy as np
import pickle

n_features = 768 # from the embeddings
n_instances = 1000 # made up


haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']

h = haplotypes[0]
for h in haplotypes:
  X = np.random.uniform(low=-1, high=1, size=(n_instances, n_features*2))
  filename = h+'_features.npy'
  np.save(filename,X, allow_pickle=False)
  #pickle.dump(X, open(filename, 'wb'))
  print('Written synthetic data in ',filename)
