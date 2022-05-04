import pandas as pd
import numpy as np

verb = True # to print debug info
##I didn´t try it with all the files, but for the biggest ones none of the proteins_id are repeated
##between them
#1. Load the data
haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
# There is one epitope file per haplotype

# Public repository with all the data: https://gitlab.com/ivan_ea/epitopes
DATA_URL = 'https://gitlab.com/ivan_ea/epitopes/-/raw/master/'

epitopes_dfs = {}

for h in haplotypes: # Takes like 5 seconds 
  if verb: print('Fetching {}.csv...'.format(h), end=' ')
  epitopes_dfs[h] = pd.read_csv(DATA_URL+h+'.csv')
  if verb: print('It has {} epitopes'.format(len(epitopes_dfs[h])))

if verb:
  print('\n Beginning of the {} epitopes dataframe:'.format(h))
  print(epitopes_dfs[h].head()) 

id1 = epitopes_dfs['MHC_2_DP']['protein_id']

id2 = epitopes_dfs['MHC_2_DQ']['protein_id']

coincidences = id1.isin([id2])


print(coincidences.value_counts())