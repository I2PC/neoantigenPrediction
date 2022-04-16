#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
verb = True # to print debug info

#1. Load the data
haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
# There is one epitope file per haplotype

# Public repository with all the data: https://gitlab.com/ivan_ea/epitopes
DATA_URL = 'https://gitlab.com/ivan_ea/epitopes/-/raw/master/'

if verb: print('Fetching proteins file...')
proteins_df = pd.read_csv(DATA_URL+'proteins.csv')

if verb:
  print('Beginning of the proteins dataframe:')
  print(proteins_df.head()) 
  print('It has {} unique proteins'.format(len(proteins_df)))
if verb:
  print('\n Explanation of the columns of the epitope files:')
  print(pd.read_csv(DATA_URL+'explain_columns.csv'), end='\n')

epitopes_dfs = {}

for h in haplotypes: # Takes like 5 seconds 
  if verb: print('Fetching {}.csv...'.format(h), end=' ')
  epitopes_dfs[h] = pd.read_csv(DATA_URL+h+'.csv')
  if verb: print('It has {} epitopes'.format(len(epitopes_dfs[h])))

if verb:
  print('\n Beginning of the {} epitopes dataframe:'.format(h))
  print(epitopes_dfs[h].head()) 


# 2. Once data is loaded (proteins_df & epitopes_dfs) we now search what epitopes are contained into each protein
# by searching the protein_id in each epitope_df

protein_epitope_dfs = {}
for index in range(len(proteins_df)):
  protein_id = proteins_df.loc[index,'protein_id']
  #Once we get the protein_id we search for that protein in the haplotypes dfs, and create a new dataframe with the 
  #epitopes found for that protein
  #print('Protein {}'.format(protein_id))
  for h in haplotypes:
    protein_epitope_dfs[h] = epitopes_dfs[h].loc[epitopes_dfs[h]['protein_id'] == protein_id]
    #print('Contains {} possible epitopes for haplotype {}'.format(len(protein_epitope_dfs[h]), h))
if verb:
  print('\n Beginning of the {} epitopes dataframe:'.format(h))
  print(protein_epitopes_dfs[h].head()) 



# 3.
# TODO sliding window and generation of dataset for training (1 per haplotype)
# Output: 6 .csv files with these header and content:
#  30aa_seq,contains_epitope?
#  EJEMPLO_DE_SEQUENCIA_DE_30_AA,0
#  JEMPLO_DE_SEQUENCIA_DE_30_AAs,1
#  etc...



