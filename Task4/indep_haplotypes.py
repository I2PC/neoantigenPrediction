#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import sys
verb = True # to print debug info

# 0. Select the haplotype
haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']

h_index = 0 if len(sys.argv) == 1 else int(sys.argv[1]) 
if h_index > 5 or h_index < 0:
  print('ERROR: The argument must be a number between 0 and 5')
  sys.exit(1)

if verb: print('Generating epitope/non-epitope data for haplotype {}: {}'.
  format(h_index,haplotypes[h_index]),end='\n\n')

# 1. Load the data
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
  print(pd.read_csv(DATA_URL+'explain_columns.csv'), end='\n\n')

h = haplotypes[h_index] 
if verb: print('Fetching {}.csv...'.format(h), end=' ')
epitopes_df = pd.read_csv(DATA_URL+h+'.csv')
# remove epitopes without start and end information (only 250 out of 600k)
epitopes_df.dropna(subset=['start'], inplace = True) 
# now columns 'start' and 'end' can be treated as ints 
epitopes_df = epitopes_df.astype({'start': int, 'end': int})
if verb: print('It has {} epitopes'.format(len(epitopes_df)))

if verb:
  print('\n Beginning of the {} epitopes dataframe:'.format(h))
  print(epitopes_df.head()) 

sys.exit(0)

# 2. Once data is loaded we want to create a new dataframe containing the epitope, the aa chain and the protein id
# TAKES A LOT OF TIME 3,5 H 

start_time = time.time()

protein_epitope_df = pd.DataFrame()

for index,protein in proteins_df.iterrows():
  protein_id = protein['protein_id']
  aa_chain = protein['aas']
  #Once we get the protein_id we search for that protein in the haplotypes dfs, and create a new dataframe with the 
  #epitopes found for that protein
  for h in haplotypes:
    epitopes = epitopes_dfs[h].loc[epitopes_dfs[h]['protein_id'] == protein_id]
    for index, epi in epitopes.iterrows():
      entry = {'protein_id': protein_id, 'SEQ': aa_chain, 'epitope':epi['epitope']}
      protein_epitope_df = protein_epitope_df.append(entry, ignore_index=True)

protein_epitope_df.to_csv('epitope_sequence.csv')

if verb:
  print('\n Beginning of the epitopes/protein dataframe:')
  print(protein_epitope_df.head()) 

print("--- %s seconds ---" % (time.time() - start_time))


# 3.
# TODO sliding window and generation of dataset for training (1 per haplotype)
# Output: 6 .csv files with these header and content:
#  30aa_seq,contains_epitope?
#  EJEMPLO_DE_SEQUENCIA_DE_30_AA,0
#  JEMPLO_DE_SEQUENCIA_DE_30_AAs,1
#  etc...



