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

if verb: print('\nGenerating epitope/non-epitope data for haplotype {}: {}'.
  format(h_index,haplotypes[h_index]),end='\n\n')

# 1. Load the data

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


# 2. Sliding window and generation of dataset for training (1 per haplotype)
# Output: one csv files with these header and content:
#  30aa_seq,contains_epitope?
#  EJEMPLO_DE_SEQUENCIA_DE_30_AA,0
#  JEMPLO_DE_SEQUENCIA_DE_30_AAs,1
#  etc...
#  13248713 = Max number of rows (max windows for our proteins) 

if verb: print('\nSliding window for epitopes in haplotype {} \n'.format(h))

results_df = pd.DataFrame(columns=['30aa_seq','contains_epitope?'])
output_name = 'trainig_indep_'+h+'.csv'
pr_e = 20 # print info every n proteins
WIN_SIZE=30

def condition_1(window,epitopes_inside):
  '''Check if whole epitope inside the window'''
  for e in epitopes_inside.itertuples():
    if (window[0] <= e[2]) and (window[1] >= e[3]):
      return True
  return False

def condition_2(window, epitopes_inside):
  '''Check if more than half the window belongs to 1 or more epitopes'''
  overlaps = np.zeros(WIN_SIZE,dtype=int)
  for e in epitopes_inside.itertuples():
    if (window[0] <= e[3]) and (window[1] >= e[2]): # this checks if overlap
      overlap_start = max(window[0],e[2])-window[0]
      overlap_end = min(window[1],e[3])-window[0]
      overlaps[overlap_start: overlap_end+1] = 1     
  return np.sum(overlaps) > int(WIN_SIZE/2 +0.5) 

def contains_epitope(window, epitopes_inside):
  '''Check if a window of 30aa satisfies the conditions for epitope'''
  if condition_1(window, epitopes_inside): return 1
  elif condition_2(window, epitopes_inside): return 1
  return 0

# Association between protein and corresponding epitopes
current_prot = ''
assoc_protein_epitopes = {}
for e in epitopes_df.itertuples():
  if e[5] != current_prot:
    current_prot=e[5]
    assoc_protein_epitopes[current_prot]=[]
  assoc_protein_epitopes[current_prot].append(e[0])
    
if verb: print('   Proteins time (s) time (min)  Rows in training set')
f = '{:5}/{:5} {:8.1f} {:10.2f}  {}'
start_t = time.time()

# for all proteins (remove iloc after testing)
for protein in proteins_df.iloc[0:100].itertuples():
  if (protein[0]%5 == 0): 
    print(f.format(protein[0],len(assoc_protein_epitopes),time.time()-start_t,
                   (time.time()-start_t)/60, len(results_df)))
    
  # check epitopes that have that protein as parent_id
  #epitopes_inside = epitopes_df.loc[epitopes_df['protein_id'] == protein[2]] <-slow
  epitopes_inside = epitopes_df.iloc[assoc_protein_epitopes[protein[2]]] # <- fast?
  if(len(epitopes_inside) <= 0):
    continue # skip protein if it has no epitopes for this haplotype

  # slide through the windows
  n_windows = len(protein[3]) - (WIN_SIZE - 1)
  list_windows, list_conditions = [],[]
  
  for i in range(n_windows):
    window = [i+1,i+WIN_SIZE]
    condition = contains_epitope(window, epitopes_inside)
    list_windows.append(protein[3][i:i+WIN_SIZE])
    list_conditions.append(condition)
  
# save results  
  w_df=pd.DataFrame(zip(list_windows,list_conditions),columns=results_df.columns)
  results_df = pd.concat([results_df, w_df], ignore_index=True)

print(f.format(protein[0],len(proteins_df),time.time()-start_t,
(time.time()-start_t)/60, len(results_df)))

# write in csv (romeve duplicates and contradictory later?)
results_df.to_csv(output_name, header=True, index=False)
print('Output saved in {}, it has {} rows'.format(output_name,len(results_df)))

sys.exit(0)



