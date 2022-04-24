#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import time, sys 
from functools import partial, reduce
verb = True # verbose output

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
#  13248713 = Max number of rows (max sliding windows for our proteins) 

if verb: print('\nSliding window for epitopes in haplotype {} \n'.format(h))

WIN_SIZE = 30
output_cols = ['30aa_seq','contains_epitope?']

# keep only the proteins actually used in this haplotype
proteins_df=proteins_df.loc[proteins_df['protein_id'].isin(
  pd.unique(epitopes_df['protein_id']))].copy().reset_index(drop=True)
#keep only proteins with more aminoacids than window size
proteins_df['len'] = list(map(lambda x:len(x), proteins_df['aas']))
proteins_df=proteins_df.loc[proteins_df['len']>WIN_SIZE].copy().reset_index(drop=True)
total_windows = np.sum(proteins_df['len']) - (WIN_SIZE-1)*len(proteins_df)

# auxiliary functions 
def condition_1(window, epitopes_in):
  '''Check if whole epitope inside the window'''
  return ((window[0]<=epitopes_in['start'])&(window[1]>=epitopes_in['end'])).any()

def condition_2(window, epitopes_in):
  '''Check if more than half the window belongs to 1 or more epitopes'''
  overlap_positions = np.zeros(WIN_SIZE,dtype=int)
  overlap_eps = epitopes_in.loc[(window[0] <= epitopes_in['end']) & 
  (window[1] >= epitopes_in['start'])]
  def fu(x,y):
    o_s = max(window[0], x)-window[0]
    o_e = min(window[1], y)-window[0]
    o = np.zeros(WIN_SIZE, dtype=int)
    o[o_s: o_e + 1]=1
    return o
  all_overlaps = list(map(fu, overlap_eps['start'], overlap_eps['end']))
  a = reduce(np.add, all_overlaps, overlap_positions)
  return len(a[a==0]) < int(WIN_SIZE/2+0.5)

def contains_epitope(window, epitopes_inside):
  '''Check if a window of 30aa satisfies the conditions for epitope'''
  if condition_1(window, epitopes_inside): return 1
  elif condition_2(window, epitopes_inside): return 1
  return 0


n_prots=20
results_df = pd.DataFrame(columns=output_cols)
pr_e = 200 # print info every n proteins

print('   Proteins time (s) time (min)  Rows in training set')
f = '{:5}/{:5} {:7.1f} {:9.2f}     {:8} / {:8}'

start_t = time.time()

# SLIDING WINDOW FOR ALL PROTEINS
#for protein in proteins_df.iloc[:n_prots].itertuples():
for protein in proteins_df.itertuples():
  if (protein[0] % pr_e == 0): 
    print(f.format(protein[0],len(proteins_df),time.time()-start_t,
                   (time.time()-start_t)/60, len(results_df), total_windows))
    
  # check epitopes that have that protein as parent_id
  epitopes_in = epitopes_df.loc[epitopes_df['protein_id'] == protein[2]]
  if(len(epitopes_in) <= 0): continue # skip protein if it has no epitopes for this haplotype

  # slide through the windows (paralell version)
  n_windows = len(protein[3]) - (WIN_SIZE - 1)
  all_windows = np.zeros([n_windows, 2],dtype=int)
  all_windows[:,0] = np.arange(n_windows)+1
  all_windows[:,1] = np.arange(n_windows)+WIN_SIZE
  fun_contains = partial(contains_epitope,epitopes_inside=epitopes_in)
  list_conditions = list(map(fun_contains, all_windows))
  list_windows =  list(map(lambda x: protein[3][x[0]-1:x[1]],all_windows))
    
# save results  
  w_df=pd.DataFrame(zip(list_windows,list_conditions),columns=results_df.columns)
  results_df = pd.concat([results_df, w_df], ignore_index=True)

print('Haplotype    Proteins time (s) time (min)  Rows in training set')
print(('{:9} '+f).format(h,protein[0]+1,len(proteins_df),time.time()-start_t,
(time.time()-start_t)/60, len(results_df),total_windows))

# write in csv (romeve duplicates and contradictory later?)
output_name = 'trainig_indep_'+h+'.csv'
results_df.to_csv(output_name, header=True, index=False)
print('Output saved in {}, it has {} rows'.format(output_name,len(results_df)))




