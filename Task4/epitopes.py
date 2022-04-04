#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']

# data still work in progress:
# TODO
# class 2 epitopes missing
# not all proteins in file

data_url = 'https://gitlab.com/ivan_ea/epitopes/-/raw/master/'

proteins_df = pd.read_csv(data_url+'proteins.csv')
print(proteins_df.head()) # debug info


epitopes_dfs = {}

for h in haplotypes[0:3]:
  epitopes_dfs[h] = pd.read_csv(data_url+h+'.csv')
  
print(epitopes_dfs['MHC_1_A'].head()) #debug info

