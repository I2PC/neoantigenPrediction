#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import zipfile

haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
for h in haplotypes:

	#Reading the csv and loading into a dataframe
	data_path = 'training_indep_' + h + '_full.zip'
	with zipfile.ZipFile(data_path,'r') as zip_ref:
		zip_ref.extractall()

	file = 'training_indep_' + h + '_full.csv'
	df = pd.read_csv(file)
	df = df.drop_duplicates()
	sequences = df['30aa_seq'].unique()

	#Duplicates, boolean series True when the sequence is duplicated, false if non duplicated
	duplicates = df.duplicated(subset='30aa_seq',keep = False)
	seq_dupl = df[duplicates]
	seq_non_dupl = df[duplicates != True]

	#Remove from the duplicated ones, the chains labeled as 0
	positives = seq_dupl[seq_dupl['contains_epitope?'] != 0]

	#Merge both df, the one with non duplicated sequences and the one containing the duplicated ones labeled as 1
	#the size obtained should be the same as sequences
	final_df = seq_non_dupl.append(positives)

	print('In haplotype {}, length {}, unique sequences {}'.format(h,len(df),len(sequences)))
	print('Duplicated sequences {}, after dropping 0s {}'.format(len(seq_dupl), len(positives)))
	print('Final length: {}'.format(len(final_df)))

	#Write a csv per haplotype with the final dataframe
	compression_opts = dict(method='zip',
                        archive_name=h+'.csv')  

	filename = h + '.zip'

	final_df.to_csv(filename,index=False,compression=compression_opts)
	print('Output saved in {}'.format(filename))
