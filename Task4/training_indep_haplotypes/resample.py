#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import zipfile
from imblearn.under_sampling import RandomUnderSampler

haplotypes = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
for h in haplotypes:

	#Extracting the csv from the zip file
	data_path = h + '.zip'
	with zipfile.ZipFile(data_path,'r') as zip_ref:
		zip_ref.extractall()

	#Reading the csv into a dataframe
	file = h + '.csv'
	df = pd.read_csv(file)

	print('Haplotype {}'.format(h,))
	print(df['contains_epitope?'].value_counts())

	#Imblearn RandomUnderSampler won´t take samples with just one feature, we rename contains_epitope?
	#to cheat imblearn, then we will drop that column so we get a df with the same columns
	df2 = df.rename({'contains_epitope?': 'label'}, axis = 1)
	labels = df['contains_epitope?']
	under_sampler = RandomUnderSampler(random_state=42)
	win_resamp, labels_resamp = under_sampler.fit_resample(df2, labels)

	#Create the resampled df and drop the 'new' column
	resampled_df = pd.concat([win_resamp, labels_resamp], axis = 1)
	resampled_df = resampled_df.drop('label', axis = 1)

	#Print info so we know everything is going well
	print('Resampled Haplotype {}'.format(h,))
	print(resampled_df['contains_epitope?'].value_counts())
	print('Final length: {}'.format(len(resampled_df)))

	#Write a csv per haplotype with the resampled dataframe
	compression_opts = dict(method='zip',
                        archive_name=h+'_balanced.csv')  

	filename = h + '_balanced.zip'

	resampled_df.to_csv(filename,index=False,compression=compression_opts)
	print('Output saved in {}'.format(filename))

