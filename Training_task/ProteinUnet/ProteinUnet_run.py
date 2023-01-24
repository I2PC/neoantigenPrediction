# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:20:46 2021

@author: User
"""


import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pandas as pd
from optparse import OptionParser

# PROTEIN UNET
models_folder = "/home/student/dataLocal/workingdir/proyectomaster/neoantigenPrediction/Training_task/ProteinUnet/models" #folder with the models 

# define problem properties
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
ANGLE_NAMES_LIST = ["PHI", "PSI", "THETA", "TAU"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
#FASTA_RESIDUE_LIST = ["a", "d", "n", "r", "c", "e", "q", "g", "h", "i", "l", "k", "m", "f", "p", "s", "t", "w", "y", "v"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

EXP_NAMES_LIST = ["ASA", "CN", "HSE_A_U", "HSE_A_D"]
EXP_MAXS_LIST = [330.0, 131.0, 76.0, 79.0]  # Maximums from our dataset
UPPER_LENGTH_LIMIT = 1024

# Load models
path_model= os.path.join(models_folder, "unet_c_ensemble")
ensemble_c = load_model(path_model)
# ensemble_r = load_model(os.path.join(models_folder, "unet_r_ensemble"))
def divide_sequence(sequence):
  divisions = list()
  while len(sequence) > 1024:
    seq = sequence[0:1024]
    divisions.append(seq)
    sequence = sequence[1024:len(sequence)]
  divisions.append(sequence)
  return divisions


def read_input(input):
    protein_names = []
    sequences = []
    protein_names.append(input[0])
    sequences.append(input[1])
    return protein_names, sequences

def clean_seq(mat):
    prot_clean=[]
    for x in mat:
        if "/" not in x:
            prot_clean.append()
    return prot_clean
def fill_array_with_value(array: np.array, length_limit: int, value):
    array_length = len(array)

    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


def save_predictions(resnames, pred_c):
    sequence_length = len(resnames)
    def get_ss(one_hot):
        return [SS_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    list_predictions = get_ss(pred_c[0][0])[:sequence_length]
    prediction = ''
    for r in range(len(list_predictions)):
      prediction = prediction + list_predictions[r]
    file2.write(prediction)
    file2.write('\n')

def split_seq(inpath, outpath):
    file1= open(outpath, 'w')

    with open(inpath, 'r') as f:
        for line in f:
            line1=line[:-1]
            for i in range(len(line1)):
                file1.write(line1[i])
                file1.write(" ")
            file1.write('\n')
    file1.close()

def main():

    protein_names, residue_lists = read_input(input)

    for protein_name, resnames in zip(protein_names, residue_lists):
        if len(resnames) > UPPER_LENGTH_LIMIT:
            print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues!")
            continue
        residue_valid=""
        for residue in resnames:
            if residue in RESIDUE_DICT:
                residue_valid=residue_valid + residue
                
        sequence = to_categorical([RESIDUE_DICT[residue] for residue in residue_valid], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)

        pred_c = ensemble_c.predict(np.array([sequence]))
        # pred_r = ensemble_r.predict(np.array([sequence]))
        
        save_predictions(resnames, pred_c)

parser=OptionParser()

parser.add_option("-i", "--input", dest="input", help="input file", metavar="FILE")

options, args= parser.parse_args()
INpath= options.input

file = open(INpath, 'r') #name of the text file with the protein sequence
file2 = open('secondary_structure.txt', 'w') #name of the text file for secondary structure

data = file.read().splitlines()

# WE CALL THE MAIN FUNCTION HERE


count = 0

for i in range(len(data)):
    input=[count, data[i]]
    if __name__=='__main__':
        main()
    count +=1

file.close()
file2.close()

split_seq('secondary_structure.txt', 'secondary_structure_spaces')

print('Done secondary struct prediction')
