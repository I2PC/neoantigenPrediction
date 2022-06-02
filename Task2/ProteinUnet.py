# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:20:46 2021

@author: User
"""


import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
#import pandas as pd

# PROTEIN UNET
models_folder = "./data/models_good" #folder with the models

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


def save_predictions(resnames, pred_c,real_length): #save predictions
    sequence_length = len(resnames)
    def get_ss(one_hot):
        return [SS_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    list_predictions = get_ss(pred_c[0][0])[:sequence_length]
    prediction = ''
    for r in range(len(list_predictions)):
      prediction = prediction + list_predictions[r]
    file2.write(prediction[0:real_length]) # reduce the length of the prediction to not make a prediction of 0s.
    file2.write('\n')
    
def fill_array_with_value(array: np.array, length_limit: int, value): # fill the sequence with 0 until reach a 1024 sequence length
    array_length = len(array)

    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array
    
def creation_dictionary():  # create a dictionary with the number of the protein as a key and then the chain of aminoacids 
    Data= open('Database.txt','r')
    Database2=[]
    Database3=[]
    list1=[]
    for e in Data:
        for i in e:
            if i!='\n':
                if i!='-':
                    list1.append(i)                
                else:
                    if len(list1)>0:
                        Database3.append(list1)
                        list1=[]
    Database3.append(list1)
    dictionary={}
    for n,i in enumerate(Database3):
        list1=""
        for j in i:
            list1=list1 + j
        dictionary[n]=list1
        
    return dictionary


def save_predictions2(resnames, pred_c): # Save the prediction of sequences longer than 1024
    sequence_length = len(resnames)
    def get_ss(one_hot):
        return [SS_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    list_predictions = get_ss(pred_c[0][0])[:sequence_length]
    prediction = ''
    for r in range(len(list_predictions)):
      prediction = prediction + list_predictions[r]
    return prediction



def main():  

    dictionary= creation_dictionary()    
    #e = 23232
    for e in range(len(dictionary)): # iterate over the keys of the dictionary
        residue_valid=""
        for residue in dictionary[e]:    
            if residue in RESIDUE_DICT: # chech if teh residue is in de dictionary or not
                residue_valid=residue_valid + residue
        if len(residue_valid)>1024: # In the case that the chain is longer than 1024 residues
            sequence=residue_valid
            l_prediction= [0]*len(residue_valid)
            count=0
            while len(sequence) > 1024:# predict he sequences longer than 1024
                seq = sequence[0:1024] # # divide into windows of length 1024
                sequence = sequence[1024:len(sequence)]
                sequence2 = to_categorical([RESIDUE_DICT[residue] for residue in seq], num_classes=NB_RESIDUES) # create matrix
                pred_c = ensemble_c.predict(np.array([sequence2])) # predict
                prediction=save_predictions2(seq, pred_c)            
                l_prediction[(count+0):(count+1024)] =prediction # save in a list the predictions
                count+=1024
                            
            sequence2 = to_categorical([RESIDUE_DICT[residue] for residue in residue_valid[-1024:]], num_classes=NB_RESIDUES) # now predict the last window of the protein which is lower than 1024,
            pred_c = ensemble_c.predict(np.array([sequence2]))# predict the last window #  in this case we took as the last window the final 1024 aminoacids so there is an overlape with the penultimate window
            prediction=save_predictions2(residue_valid[-1024:], pred_c) # save predictions
            l_prediction[-1024:] = prediction# save the predictions in a list and for the aminoacids in the overlapped part we select the second prediction
        
            file2.write(''.join(l_prediction)) # write the predictions in the prediction file
            file2.write('\n')

        else: # Make the procedure with the sequences lower than 1024
            sequence = to_categorical([RESIDUE_DICT[residue] for residue in residue_valid], num_classes=NB_RESIDUES)
            sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0) # fill the array with 0 until reach a length of 1024
            pred_c = ensemble_c.predict(np.array([sequence]))
            save_predictions(sequence, pred_c,len(residue_valid))


f = open ('uniprot-proteome_UP000005640 .fasta','r') #open the file with the human 80.000 proteins
f2 = open ('Database.txt','w')# create a modifies database with the proteins separated by -
count=0
for e in f:
    if e.startswith('>'):
        f2.write('-')
    else:
        f2.write(e)
f.close()
f2.close()

# OPEN FILES
file2 = open('prediction.txt', 'w') #name of the text file for secondary structure

# WE CALL THE MAIN FUNCTION HERE

if __name__=='__main__':
    main()

file2.close()