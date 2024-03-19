#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from optparse import OptionParser
import numpy as np
import pandas as pd

"""
Uses keras integration in tensorflow
"""

def clean_seq(mat): #Removes the / from the sequences
    prot_clean=[]
    for x in mat:
        if "/" not in x:
            prot_clean.append(x)
    return prot_clean


#BLOSUM function

def BLOSUMAUG(mat):
    strings = []
    probs = []
    stringsin = []
    with open("/services/tools/blosum62.txt") as matrix_file:
        matrix = matrix_file.read()
    lines = matrix.strip().split('\n')
    header = lines.pop(0)
    columns = header.split()
    matrix = {}
    for row in lines:
        entries = row.split()
        row_name = entries.pop(0)
        matrix[row_name] = {}
        if len(entries) != len(columns):
            raise Exception('Improper entry number in row')
        for column_name in columns:
            matrix[row_name][column_name] = int(entries.pop(0))
    for count in range(0,mat.shape[0]):
        stringin = mat[count]
        strings.append(stringin)
        probs.append(1)
        stringsin.append(stringin)
        for num in range(0,len(stringin)):
            string = list(stringin)
            char=string[num]
            #Retrieve the aa with highest similarity
            listaas = dict(map(reversed, matrix.get(char).items()))
            listprob = np.array(list(listaas.keys()))
            listprob[::-1].sort()
            nn = int(listprob[1])
            chrf = listaas.get(nn)
            prob = (np.exp(nn))/(sum(np.exp(listprob[1:])))
            string[num] = chrf
            string = "".join(string)
            strings.append(string)
            probs.append(prob)
            stringsin.append(stringin)
    augmented = pd.DataFrame({'original sequence':stringsin,'peptide after BLOSUM':strings,'probability BLOSUM':probs}).drop_duplicates(subset='peptide after BLOSUM').reset_index(drop=True)
    return augmented


def main():
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input",
                  help="Input file", metavar="FILE")
    
    # Modified in order to have 2 outputs

    #parser.add_option("-o", "--output", dest="out",
    #          help="Output file", metavar="FILE")

    parser.add_option("-o", "--output1", dest="output1",
              help="Output file 1", metavar="FILE")
    parser.add_option("-p", "--output2", dest="output2",
              help="Output file 2", metavar="FILE")

    options, args = parser.parse_args()
    INpath = options.input
    #OUTpath = options.out
    output_file1 = options.output1
    output_file2 = options.output2

    data=pd.read_csv(INpath,delimiter=",")
    sequence= clean_seq(data['seq'].values) 
    sequence_new= data.loc[data['seq'].isin(sequence)]
    blosexp = BLOSUMAUG(sequence_new['seq'].values)
    peptides= pd.DataFrame(columns=["seq","seq_blosum",'seq_extended', "FPKM", "Gene symbol"])
    current_original=""
    cc=0
    for index, row in blosexp.iterrows():
        if row['original sequence']!=current_original:
            cc=cc+1
            peptides.loc[cc]= [row['original sequence'], row['peptide after BLOSUM'],sequence_new.loc[sequence_new['seq']==row['original sequence'], 'seq_extended'].iloc[0], sequence_new.loc[sequence_new['seq']==row['original sequence'], 'FPKM'].iloc[0], sequence_new.loc[sequence_new['seq']==row['original sequence'], 'gene_symbol'].iloc[0]]  
        else:
            peptides.loc[cc, 'seq_blosum']= row['seq_blosum']
            cc=cc+1
    # peptides.to_csv(OUTpath, index=False)
    #peptides_aug= peptides['seq_blosum']
    #peptides_aug.to_csv(OUTpath, index=False)
            
    peptides.to_csv(output_file1, index=False)
    peptides_aug= peptides['seq_blosum']
    peptides_aug.to_csv(output_file2, index=False)
    print('Done postprocessing')

if __name__ == '__main__':
    main()
