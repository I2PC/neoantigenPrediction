import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

file_epi=open('/Users/paolanunez/Downloads/dataset30aa_16centered.csv', mode='r')
file_epi_tab=open('dataset30_centered_sara_pretrained.csv', mode='w')
csv_file_epi=csv.writer(file_epi_tab,delimiter=',')
index=0
for item in file_epi:
    seq=item.split(',')[0]
    tag=int(item.split(',')[1].split('\n')[0])
    al=''
    seq1=(" ".join(seq))
    csv_file_epi.writerow([seq1])
    #csv_file_epi.writerow([index,tag,al,seq1])
    index=index+1


#file_epi_tab=pd.read_csv('dataset30_centered_sara.csv', delimiter=',',names=['Index','Tag','alpha','SEQ'])

#output=open('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized/train.tsv', 'w')
#output2=open('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized/test.tsv', 'w')
#output3=open('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized/dev.tsv', 'w')


#A=file_epi_tab[file_epi_tab['Tag']==1]
#x=A.shape[0]

#B=file_epi_tab[file_epi_tab['Tag']==0]
#B=B.iloc[0:x, :]
#x=B.shape[0]

#A_train, A_test,B_train,B_test= train_test_split(A,B, test_size=0.1, random_state=1)
#A_train, A_val,B_train,B_val= train_test_split(A_train,B_train, test_size=0.2222223, random_state=1)
# 0.2222222*0.9=0.2

#A_train.to_csv ('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/train.tsv',sep = '\t',mode='a',index = False, header=False)
#B_train.to_csv('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/train.tsv',sep = '\t',mode='a', index = False, header=False)
#A_val.to_csv ('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/dev.tsv',sep = '\t',mode='a',index = False, header=False)
#B_val.to_csv('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/dev.tsv',sep = '\t',mode='a', index = False, header=False)
#A_test.to_csv ('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/test.tsv',sep = '\t',mode='a',index = False, header=False)
#B_test.to_csv('/Users/paolanunez/Documents/TFG_PaolaNuñez/data_30aa_centralized2/test.tsv',sep = '\t',mode='a', index = False, header=False)  

