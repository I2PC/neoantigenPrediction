import pandas as pd
import numpy as np
import csv
df1=pd.read_csv('/Users/paolanunez/Documents/TFG_PaolaNuñez/Epitope_SEQUENCES.csv',names=['Uniprot', 'Epitope', 'Begin','End','SEQ'])
x=df1.Uniprot.unique()
#archivo = open("seq_positive1.txt","w")
#file_epi=open('dataset30aa1_0.csv', mode='w')
#csv_file_epi=csv.writer(file_epi,delimiter=',')
for i in x:
    #archivo = open("seq_positive1.txt","a")
    is_epi = df1.loc[:, 'Uniprot'] == i
    df_m= df1.loc[is_epi]
    epi_uni=df_m.Epitope.unique()
    word=''
    for element in epi_uni:
        try:
            if len(element) > len(word):
                word=element
        except:
            pass
    epi=[word]
    for i in epi_uni:
        try: 
            if word.find(i)==-1:
                epi.append(i)
        except:
            pass
    
    interval=[]
    for y in epi:
        try:
            SEQ=df_m.iloc[1,4]
            init_pos=SEQ.find(y)
            len_epi=len(y)
            if init_pos!=-1:
                interval.append([init_pos,init_pos+len_epi-1])
        except:
            pass
    union = []
    for begin,end in sorted(interval):
        if union and union[-1][1] >= begin - 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])
    tex=[]
    tex.append(SEQ)
    try:
        NUM=np.zeros(len(SEQ))
        for begin,end in union:
            tex.append(SEQ[begin:end])
            NUM[begin:end]=1
        
        tex.append(NUM)
        #archivo.write(str(tex)+'\n')
        #archivo.close()
    except:
        pass

    begin=0
    end=30
    try:
        if SEQ.isupper():
            while end<=len(SEQ):
                mini_seq=SEQ[begin:end]
                mini_num=NUM[begin:end]
                non_zero_els = np.count_nonzero(mini_num)
                tag=0
                if non_zero_els>=15:
                    tag=1
                else:
                    for i in epi:
                        if mini_seq.find(i)!=-1:
                            tag=1
                #csv_file_epi.writerow([mini_seq,tag])
                begin+=1
                end+=1
    except:
        pass














# for y in epi:
#         SEQ=df_m.iloc[1,4]
#         try:
#             init_pos=SEQ.find(y)
#             len_epi=len(y)
#             rest=30-len_epi
#             left=rest+5
#             rigth=-5
#             X=SEQ[init_pos-left:init_pos+len_epi+rigth]
#             X=X+'\t 0'
#             print(X)
#             while left!=-5:
#                 left=left-1
#                 rigth=rigth+1
#                 X=SEQ[init_pos-left:init_pos+len_epi+rigth]
#                 if X.find(y)==-1:
#                     X=X+'\t 0'
#                 else:
#                     X=X+'\t 1'
#                 print(X)
#         except:
#             pass