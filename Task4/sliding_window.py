import pandas as pd
import numpy as np

#### SE CORRESPONDE CON EPITOPES1_0.PY. SLIDING WINDOW

df1=pd.read_csv('epitope_sequence.csv',names=['SEQ', 'epitope','protein_id']) #Leo el fichero que tiene epitopos y secuencia entera de prot
print('Data loaded')
x=df1.protein_id.unique()#Cojo solo el identificador de cada proteína una vez
final_df = pd.DataFrame()
for i in x: #itero por cada proteína
    is_epi = df1.loc[:, 'protein_id'] == i #miro para todas las filas de mi database las entradas que tienen el mismo identificador de proteina
    df_m= df1.loc[is_epi] #hago una nueva df con las entries de la misma proteina
    epi_uni=df_m.epitope.unique() #cojo las secuencias de epítopos diferentes
    word=''
    for element in epi_uni:  #cojo el epítopo más largo
        if len(element) > len(word):
            word=element
    epi=[word]
    for i in epi_uni:
        if word.find(i)==-1:  #miro si el resto de epítopos descartados forman tienen la misma secuencia que el elegido pero más corta. Si no es la misma añado ese epítopo también
            epi.append(i)
    #print(epi)
    interval=[]
    for y in epi:
        SEQ= df_m.iloc[0,0] #cojo la secuencia de proteínas entera
        init_pos=SEQ.find(y) #encuentro dónde empieza el epítopo
        len_epi=len(y) #encuentro la longitud del epítopo
        if init_pos!=-1:
                interval.append([init_pos,init_pos+len_epi-1]) #Tengo el intervalo de posiciones de inicio y fin del epítopo
    union = []
    for begin,end in sorted(interval):  #no tengo claro lo que es union
        if union and union[-1][1] >= begin - 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])
    #print(union)  
    tex=[]
    tex.append(SEQ)
    try:   #tengo una lista de números = a la seq. Tendré uno en los puntos en los aa en que sea un epítopo y cero en los que no
        NUM=np.zeros(len(SEQ))
        for begin,end in union:
            tex.append(SEQ[begin:end])
            NUM[begin:end]=1
        
        tex.append(NUM)
    except:
        pass
    print(tex)
    begin=0
    end=30
    try:
        if SEQ.isupper(): #check if sequence is upper case
            while end<=len(SEQ): #compruebo que no intente coger aa una vez terminada toda la secuencia
                mini_seq=SEQ[begin:end] #cojo la parte de la secuencia en la que estoy
                mini_num=NUM[begin:end] #cojo el vector de 0 y 1
                non_zero_els = np.count_nonzero(mini_num) #cuento cuántos elementos son diferentes a cero, es decir, parte del epítopo
                tag=0
                if non_zero_els>=15: #si hay más de 15 aa del epítopo (aunque no esté entero lo considero). El 50% de la secuencia es epítopo
                    tag=1
                else:
                    for i in epi:
                        if mini_seq.find(i)!=-1: #si el epítopo está completamente contenido en mi secuencia
                            tag=1

                row = {'aas':mini_seq,'y':tag}
                print(row)
                final_df = final_df.append(row,ignore_index=True)

                begin+=1
                end+=1
    except:
        pass

final_df.to_csv('30aas_train.csv')

       
      