# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 09:50:02 2021

@author: User
"""

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from random import randint
import random
from math import floor
import matplotlib.pyplot as plt
import pandas as pd

def change_array(string):
  A = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.']
  B = []
  value = ''
  for j in range(0, len(string), 1):
    if string[j] in A:
      value = value + string[j]
    else:
      if len(value)>0:
        value1 = float(value)
        B.append(value1)
      value = ''
  array = np.array(B)
  return array

        #choose file number randomly
epoch_file = randint(0, 120)
name_file1 = 'output_p_' + str(epoch_file) + '.txt'
name_file2 = 'output_s_' + str(epoch_file) + '.txt'   
file1 = open(name_file1, 'r')

file2 = open(name_file2, 'r')

data1 = file1.readlines()

data2 = file2.readlines()

        
#get corresponding labels
label_file = floor(epoch_file/10)
name_file3 = 'labels_positivas_' + str(label_file)  + '.txt'
name_file4 = 'labels_negativas_' + str(label_file)  + '.txt'
file3 = open(name_file3, 'r')
file4 = open(name_file4, 'r')
        
labels_positivas = file3.readlines()

labels_negativas = file4.readlines()

        
data_pos_train = list()
data_pos_val = list()
data_neg_train = list()
data_neg_val = list()    
train = round(0.9*len(labels_positivas))
train2 = round(0.9*len(labels_negativas))
 #val = len(data1) - train
data_pos = list()
data_neg = list()
        
for i in labels_positivas:
    if int(i)>= epoch_file*100000 and int(i)<(epoch_file+1)*100000:
        num = int(i) - epoch_file*100000
        num2 = num*2 + 1
        seq1 = change_array(data1[num2])
        seq2 = change_array(data2[num2])
        seqs = np.concatenate((seq1, seq2), axis=None)
        data_pos.append(seqs)

A = random.sample(range(0, len(data_pos)-1), round(0.1*len(data_pos)))

for i in range(0, len(data_pos)):
    if i in A:
        data_pos_val.append(data_pos[i])
    else:
        data_pos_train.append(data_pos[i])
    

for i in labels_negativas:
            
    if int(i)>= epoch_file*100000 and int(i)<(epoch_file+1)*100000:
        num = int(i) - epoch_file*100000
        num2 = num*2 + 1
        seq1 = change_array(data1[num2])
        seq2 = change_array(data2[num2])
        seqs = np.concatenate((seq1, seq2), axis=None)
        data_neg.append(seqs)
A = random.sample(range(0, len(data_neg)-1), round(0.1*len(data_neg)))

for i in range(0, len(data_neg)):
    if i in A:
        data_neg_val.append(data_neg[i])
    else:
        data_neg_train.append(data_neg[i])
 

num_pos_train = len(data_pos_train)
num_pos_val = len(data_pos_val)
num_neg_train = len(data_neg_train)
num_neg_val = len(data_neg_val)
num_rows = num_pos_train + num_neg_train
num_rows2 = num_pos_val + num_neg_val






def generate_arrays(batchsize): #coger la mitad postiivas la mitad negativas, aleatoriamente
    batchcount = 0
    inputsA = []
    inputsB = []
    targets = []

    while True:
        pos = randint(0, num_pos_train-1) #numero de postivas
        neg = randint(0, num_neg_train-1) #numero de negativas

        x1_pos = data_pos_train[pos]
        xA_pos = x1_pos[0:768]
        xB_pos = x1_pos[768:1536]
        y1 = 1

        x2_neg = data_neg_train[neg]
        xA_neg = x2_neg[0:768]
        xB_neg = x2_neg[768:1536]
        y2 = 0

        inputsA.append(xA_pos)
        inputsB.append(xB_pos)
        inputsA.append(xA_neg)
        inputsB.append(xB_neg)
        targets.append(y1)
        targets.append(y2)
        batchcount += 1
        if batchcount > (batchsize/2-1):
          XA = np.array(inputsA, dtype='float32')
          XB = np.array(inputsB, dtype='float32')
          y = np.array(targets, dtype='float32')
          yield ([XA, XB], y)
          inputsA = []
          inputsB = []
          targets = []
          batchcount = 0
          
def generate_validation(batchsize): #coger la mitad postiivas la mitad negativas, aleatoriamente
    batchcount = 0
    inputsA = []
    inputsB = []
    targets = []
    while True:

        #random number for positive and negative 

        pos = randint(0, num_pos_val-1) #numero de postivas
        neg = randint(0, num_neg_val-1) #numero de negativas

        x1_pos = data_pos_val[pos]
        xA_pos = x1_pos[0:768]
        xB_pos = x1_pos[768:1536]
        y1 = 1

        x2_neg = data_neg_val[neg] 
        xA_neg = x2_neg[0:768]
        xB_neg = x2_neg[768:1536]
        y2 = 0

        inputsA.append(xA_pos)
        inputsB.append(xB_pos)
        inputsA.append(xA_neg)
        inputsB.append(xB_neg)
        targets.append(y1)
        targets.append(y2)
        batchcount += 1
        if batchcount > (batchsize/2-1):
          XA = np.array(inputsA, dtype='float32')
          XB = np.array(inputsB, dtype='float32')
          y = np.array(targets, dtype='float32')
          yield ([XA, XB], y)
          inputsA = []
          inputsB = []
          targets = []
          batchcount = 0
def get_model():
    # define two sets of inputs
    inputA = Input(shape=(768,))
    inputB = Input(shape=(768,))

    # the first branch operates on the first input
    x = Dense(128, activation="relu")(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Model(inputs=inputA, outputs=x)

    # the second branch opreates on the second input
    y = Dense(128, activation="relu")(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Dense(64, activation="relu")(y)
    y = Dropout(0.3)(y)
    y = Dense(32, activation="relu")(y)
    y = Dropout(0.3)(y)
    y = Model(inputs=inputB, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(32, activation="relu")(combined)
    z = Dropout(0.3)(z)
    z = Dense(32, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(16, activation="relu")(z)
    z = BatchNormalization()(z)
    z = Dense(16, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(8, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(1, activation="sigmoid")(z)
# our model will accept the inputs of the two branches and
# then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)
    opt = Adam(learning_rate=3e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0 and epoch!=0:
            
        
            #choose file number randomly
            epoch_file = randint(0, 79)
            name_file1 = 'output_p_' + str(epoch_file) + '.txt'
            name_file2 = 'output_s_' + str(epoch_file) + '.txt'       
        
            file1 = open(name_file1, 'r')
            
            file2 = open(name_file2, 'r')
            
            data1 = file1.readlines()
            
            data2 = file2.readlines()
            
        
            #get corresponding labels
            label_file = floor(epoch_file/10)
            name_file3 = 'labels_positivas_' + str(label_file)  + '.txt'
            name_file4 = 'labels_negativas_' + str(label_file)  + '.txt'
            file3 = open(name_file3, 'r')
            file4 = open(name_file4, 'r')
        
            labels_positivas = file3.readlines()
            
            labels_negativas = file4.readlines()
            
        
            data_pos_train = list()
            data_pos_val = list()
            data_neg_train = list()
            data_neg_val = list()    
            #val = len(data1) - train
            data_pos = list()
            data_neg = list()
        
            for i in labels_positivas:
                if int(i)>= epoch_file*100000 and int(i)<(epoch_file+1)*100000:
                    num = int(i) - epoch_file*100000
                    num2 = num*2 + 1
                    seq1 = change_array(data1[num2])
                    seq2 = change_array(data2[num2])
                    seqs = np.concatenate((seq1, seq2), axis=None)
                    data_pos.append(seqs)
        
            A = random.sample(range(0, len(data_pos)-1), round(0.1*len(data_pos)))

            for i in range(0, len(data_pos)):
                if i in A:
                    data_pos_val.append(data_pos[i])
                else:
                    data_pos_train.append(data_pos[i])       
            
            for i in labels_negativas:
                if int(i)>= epoch_file*100000 and int(i)<(epoch_file+1)*100000:
                    num = int(i) - epoch_file*100000
                    num2 = num*2 + 1
                    seq1 = change_array(data1[num2])
                    seq2 = change_array(data2[num2])
                    seqs = np.concatenate((seq1, seq2), axis=None)
                    data_neg.append(seqs)
            A = random.sample(range(0, len(data_neg)-1), round(0.1*len(data_neg)))

            for i in range(0, len(data_neg)):
                if i in A:
                    data_neg_val.append(data_neg[i])
                else:
                    data_neg_train.append(data_neg[i])  
 
            
            num_pos_train = len(data_pos_train)
            num_pos_val = len(data_pos_val)
            num_neg_train = len(data_neg_train)
            num_neg_val = len(data_neg_val)
            num_rows = num_pos_train + num_neg_train
            num_rows2 = num_pos_val + num_neg_val
            return data_neg_val, data_neg_train, data_pos_train, data_pos_val, num_rows2, num_rows, num_neg_train, num_neg_val, num_pos_train, num_pos_val 

batch_size = 32     
model = get_model()
print(model.summary())

history = model.fit(generate_arrays(batch_size),  epochs=160, steps_per_epoch=num_rows / batch_size, batch_size=batch_size, validation_data = generate_validation(batch_size), validation_steps = num_rows2/batch_size, verbose=1, callbacks=[MyCustomCallback()])        

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'saved_model4/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# serialize model to JSON
model_json = model.to_json()
with open("saved_model4/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("saved_model4/model.h5")
print("Saved model to disk")

#plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model16.png')