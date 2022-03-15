# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:06:03 2021

@author: User
"""

from tensorflow.keras.models import model_from_json
import numpy as np
from math import floor

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


# load json and create model
json_file = open('../neuralnetwork/saved_model3/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../neuralnetwork/saved_model3/model.h5")
print("Loaded model from disk")

file1 = open('output_primary.txt', 'r')
file2 = open('output_secondary.txt', 'r')

prot = file1.readlines()
pred = file2.readlines()

inputsA = []
inputsB = []

for i in range(floor(len(prot)/2)):
    num2 = i*2 + 1
    X_prot = change_array(prot[num2])
    X_prot = X_prot[0:768]
    X_pred = change_array(pred[num2])
    X_pred = X_pred[0:768]
    
    inputsA.append(X_prot)
    inputsB.append(X_pred)

XA = np.array(inputsA, dtype='float32')
XB = np.array(inputsB, dtype='float32')

y_new = loaded_model.predict([XA, XB], verbose=1, steps=1)
y_new[y_new <= 0.5] = 0
y_new[y_new > 0.5] = 1

file3 = open('results.txt', 'w')
for i in range(len(y_new)):
    x = np.array_str(y_new[i])
    file3.write(x)
    file3.write('\n')
file3.close()