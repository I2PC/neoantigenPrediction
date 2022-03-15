# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 17:52:36 2021

@author: User
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
file1 = open('results6.txt', 'r')

data = file1.readlines() 

predicted = []
for i in range(703, 2234):
    a = int(data[i][1])
    predicted.append(a)

n = 0
x = [0] * 50
for i in range(len(predicted)-1):
    if predicted[i] == 1:
        n+=1
    
    if n < 5 and predicted[i+1] == 0:
        x[n] += 1
        n = 0
    elif n>=5 and predicted[i+1] == 0:
        x[n] += 1
        n = 0


# fixed bin size
bins = np.arange(-100, 100, 1) # fixed bin size

plt.xlim([0,100])

plt.hist(x, bins=bins, alpha=0.5)
plt.title('Distribution of consecutive positive subsequences size')
plt.xlabel('size of set (# of subsequences)')
plt.ylabel('# of sets')

plt.show()

correct = [0] * 1531

for i in range(85,106):
    correct[i] = 1

for i in range(1052, 1074):
    correct[i] = 1

a = confusion_matrix(correct, predicted)


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(1531):
    if correct[i] == 1 and predicted[i] == 1:
        TP += 1
    elif correct[i] == 0 and predicted[i] == 0:
        TN += 1
    elif correct[i] == 1 and predicted[i] == 0:
        FN += 1
    elif correct[i] == 0 and predicted[i] == 1:
        FP += 1   


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

a = confusion_matrix(correct, predicted)

df_cm = pd.DataFrame(a, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues") # font size
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()