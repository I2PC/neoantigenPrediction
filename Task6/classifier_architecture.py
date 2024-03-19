import torch

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os

# hyperparameters
DROPOUT_P = 0.2
MB_NORM = True
LAYERS = [2048, 512, 128, 32]
N_FEATURES = 768

class MLP(nn.Module):
  def __init__(self,dimx=N_FEATURES*2, nlabels=2, layers=LAYERS, dropout_p=DROPOUT_P, use_batch_norm=MB_NORM,
               fnModel=None):
    super().__init__()
    
    # Change the architecture as you please, but it has to be THE SAME for the
    # pretraining and the refined training
    
    self.use_batch_norm = use_batch_norm
  
    self.output0 = nn.Linear(dimx, layers[0])
    self.output1 = nn.Linear(layers[0], layers[1])    
    #self.output2 = nn.Linear(1024,512)
    self.output3 = nn.Linear(layers[1], layers[2])
    self.output4 = nn.Linear(layers[2], layers[3])
    self.output5 = nn.Linear(layers[3], nlabels)

    self.dropout = nn.Dropout(p=DROPOUT_P)
    
    self.nonlinear = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    
    if self.use_batch_norm:
      self.batch_norm0 = nn.BatchNorm1d(layers[0])      
      self.batch_norm1 = nn.BatchNorm1d(layers[1])
      self.batch_norm3 = nn.BatchNorm1d(layers[2])
      self.batch_norm4 = nn.BatchNorm1d(layers[3])

    #Load pretrained weigths if provided  
    if fnModel:
      # Cargar la red
      self.load_state_dict(torch.load(fnModel))
      print("Loaded pre-trained weights from all haplotypes")

    
    self.total_params = sum(p.numel() for p in self.parameters())    
      
  def forward(self, x):
    # Pass the batched input tensor through each of our operations
    x = self.output0(x)
    if self.use_batch_norm:
      x = self.batch_norm0(x)
    x = self.nonlinear(x)
    x = self.dropout(x)
    
    x = self.output1(x)
    if self.use_batch_norm:
      x = self.batch_norm1(x)    
    x = self.nonlinear(x)
    x = self.dropout(x) 
    
    #x = self.output2(x) 
    #x = self.nonlinear(x)   
    
    x = self.output3(x)
    if self.use_batch_norm:
      x = self.batch_norm3(x)    
    x = self.nonlinear(x)
    x = self.dropout(x) 
    
    x = self.output4(x)
    if self.use_batch_norm:
      x = self.batch_norm4(x)    
    x = self.nonlinear(x)
    x = self.dropout(x) 
    
    x = self.output5(x)
    x = self.softmax(x)
    return x
    
