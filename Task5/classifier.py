# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import time, numpy as np, matplotlib.pyplot as plt

#constants
N_FEATURES = 768
BATCH_SIZE=100

# select mode:
# -1: pre-train with all haplotypes
# 0 - 5: refined train with selected haplotype (keep training)
hp_types = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
MODE = -1
ALL = not (MODE >= 0 and MODE <=5)
if ALL: 
  print('Pre-training with all haplotypes')
else: 
  hp_types = [hp_types[MODE]]
  print('Refined training for haplotype',hp_types)


# load data (features and labels)
X = []
Y = []
for h in hp_types:
 X.append(np.load('synthetic_data/'+h+'_features.npy'))
 Y.append(np.load('synthetic_data/'+h+'_labels.npy'))
 
tensor_x = torch.as_tensor(np.array(X,dtype=np.float32).reshape(-1,N_FEATURES*2)) # transform to torch tensor
tensor_y = torch.as_tensor(np.array(Y,dtype=np.int64).reshape(-1))

# Split data into training, validation and test 
tr_= int(0.8*len(tensor_x))
val_ = int(0.1*len(tensor_x))

# if arrays are too long, a custom dataset might be needed, 
# and potentially split the arrays to several files...
trainset = TensorDataset(tensor_x[:tr_], tensor_y[:tr_])
validset = TensorDataset(tensor_x[tr_:tr_+val_], tensor_y[tr_:tr_+val_])
testset = TensorDataset(tensor_x[tr_+val_:], tensor_y[tr_+val_:])

print('''      Training instances   {} (80%), 
      Validation instances {} (10%),
      Test instances       {} (10%)'''.format(
      len(trainset),len(validset),len(testset)))

# dataloaders
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
trainloader_nobatch = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
validloader_nobatch = DataLoader(validset, batch_size=len(validset), shuffle=False)
testloader_nobatch = DataLoader(testset, batch_size=len(testset), shuffle=False)


#Neural Network class
class MLP(nn.Module):
  def __init__(self,dimx=768*2,nlabels=2):
    super().__init__()
    
    # Change the arachitecture as you please, but it has to be THE SAME for the
    # pretraining and the refined training
    
    self.output1 = nn.Linear(dimx,1024)
    self.output2 = nn.Linear(1024,512)
    self.output3 = nn.Linear(512,128)
    self.output4 = nn.Linear(128,32)
    self.output5 = nn.Linear(32,nlabels)

    self.nonlinear = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)                                                             
      
  def forward(self, x):
    # Pass the batched input tensor through each of our operations
    x = self.output1(x)
    x = self.nonlinear(x)
    x = self.output2(x) 
    x = self.nonlinear(x)   
    x = self.output3(x)
    x = self.nonlinear(x)   
    x = self.output4(x)
    x = self.nonlinear(x)   
    x = self.output5(x)
    x = self.softmax(x)
    return x

# Extended class with training and evaluation
class MLP_extended(MLP):
  def __init__(self, dimx=N_FEATURES*2, nlabels=2, lr=5e-4):
    super().__init__(dimx, nlabels)

    self.lr = lr # learning rate  
    self.trained_epochs = 0

    # gpu
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device) # do this before defining optimizer
    self.optim = optim.Adam(self.parameters(), self.lr)  
    self.criterion = nn.CrossEntropyLoss()
    self.loss_during_training = []
    self.traces = {'tr_loss': [], 'val_loss': [], 'tr_acc':[],'val_acc':[]}
    
    print('Optimizer:',self.optim)
    print('Nonlinearity:',self.nonlinear)
    print('Device:',self.device)
    
  def calc_acc(self,features,labels):
    acc=0
    with torch.no_grad():
      probs = self.forward(features)      
      top_p, top_class = probs.topk(1, dim=1)
      equals = (top_class == labels.view(features.shape[0], 1))
      acc += torch.mean(equals.type(torch.DoubleTensor))
    return acc
  
  def calc_loss(self,X_nobatch,Y_nobatch):
    with torch.no_grad():
      out = self.forward(X_nobatch)
      loss = self.criterion(out, Y_nobatch) 
    return loss.item()

  def trainloop(self,epochs,trainloader,tr_loader_nobatch,val_loader_nobatch
                ,max_pr = 10):
    
    pr = max(1, int(epochs/max_pr))
    t_ini = time.time()
    
    # Get train and valid tensors
    for features, labels in tr_loader_nobatch:
      X_train = features.to(self.device)
      Y_train = labels.to(self.device)
    for f, l in val_loader_nobatch:
      X_val = f.to(self.device)
      Y_val = l.to(self.device)
      
    print('Epoch  Train_loss  Valid_loss  Train_acc  Valid_acc  time(s)  min')
    pr_fmt = '{:5}  {:10.4f}  {:10.4f}  {:9.4f}  {:9.4f}  {:7.1f}  {:3.1f}'
    
    # Calculations before trainng
    if self.trained_epochs==0:
      # accuracy before training
      self.traces['tr_acc'].append(self.calc_acc(X_train,Y_train))
      self.traces['val_acc'].append(self.calc_acc(X_val,Y_val))

      # loss before training
      self.traces['tr_loss'].append(self.calc_loss(X_train,Y_train))
      self.traces['val_loss'].append(self.calc_loss(X_val,Y_val))
      
      print(pr_fmt.format(0,self.traces['tr_loss'][-1],self.traces['val_loss'][-1],
                          self.traces['tr_acc'][-1],self.traces['val_acc'][-1],
                          time.time()-t_ini,(time.time()-t_ini)/60))
      
    
    for e in range(1, int(epochs)+1): 
      tic = time.time()
      running_loss = 0.
      for features, labels in trainloader:       
        features = features.to(self.device)
        labels = labels.to(self.device) 
        
        self.optim.zero_grad()  #TO RESET GRADIENTS!
        
        out = self.forward(features)
        loss = self.criterion(out, labels)
        running_loss += loss.item()
        
        loss.backward() ## Compute gradients        
        self.optim.step() ## Perform one SGD step            
      self.traces['tr_loss'].append(running_loss/len(trainloader))
      
      # accuracy during training
      self.traces['tr_acc'].append(self.calc_acc(X_train,Y_train))
      self.traces['val_acc'].append(self.calc_acc(X_val,Y_val))

      # loss during training
      self.traces['val_loss'].append(self.calc_loss(X_val,Y_val))
      
      toc = time.time()
      
      self.trained_epochs += 1
      
      # Print info every pr_e epochs
      if(e==1 or e % pr == 0 or e ==epochs): 
              print(pr_fmt.format(e,self.traces['tr_loss'][-1],self.traces['val_loss'][-1],
                          self.traces['tr_acc'][-1],self.traces['val_acc'][-1],
                          toc-tic,(toc-tic)/60))
      # end of SGD loop

    t_end = time.time()
    print('Trained {} epochs (total {})'.format(epochs, self.trained_epochs))
    print('Total time: {:4.1f} s  ({:4.1f} min)'.format(t_end-t_ini, (t_end-t_ini)/60))
# end of MLP extended class    


my_mlp = MLP_extended()
epochs=10
my_mlp.trainloop(epochs,trainloader,trainloader_nobatch,validloader_nobatch)


for features, labels in trainloader_nobatch:
  X_train = features.to(my_mlp.device)
  Y_train = labels.to(my_mlp.device)

def plot_traces(sae, epoch_0=True, max_points = 10):
  ''' plot evolution of loss during training '''
  ini_val = 0 if epoch_0 else 1
  s = '.-' if sae.trained_epochs < 100 else '-'
  fig, ax = plt.subplots(1,2,figsize=(15,5))
      
  ax[0].plot(range(ini_val, sae.trained_epochs+1), sae.traces['tr_loss'][ini_val:],s+'b',label='Training')
  ax[0].plot(range(ini_val, sae.trained_epochs+1), sae.traces['val_loss'][ini_val:],s+'r',label='Validation')
  ax[0].set_title('Loss vs Epochs')
  ax[0].set_xlabel('Epochs')
  ax[0].set_xticks(range(ini_val, sae.trained_epochs+1, max(1, int(sae.trained_epochs/max_points))))
  ax[0].set_ylabel('Loss')
  if epoch_0: ax[0].set_yscale('log')

  ax[1].plot(range(ini_val, sae.trained_epochs+1), sae.traces['tr_acc'][ini_val:],s+'b',label='Training')
  ax[1].plot(range(ini_val, sae.trained_epochs+1), sae.traces['val_acc'][ini_val:],s+'r',label='Validation')
  ax[1].set_title('Accuracy vs Epochs')
  ax[1].set_xlabel('Epochs')
  ax[1].set_xticks(range(ini_val, sae.trained_epochs+1, max(1, int(sae.trained_epochs/max_points))))
  ax[1].set_ylim([0.2, 1])
  ax[1].set_ylabel('Accuracy')

  ax[0].legend()
  plt.show()

plot_traces(my_mlp)

# Accuracy in test:
for features, labels in testloader_nobatch:
  X_test = features.to(my_mlp.device)
  Y_test = labels.to(my_mlp.device)
test_acc = my_mlp.calc_acc(X_test, Y_test)
print('Accuracy in test: {:.3f}'.format(test_acc.item()))
