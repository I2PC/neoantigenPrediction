"""
MLP classifier for ALL haplotypes (alleles)
Refined training not implemented!
To do that:
  - separate achitecture of the network in another file arch.py (same folder)
  - data needs to be separated by allele beforehand (start from the split data)
"""

# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os

# hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-5
EPOCHS_TO_TRAIN = 3000
DROPOUT_P = 0.2
ARCH_NAME = "arch_8"
MB_NORM = True
#LAYERS = [1024, 256, 64, 16]
LAYERS = [2048, 512, 128, 32]

print("Training parameters:")
print("  Batch size:", BATCH_SIZE)
print("  Learning rate: {:.0e}".format(LEARNING_RATE))
print("  Epochs:", EPOCHS_TO_TRAIN)
print("Network architecture:", ARCH_NAME)
print("  Layers:", [1536] + LAYERS + [2])
print("  Dropout probability:", DROPOUT_P)
print("  Minibatch normalization?:", MB_NORM)

# constants
N_FEATURES = 768
splits = ["train", "validation", "test"]

# select mode passing an argument to the script:
# -1: pre-train with all haplotypes
# otherwise: refined train with selected haplotype (keep training)
# hp_types = ['MHC_1_A','MHC_1_B','MHC_1_C','MHC_2_DP','MHC_2_DQ','MHC_2_DR']
hp_types =['MHC_I_A', 'MHC_I_B', 'MHC_I_C']

MODE = -1 if len(sys.argv) != 2 or (int(sys.argv[1]) <= 0 or int(sys.argv[1]) > len(hp_types)) else int(sys.argv[1])

ALL = (MODE == -1)
if ALL: 
  title = 'Pre-training with ALL haplotypes'
  h_name = "ALL"
else: 
  h_name = hp_types[MODE-1]
  hp_types = [h_name]
  title = 'Refined training for haplotype ' + h_name
print(title)


train_id = "_lr{:.0e}_epochs{}_mbatch{}".format(LEARNING_RATE,EPOCHS_TO_TRAIN,BATCH_SIZE)

#data_root = Path("data_split", "data_split_reduced")
#train_outputs_dir = Path("train_outputs", ARCH_NAME, "ALL_reduced" + train_id)

data_root = Path("data_split")
train_outputs_dir = Path("train_outputs", ARCH_NAME,  h_name + train_id)

weights_dir = train_outputs_dir / "weights"
traces_dir = train_outputs_dir / "traces"

weights_dir.mkdir(parents=True, exist_ok=True)
traces_dir.mkdir(parents=True, exist_ok=True)
print("Created directory for training outputs: ", train_outputs_dir.absolute())

 
# Auxiliary functions
def print_start_msg(msg, indent_level=0):
    tic = datetime.now()
    indent = " " * 2 * indent_level
    print("{}{}, started at: {}".format(indent, msg, tic))
    return tic

def print_elapsed_time(tic, msg, indent_level=0):
    tac = datetime.now()
    indent = " " * 2 * indent_level
    print("{}{} at: {}".format(indent, msg, tac))
    print("{}Elapsed time: {}".format(indent, tac - tic))
    return tac


# Load data (features and labels)

X = {}
Y = {}
datasets = {}
x_numpy = {}
y_numpy = {}

tic = print_start_msg("Loading the data from disk")

for sp in splits:
#for sp in splits[1:]:
  X[sp] = []
  Y[sp] = []
  for h in hp_types:
    X[sp].append(np.load(data_root / sp / (h + "_features.npy")))
    Y[sp].append(np.load(data_root / sp / (h + "_labels.npy")))

  x_numpy[sp] = np.concatenate(X[sp],dtype=np.float32)
  y_numpy[sp] = np.concatenate(Y[sp],dtype=np.int64)

  datasets[sp] = TensorDataset(torch.as_tensor(x_numpy[sp]), torch.as_tensor(y_numpy[sp]))

  # debug prints to make sure concatenation is working
  print("  Split:", sp)
  print("    X.shape:", list(map(lambda x: x.shape, X[sp])))
  print("    Y.shape:", list(map(lambda x: x.shape, Y[sp])))

  print("    xshape:", x_numpy[sp].shape)
  print("    yshape:", y_numpy[sp].shape)

  print("    dataset length:", len(datasets[sp]))
  print_elapsed_time(tic, "Finished {} split".format(sp), 1)


print("""        Training instances   {} (80%),
        Validation instances {} (10%),
        Test instances       {} (10%)""".format(len(datasets["train"]),
        len(datasets["validation"]), len(datasets["test"])))

print_elapsed_time(tic, "Finished reading the data")

# Dataloaders

tic = print_start_msg("Creating dataloaders with minibatches of {}".format(BATCH_SIZE))

n_workers = 0 if "posix" not in os.name else 2

trainloader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=n_workers)

# do not use, CUDA out of memory error
#trainloader_nobatch = DataLoader(datasets["train"], batch_size=len(datasets["train"]),
#                                 shuffle=False, num_workers=n_workers)

validloader = DataLoader(datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers)
#validloader_nobatch = DataLoader(datasets["validation"], batch_size=len(datasets["validation"]),
#                                 shuffle=False, num_workers=n_workers)

testloader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=n_workers)
#testloader_nobatch = DataLoader(datasets["test"], batch_size=len(datasets["test"]),
#                                shuffle=False, num_workers=n_workers)

for i, l in enumerate([trainloader, validloader, testloader]):
    sp = splits[i]
    print("Size of {}-loader: {} \n        {}-dataset: {}".format(
        sp, len(l), sp, len(datasets[sp])))

print_elapsed_time(tic, "Finished dataloaders")

#sys.exit(0) # debug

# Neural Network class
class MLP(nn.Module):
  def __init__(self,dimx=768*2, nlabels=2, layers=LAYERS, dropout_p=DROPOUT_P,
  use_batch_norm=MB_NORM,fnModel=None):
    super().__init__()
    
    # Change the arachitecture as you please, but it has to be THE SAME for the
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
    
    #Load pretrained weights if provided
    if fnModel:
      self.load_state_dict(torch.load(fnModel))
      print("Loaded pre-trained weights from all haplotypes training")

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

# Extended class with training and evaluation
class MLP_extended(MLP):
  def __init__(self, dimx=N_FEATURES*2, nlabels=2, lr=LEARNING_RATE, verb=True, fnModel=None, layers=LAYERS):
    super().__init__(dimx, nlabels, layers=layers, fnModel=fnModel)

    self.lr = lr # learning rate  
    self.trained_epochs = 0

    # gpu
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #self.device = torch.device("cpu")
    self.to(self.device) # do this before defining optimizer
    self.optim = optim.Adam(self.parameters(), self.lr)  
    self.criterion = nn.CrossEntropyLoss()
    self.loss_during_training = []
    self.traces = {'tr_loss': [], 'val_loss': [], 'tr_acc':[],'val_acc':[]}
    self.epoch_durations = []  # time elapsed for training in 1 epoch
    
    if verb:
        print('Information of the network:')
        print("-- Number of parameters: {:.2e}".format(self.total_params))
        print('-- Nonlinearity:', self.nonlinear)
        print('-- Device:', self.device)
        #print('-- Optimizer:', self.optim)
    
  def calc_acc(self,features,labels):
    acc=0
    with torch.no_grad():
      self.eval()
      probs = self.forward(features)      
      top_p, top_class = probs.topk(1, dim=1)
      equals = (top_class == labels.view(features.shape[0], 1))
      acc += torch.mean(equals.type(torch.DoubleTensor))
    return acc
  
  def calc_acc_batch(self,dataloader):
    """Calculate accuracy, but with batched dataloader"""
    acc=0
    with torch.no_grad():
      self.eval()        
      for features,labels in dataloader:
        features = features.to(self.device)
        labels = labels.to(self.device) 
        probs = self.forward(features)      
        top_p, top_class = probs.topk(1, dim=1)
        equals = (top_class == labels.view(features.shape[0], 1))
        acc += torch.mean(equals.type(torch.DoubleTensor))
        
    self.train()
    return acc/len(dataloader) 
  
  def calc_loss(self,X_nobatch,Y_nobatch):
    with torch.no_grad():
      self.eval()        
      out = self.forward(X_nobatch)
      loss = self.criterion(out, Y_nobatch)
      
    self.train()
    return loss.item()
  
  
  def calc_loss_batch(self, dataloader):
    running_loss = 0.
    with torch.no_grad():
      self.eval()        
      for features, labels in dataloader:       
        features = features.to(self.device)
        labels = labels.to(self.device)
        out = self.forward(features)
        loss = self.criterion(out, labels)
        running_loss += loss.item()
        
    self.train()
    return running_loss/len(dataloader)
    

  def trainloop(self,epochs, trainloader, val_loader_nobatch, max_pr = 500, 
                traces_file_path=traces_dir/"traces.txt", weights_save_path=weights_dir):
    
    self.best_val_acc = 0
    self.beast_epoch = 0
    t_ini = print_start_msg("TRAINING")
    pr = max(1, int(epochs/max_pr))
    
    # Get train and valid tensors
    #for features, labels in tr_loader_nobatch:
    #  X_train = features.to(self.device)
    #  Y_train = labels.to(self.device)
    #for f, l in val_loader_nobatch:
    #  X_val = f.to(self.device)
    #  Y_val = l.to(self.device)
    
    header = 'Epoch  Train-loss  Valid-loss  Train-acc  Valid-acc  time taken'
    print(header)
    with open(traces_file_path, "w") as traces_file:
        traces_file.write(header+"\n")
    pr_fmt = '{:5}  {:10.5f}  {:10.5f}  {:9.5f}  {:9.5f}  {}'
    
    # Calculations before trainng
    if self.trained_epochs==0:
      # accuracy before training
      self.traces['tr_acc'].append(self.calc_acc_batch(trainloader))
      self.traces['val_acc'].append(self.calc_acc_batch(validloader))

      # loss before training
      self.traces['tr_loss'].append(self.calc_loss_batch(trainloader))
      self.traces['val_loss'].append(self.calc_loss_batch(validloader))
      
      msg_to_print = pr_fmt.format(0,self.traces['tr_loss'][-1],self.traces['val_loss'][-1],
                          self.traces['tr_acc'][-1],self.traces['val_acc'][-1],
                          datetime.now()-t_ini)
      print(msg_to_print)
      with open(traces_file_path, "a") as traces_file:
          traces_file.write(msg_to_print+"\n")
      
    
    for e in range(1, int(epochs)+1): 
      tic = datetime.now()
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
      self.traces['tr_acc'].append(self.calc_acc_batch(trainloader))
      self.traces['val_acc'].append(self.calc_acc_batch(validloader))

      # loss during training
      self.traces['val_loss'].append(self.calc_loss_batch(validloader))
      
      toc = datetime.now()
      
      self.trained_epochs += 1
      
      # Print info every pr_e epochs
      if(e==1 or e % pr == 0 or e ==epochs):
        msg_to_print = pr_fmt.format(e,self.traces['tr_loss'][-1],self.traces['val_loss'][-1],
                            self.traces['tr_acc'][-1],self.traces['val_acc'][-1],toc-tic)
        print(msg_to_print)
        with open(traces_file_path, "a") as traces_file:
            traces_file.write(msg_to_print+"\n")
      
      # save weights every epoch
      name = 'all_haplotypes' if ALL else hp_types[0]
      #weights_filename = "{}_epochs_{}.pt".format(name,str(self.trained_epochs))
      weights_filename = "{}.pt".format(name)

      if self.traces['val_acc'][-1] > self.best_val_acc:
        self.best_val_acc = self.traces['val_acc'][-1]
        self.best_epoch = self.trained_epochs
        torch.save(my_mlp.state_dict(), weights_save_path / weights_filename)
      # end of SGD loop

    print('Trained {} epochs (total {})'.format(epochs, self.trained_epochs))
    print("Traces saved in:", traces_file_path.absolute())
    print("Weights saved in:", weights_save_path.absolute())
    print_elapsed_time(t_ini, "Training finished")
# end of MLP extended class    

# Train
if ALL:
  my_mlp = MLP_extended()
else:
  #when training for haplotypes we load the weigths of the training with all haplotypes
  train_id_all = "_lr{:.0e}_epochs500_mbatch{}".format(LEARNING_RATE,BATCH_SIZE)
  fnModel_dir = Path("train_outputs", ARCH_NAME, "ALL" + train_id_all, "weights")
  weights_path = fnModel_dir/'all_haplotypes.pt'
  my_mlp = MLP_extended(fnModel=weights_path)

print("Model Architecture:")
print(my_mlp)

my_mlp.trainloop(EPOCHS_TO_TRAIN,trainloader,validloader,
                 traces_file_path=traces_dir/"traces_during_training.txt")

  
# Results

def plot_traces(mlp, epoch_0=True, max_points = 10,title=None):
  ''' plot evolution of loss during training '''
  ini_val = 0 if epoch_0 else 1
  s = '.-' if mlp.trained_epochs < 100 else '-'
  fig, ax = plt.subplots(1,2,figsize=(15,5))
  if not (title is None): fig.suptitle(title)
      
  ax[0].plot(range(ini_val, mlp.trained_epochs+1), mlp.traces['tr_loss'][ini_val:],s+'b',label='Training')
  ax[0].plot(range(ini_val, mlp.trained_epochs+1), mlp.traces['val_loss'][ini_val:],s+'r',label='Validation')
  ax[0].set_title('Loss vs Epochs')
  ax[0].set_xlabel('Epochs')
  ax[0].set_xticks(range(ini_val, mlp.trained_epochs+1, max(1, int(mlp.trained_epochs/max_points))))
  ax[0].set_ylabel('Loss')
  if epoch_0: ax[0].set_yscale('log')

  ax[1].plot(range(ini_val, mlp.trained_epochs+1), mlp.traces['tr_acc'][ini_val:],s+'b',label='Training')
  ax[1].plot(range(ini_val, mlp.trained_epochs+1), mlp.traces['val_acc'][ini_val:],s+'r',label='Validation')
  ax[1].set_title('Accuracy vs Epochs')
  ax[1].set_xlabel('Epochs')
  ax[1].set_xticks(range(ini_val, mlp.trained_epochs+1, max(1, int(mlp.trained_epochs/max_points))))
  ax[1].set_ylim([0.2, 1])
  ax[1].set_ylabel('Accuracy')

  ax[0].legend()
  
  # save the plot
  out_path = traces_dir / "Accuracy_and_loss_during_training_epochs_{}.pdf".format(my_mlp.trained_epochs)
  plt.savefig(out_path, bbox_inches="tight")
  print("plot saved in: {}".format(out_path))  
  plt.show()

plot_traces(my_mlp,title=h_name+train_id)

# Accuracy in test

print("Best validation accuracy in epoch {}".format(my_mlp.best_epoch))
#best_mlp =  MLP_extended(verb=False)
#name = 'all_haplotypes' if ALL else hp_types[0]
#weights_filename = "{}_epochs_{}.pt".format(name, my_mlp.best_epoch)
#state_dict = torch.load(weights_dir / weights_filename)
#best_mlp.load_state_dict(state_dict)

test_acc_last_epoch = my_mlp.calc_acc_batch(testloader)
#test_acc_best_epoch = best_mlp.calc_acc_batch(testloader)
print('Accuracy in test (last epoch): {:.5f}'.format(test_acc_last_epoch.item()))
#print('Accuracy in test (best epoch {}): {:.5f}'.format(my_mlp.best_epoch,test_acc_best_epoch.item()))

