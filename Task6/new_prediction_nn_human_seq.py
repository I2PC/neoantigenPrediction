import torch
import numpy as np
from classifier_architecture import MLP
from pathlib import Path
from math import floor
from optparse import OptionParser
import pandas as pd
import sys

if __name__=='__main__':
  # def change_array(string):
  #   A = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.']
  #   B = []
  #   value = ''
  #   for j in range(0, len(string), 1):
  #     if string[j] in A:
  #       value = value + string[j]
  #     else:
  #       if len(value)>0:
  #         value1 = float(value)
  #         B.append(value1)
  #       value = ''
  #   array = np.array(B)
  #   return array

  parser= OptionParser()
  parser.add_option("-w", "--working_dir" , dest="workingdir", help= "working directory" , metavar="str")
  parser.add_option("-p", "--primary", dest="primary", help="primary BERT", metavar="FILE")
  parser.add_option("-s", "--secondary", dest="secondary", help="secondary BERT", metavar="FILE")
  parser.add_option("-o", "--output", dest="output", help="output", metavar="FILE")
  parser.add_option("-x", "--weights", dest="fnWeights", help="network weights", metavar="FILE")

  
  options, args= parser.parse_args()
  workingdir= options.workingdir
  primaryPATH= options.primary
  secondPATH= options.secondary
  OUTPATH= options.output
  weights = options.weights

  # Load model and weigths
  model = MLP()
  model.load_state_dict(torch.load(weights))
  print("Loaded model from disk")

  folderPATH= primaryPATH.split('out_bert_primary.txt')[0]
  # print(folderPATH)
  file1 = open(primaryPATH, 'r')
  file2 = open(secondPATH, 'r')

  #this code reads sequences from two files, processes them, creates input arrays XA and XB, and prints the length of the prot list.
  prot = file1.readlines()
  pred = file2.readlines()

  # Converting prot (list with separation by commas) into numpy
  data_list_prot = []
  for line in prot:
      values = [float(val) for val in line.strip().split(',')]
      data_list_prot.extend(values)

  num_elements_prot = len(data_list_prot)
  num_rows_prot = num_elements_prot // 768  
  numpy_array_prot = np.array(data_list_prot[:num_rows_prot*768]).reshape(num_rows_prot, 768)

  # Converting pred (list with separation by commas) into numpy
  data_list_pred = []
  for line in pred:
      values = [float(val) for val in line.strip().split(',')]
      data_list_pred.extend(values)

  num_elements_pred = len(data_list_pred)
  num_rows_pred = num_elements_pred // 768  
  numpy_array_pred = np.array(data_list_pred[:num_rows_prot*768]).reshape(num_rows_pred, 768)

  # inputsA = []
  # inputsB = []

  # for i in range(floor(len(prot)/2)):
  #     num2 = i*2 + 1
  #     X_prot = change_array(prot[num2])
  #     X_prot = X_prot[0:768]
  #     X_pred = change_array(pred[num2])
  #     X_pred = X_pred[0:768]
      
  #     inputsA.append(X_prot)
  #     inputsB.append(X_pred)

  # XA = np.array(inputsA, dtype='float32')
  # XB = np.array(inputsB, dtype='float32')
  # print(len(prot))

  # this code segment generates predictions using a loaded machine learning model, 
  #assigns the predicted probabilities and binary predictions to a DataFrame (predictions), 
  #and modifies the predicted values to binary format based on a threshold of 0.5.

  # Convert your input arrays (XA and XB) to PyTorch tensors
  XA_tensor = torch.tensor(numpy_array_prot, dtype=torch.float32)
  XB_tensor = torch.tensor(numpy_array_pred, dtype=torch.float32)

  # Set model to evaluation mode
  model.eval()
  # Perform predictions
  with torch.no_grad():
        concatenated_tensor = torch.cat((XA_tensor, XB_tensor), dim=1)
        # Pass the concatenated tensor through your model
        outputs = model(concatenated_tensor)

  # Convert the PyTorch tensor to a NumPy array
  y_new = outputs.numpy()

  predictions= pd.DataFrame(columns=['Prediction', 'Probability'])
  y_df= pd.DataFrame(y_new)
  predictions['Probability']=y_df.iloc[:,1] 
  y_new[y_new <= 0.5] = 0
  y_new[y_new > 0.5] = 1
  df= pd.DataFrame(y_new)
  predictions['Prediction']=df.iloc[:, 1]

  predictions.to_csv(f'{folderPATH}/predictions.txt', index=False)

  prediction= pd.read_csv(f'{folderPATH}/predictions.txt')
  post= pd.read_csv(f'{folderPATH}/proteins_post.csv') # reads CSV file named "proteins_post.txt" (resulting from postprocessing_human_seq.py), contiene seq, seq_blosum y FPKM.
  output= pd.DataFrame(columns=['sequence', 'seq_blosum', 'prediction','probability'])
  output['sequence']= post['seq'] # assigns the 'seq' column from the DataFrame post to the 'sequence' column of the output DataFrame.
  output['seq_blosum']= post['seq_blosum'] # assigns the 'seq_blosum' column from the DataFrame post to the 'seq_blosum' column of the output DataFrame

  #for i in range(prediction.shape[0]):
    # prediction.iloc[i,0]= int(prediction.iloc[i,0][0].split('[')[1].split('.')[0])
    # predicition.iloc[i,1]= int(prediction.iloc[i,1][0].split('[')[1].split(']')[0])
      

  output['prediction']= prediction['Prediction'] # assigns the values from the 'Prediction' column of the DataFrame prediction to the 'prediction' column of the output DataFrame
  output['probability']= prediction['Probability'] # assigns the values from the 'Probability' column of the DataFrame prediction to the 'probability' column of the output DataFrame

  cc=0
  current_score=0
  peptides=pd.DataFrame(columns=['extracted antigen', 'Prediction', 'Probability']) # Initializes an empty DataFrame peptides with columns for 'extracted antigen', 'Prediction', and 'Probability'.
  current_original=""

  for index, row in output.iterrows(): # Looping Through each row of output DataFrame
      if row['sequence']!=current_original: # Checks if the sequence ('sequence') is different from the current_original sequence
          cc=cc+1
          current_score=row['prediction'] # Updates current_score with the value of the current row's prediction ('prediction')
          current_original= row['sequence'] # Updates current_original with the value of the current row's sequence ('sequence')
          counter=0
          peptides.loc[cc]= [row['sequence'], row['prediction'], row['probability']] #  new row to the peptides DataFrame with the sequence, prediction, and probability of the current row
      else:
          if row['prediction'] != current_score: # Checks if the current row's prediction is different from the current_score
              counter= counter +1 # increments counter
              if counter>15:
                  peptides.loc[cc, 'Prediction']=row['prediction'] # updates the prediction for the current sequence in the peptides DataFrame with the current row's prediction
              if row['probability'] > peptides.loc[cc, 'Probability']:  # Check if current row's probability is higher
                    peptides.loc[cc, 'Probability'] = row['probability']
                    
  peptides.to_csv(f'{OUTPATH}', index=False)

