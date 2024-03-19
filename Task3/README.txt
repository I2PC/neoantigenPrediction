TASK 3. BERT on secondary structure 
Dataset 1 (epitope/no-epitope) and the secondary
predictions are input to the BERT network pre-trained
with dataset 2 (whole proteome) to have two feature
vectors as output.

This folder contains the necessary codes to obain BERT encoding from the input data and to transform the BERT encoding into the format accepted by the neural network.

##Folder organization

- Primary_structure folder contains code necessary for the BERT training (of primary structure)
- Secondary_structure folder contains code necessary for the BERT training (of secondary  structure)

For both folders:
- create_pretraining_data.py: creates file with data as binary string sequences
- run_pretraining.py: code needed to train the BERT models
- extract_features4.py: code to obtain features

Models are safed in a .json file (if retraining BERT, these need to be updated in richen-dos for the nap web to predict)


##Output folders organization:
- PrimaryStructure: contains the Primary structure BERT
- SecondaryStructure: contains the Secondary Structure BERT
- BERT concatdata: contains the primary and secondary BERT data concatenated in a txt file. This is done by ./concatdata/concat data.py
- BERT finaldata: It is the BERT data merged and in the appropriate format for the neural network. This is done by ./concatdata/data to npy.py 
