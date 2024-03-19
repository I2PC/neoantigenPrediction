Task 6. Prediction over new data

classifier_architecture.py: contains the architecture of the NN to make predictions

Input as sequence:
   - new_postprocessing_human_seq.py: processes the input sequence to perform data augmentation using the blosum62 matrix
   - new_prediction_nn_human_seq.py: makes predictions using the architecture over the newtork with saved weights

Input as NGS:
   - new_postprocessing_human.py: processes the input sequence to perform data augmentation using the blosum62 matrix
   - new_prediction_nn_human.py: makes predictions using the architecture over the newtork with saved weights



Workflow of new data is:
  - post-processing (data augmentation)
  - ProteinUnet (predict secondary strcuture of proteins form the primary one)
  - extract features with BERT (for primary and secondary structure separately)
  - both features files are used to predict epitopes with the NN