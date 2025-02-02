# epitopes.py
- Downloads epitope data for all 6 haplotypes from remote repository.
- Discards epitopes without start and end information (only 250 out of 600k)
- Generates a csv with all the protein sequences and corresponding epitopes.  
- Variable verb inside the code controls output of print info.

# indep_haplotypes.py
- Downloads protein and epitope data from only 1 haplotype
- Haplotype specified by argument (from 0 to 5)
- Discards epitopes without start and end information (only 250 out of 600k)
- Implements sliding window from scratch 
- Results in folder 'training_indep_haplotypes/'
- header: 30aa_window, contains_epitope?, protein_id, window_id
  
Haplotype    Proteins time (s) time (min)  Rows in training set  Epitopes
MHC_1_A   17037/17037  7936.4     132.28    10521323 / 10521323  16.47 %
MHC_1_B   19235/19235  8635.2     143.92    11478532 / 11478532  21.43 %
MHC_1_C   11428/11428  5102.4      85.04     7365457 /  7365457  11.58 %
MHC_2_DP   6340/ 6340  2679.1      44.65     4026748 /  4026748   7.63 %
MHC_2_DQ   3257/ 3257  1262.4      21.04     1997461 /  1997461   7.48 %
MHC_2_DR   9916/ 9916  4266.6      71.11     6130607 /  6130607  10.24 %

- possible improvement: paralellize condition computation for all windows of a
  given protein 
  (i.e. line 122) list_conditions = list(map(fun_contains,all_windows))

# indep_haplotypes_seq.py
- Sequential version of indep_haplotypes (for backup/code organization)
- The paralell parts are commented out
- Gives the same results for the sliding windows as the paralell version
- It is slower than the paralell version
- Only retunrs 30aa_window,and label (no parent protein nor window id)

# remove_repeated.py
- Remove duplicates and contradictory windows (same 30_aas chains labeled 0 and 1) from training set 

#  Rows    Haplotype     Epitopes
   8667465 MHC_1_A.csv   19.44674 %
   8882127 MHC_1_B.csv   26.44627 %
   6727170 MHC_1_C.csv   12.55016 %
   3958497 MHC_2_DP.csv   7.68769 %
   1878652 MHC_2_DQ.csv   7.72559 %
   5589346 MHC_2_DR.csv  10.76267 %
  35703257 total
  
# resample.py
- Datasets obtained in remove_repeated are higly unbalanced. Using imblearn, randomly undersamples the majority class until 50/50 is achieved
#  Rows    Haplotype     Epitopes
   3371080 MHC_1_A_balanced.csv  
   4697984 MHC_1_B_balanced.csv  
   1688542 MHC_1_C_balanced.csv  
    608634 MHC_2_DP_balanced.csv 
    290274 MHC_2_DQ_balanced.csv 
   1203126 MHC_2_DR_balanced.csv 
  11859640 total
