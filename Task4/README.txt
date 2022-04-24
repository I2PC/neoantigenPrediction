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
  
Haplotype    Proteins time (s) time (min)  Rows in training set  Epitopes
MHC_1_A   17037/17037  7936.4     132.28    10521323 / 10521323  16.47 %
MHC_1_B   19235/19235  8635.2     143.92    11478532 / 11478532  21.43 %

# indep_haplotypes_seq.py
- Sequential version of indep_haplotypes (for backup/code organization)
- The paralell parts are commented out
- Gives the same results for the sliding windows as the paralell version
- It is slower than the paralell version
