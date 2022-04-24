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

# indep_haplotypes_seq.py
- Sequential version of indep_haplotypes (for backup/code organization)
- The paralell parts are commented out
- Gives the same results for the windows as the paralell version
- It is slower than the paralell version
