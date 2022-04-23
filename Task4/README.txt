# epitopes.py
- Downloads epitope data for all 6 haplotypes from remote repository.
- Discards epitopes without start and end information (only 250 out of 600k)
- Generates a csv with all the protein sequences and corresponding epitopes.  
- Variable verb inside the code controls output of print info.

# indep_haplotypes.py
- Downloads protein and epitope data from only 1 haplotype
- Haplotype specyfied by argument (from 0 to 5)
- Discards epitopes without start and end information (only 250 out of 600k)
- Implements sliding window from scratch 
