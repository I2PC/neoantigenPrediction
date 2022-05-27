Pre-train with all haplotypes: MODE=-1
resulting weights in models/all_haplotypes_epochs20.pth

Then refine training for each haplotype: MODE= 0,1,2,3,4, or 5


Default architecture:
[1536, 1024, 512, 128, 32, 2]

- Relu nonlinearities
- Softmax after last layer
- CrossEntropy loss