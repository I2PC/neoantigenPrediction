TASK 5. NN for neoantigen prediction (training)

If pre-train with all haplotypes: MODE=-1, run: python -u classifier.py
If refine training with haplotype A: MODE=1, run: python -u classifier.py 1
If refine training with haplotype B: MODE=2, run: python -u classifier.py 2
If refine training with haplotype C: MODE=3, run: python -u classifier.py 3


Default architecture:
[1536, 2048, 512, 128, 32, 2]

- Relu nonlinearities
- Softmax after last layer
- CrossEntropy loss


resulting weights in models/
