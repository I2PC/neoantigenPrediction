# Task 3

The code contained in this folder allows the user to perform the steps included in Task 3.

## Code organization

The feature extraction for both the primary and secondary structure can be run as it was done previously (folders Primary_structure and Secondary_structure). The code is based on the [original Google research repository](https://github.com/google-research/bert/).

The required commands can be executed from the Scripts_Task_3 file. For the primary structure, an alternative is proposed in the form of the TAPE library, which allows us to obtain the embeddings for the subsequences by using a pre-trained model. As commented, the previous approach can also be used for this purpose.

If run locally, some older versions of TensorFlow and Numpy may be needed. We recommend:
```bash
!pip install bert-tensorflow
!python3.7 -m pip install tensorflow-gpu==1.15.0
!pip install -U numpy==1.18.5
```
