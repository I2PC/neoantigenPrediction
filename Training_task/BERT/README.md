
# BERT

The code and data in this folder allows to pretrain the Primary and Secondry BERT models. For the primary structure, an alternative is proposed in the form of the TAPE library, which allows us to obtain the embeddings for the subsequences by using a pre-trained model. 

## Code organization

PrimaryStructure and SecondaryStructure folder contain all the necessary code and data to pretrain the BERT models.  
requirements.txt show the specific packages needed to run the code.
bert folder is the cloned BERT repository so we can access to the different codes we need to pretrain.

- **PrimaryStructure**: 
	- data: 
		- whole_human_proteome.fasta: Provides the sequence of the whole human proteome. It is obtained from UniProt.
		- prepare_data_BERT.py: Code necessary for the preparation of the data for Primary BERT. Only imput needed is the whole_human_proteome.fasta file. It gives:
			- human_proteome_seq.txt: Proteome in txt format, with fasta header removed
			- dataset_30.txt: Proteome divided in sequences of 30aa. This is the imput for the secondary structure prediction on the whole proteome.
			- dataset_30_spaces.txt: Proteome divided in sequences of 30aa and with spaces. This is the input to train BERT primary structure language.

