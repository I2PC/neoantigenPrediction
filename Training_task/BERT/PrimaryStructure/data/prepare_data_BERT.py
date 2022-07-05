import os
import numpy as np
from optparse import OptionParser
from Bio import SeqIO

def prepare_data(proteome_file):
	"""
		introduce the function (removing headers and adding - to the beggining)
	"""

	fasta_seq= SeqIO.parse(open(proteome_file), 'fasta')
	output_file='human_proteome_seq.txt'

	with open(output_file, 'w') as out_file:
		for fasta in fasta_seq:
			name, sequence= fasta.id, str(fasta.seq)
			out_file.write( '-' + sequence + '\n')


def remove_newline(processed_proteome):

	'''
	Read the human proteome file and remove NewLine char(\n)
	'''

	entries = []
	with  open(processed_proteome) as fp: #add the file of the processed human proteome (removing headers and adding - to the beggining)
		contents = fp.read()
		for entry in contents.split('-'):
			entry = entry.replace('\n','')
			entries.append(entry)
	entries.pop(0)

	return entries

def create_window_30(file_prot):

	'''
	Create windows of 30 aminoacids
	'''
	n = 30
	strings_30 = []
	for entry in file_prot:
		split_strings = [entry[index : index + n] for index in range(0, len(entry), n)]

		strings_30.append(split_strings)

	return strings_30

def insert_spaces(file_30):


	'''
	For the regular BERT model, we need a space between each aminoacid
	This token is crucial for creating a training dataset and 
	for running BERT over the database
	'''

	final_lines = []
	for line in file_30:
		line = (line.replace("", " ")[1: -1])
		final_lines.append(line)

	return final_lines


def main():

	parser = OptionParser( description = 'BERT data preparation')
	parser.add_option('-i', '--input', dest='input', help='Input file with the proteome or secondary prediction of the proteome')
	options, args= parser.parse_args()

	INfile= options.input

	prepare_data(INfile)
	entries= remove_newline('human_proteome_seq.txt')
	strings_30= create_window_30(entries)
	file_write = open('dataset_30.txt','w')
	for string in strings_30:
		for s in string:
			file_write.write(s + "\n")

	with open('dataset_30.txt', 'r') as file_read:
		seq= file_read.readlines()

	#seq_entries= remove_newline(seq)
	final_lines=insert_spaces(seq)

	file_write = open('dataset_30_spaces.txt','w')

	for line in final_lines:
		file_write.write(line)


if __name__ == "__main__":
	main()

