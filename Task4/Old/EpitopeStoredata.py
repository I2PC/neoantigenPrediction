import requests
import json
import string
import mysql.connector
import mysql
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import csv

mydb = mysql.connector.connect(
    host="127.0.0.1",
    user='paolanu',
    passwd="<53909150paola>",
    database="pruebaEMDB"

)

output=open('EpitopeListUniprot.csv', 'w')

with open('/Users/paolanunez/Documents/HELLO/epitope_full_v3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
        writer = csv.writer(output, delimiter=',')
        Epitope=row[2]
        type_epitope=row[1]
        
        length=len(Epitope)
        if 'Linear peptide'in type_epitope:
            if '+' in Epitope:
                pass
            else:
                writer.writerow([type_epitope,Epitope,str(length)])

        