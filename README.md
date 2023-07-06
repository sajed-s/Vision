# Genomic Analysiss
import dendropy 
from dendropy.interop import genbank
import pandas as pd
import numpy as np

#######importing the text file of gene id######

# opening the file in read mode
my_file = open("E:\python\FMD\GI_A.txt", "r")
data = my_file.read()
GIlist = data.replace('\n', ' ').split(" ")
my_file.close()
data_part1=[]
data_part2=[]
#data=pd.DataFrame(GIlist)
#Duo to the lenth of data I had to split it into two lists
data_part1=GIlist[0:200]
data_part2=GIlist[200:]
############extracting the date################
gene_id=[]
date=[]
organism=[]
title=[]
journal=[]
pubmed_id=[]
gb_dna = genbank.GenBankDna(data_part1)
for gb_rec in gb_dna:
	gene_id.append(gb_rec.gi)
	date.append(gb_rec.create_date)
	organism.append(gb_rec.organism)
	for ref in gb_rec.references:
		pubmed_id.append(ref.pubmed_id)
		journal.append(ref.journal)
		title.append(ref.title)
gb_dna = genbank.GenBankDna(data_part2)
for gb_rec in gb_dna:
	gene_id.append(gb_rec.gi)
	date.append(gb_rec.create_date)
	organism.append(gb_rec.organism)
	for ref in gb_rec.references:
		pubmed_id.append(ref.pubmed_id)
		journal.append(ref.journal)
		title.append(ref.title)
data_id_org_date=[]
data_pubmed_journal_title=[]
data_id_org_date=pd.DataFrame({'gene_id':gene_id, 'organism':organism,'date':date})
data_pubmed_journal_title=pd.DataFrame({'journal':journal,'pubmed_id':pubmed_id,'title':title})
data_id_org_date.to_csv('data_id_org_date_a.csv')
data_pubmed_journal_title.to_csv('data_pubmed_journal_a.csv')

