#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm
import pdb
import sys
yaml_file_path = os.getcwd()

sys.path.append(os.path.join(yaml_file_path, "text_representations/preprocess"))
#os.chdir(yaml_file_path + "/text_representations/preprocess/")
os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/pubmed_new_abstracts"), exist_ok=True)
error=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/pubmed_new_abstracts","error_abstract_human_2.txt"), "w")
#path = os.path.dirname(os.getcwd()) +  "/text_representations/preprocess/"
#if "preprocess_data" not in os.listdir(path):
#        os.makedirs(path + "/training", exist_ok=True)
#        os.makedirs(path + "/test", exist_ok=True)
yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(os.path.join(yaml_file_path,'Hoper_representation_generetor.yaml'), 'r')

data = yaml.safe_load(stream)

def extract_relevant_info_from_uniprot(uniprot_record):
    files=os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids")
    os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids"), exist_ok=True)
    # print(record)
    try:
       
         if os.path.isfile(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids" ,uniprot_record.id + '.txt'))==False:
            
            newfile = open(os.path.join(yaml_file_path, "text_representations/preprocess/data/human_pubmed_ids" ,uniprot_record.id + '.txt'),'w')
            #print(record.id)
            len_of_ids=len(uniprot_record.annotations["references"])
            
            for i in range(0,len_of_ids):
              
               ref = uniprot_record.annotations['references'][i]
               if (ref.pubmed_id)!="":
                           
                 newfile.write(str(ref.pubmed_id) + "\n")
                   
    except Exception as e:
        error.write(uniprot_record.id)
        error.write("\n")
        error.write(str(e))
        error.write("\n\n")
        # print(record.id)
def main():
    
    with gzip.open(data["parameters"]["uniprot_dir"], 'rb') as handle:
    # file=handle.read()
    # for record in SeqIO.parse(file, "uniprot-xml"):
        for record in tqdm(SeqIO.UniprotIO.UniprotIterator(handle)): 
           if (record.annotations["organism"]=="Homo sapiens (Human)"):
             extract_relevant_info_from_uniprot(record)