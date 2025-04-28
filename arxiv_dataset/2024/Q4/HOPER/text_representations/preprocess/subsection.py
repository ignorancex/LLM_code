#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm

#removing PubMed references in the text; removing parentheses, dots and spaces in order
#breakpoint()
yaml_file_path=os.getcwd()                                                  #upload yaml file
stream = open(os.path.join(yaml_file_path,'Hoper_representation_generetor.yaml'), 'r')
data = yaml.safe_load(stream)
#os.path.join(yaml_file_path, 'text_representations/preprocess/data/subsections')   
def removing_parentheses():      #removing parentheses including PubMed


   files_par=os.listdir(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_subsections") )     #output files including PubMed subsections
   #breakpoint()
   for i in tqdm(files_par):                                           
     old_file_par=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_subsections" , i), "r")
     
     os.makedirs(os.path.join(yaml_file_path,"text_representations/preprocess/data/uniprot_par"), exist_ok=True)
    
     new_file_par=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_par" , i), "w")
     data1 = old_file_par.read()
     data2=(data1.split("\n\n"))
     for j in range(0, len(data2)):
       words_count = len(data2[j].split())      #counting words in every paragraph
       split_words = data2[j].split()           #splitting the words in the paragraphs
       for i in range(0, words_count):
           if "PubMed:" in split_words[i]:
               if ")" in split_words[i]:
                  split_words[i]= "."          
               else:
                   split_words[i]=""

           new_file_par.write(split_words[i]+ " ")
       new_file_par.write("\n\n")
   path_uniprot_par=os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_par")    
   removing_dots(path_uniprot_par)    



def removing_dots(path_uniprot_par):     #removing dots
   #breakpoint()
   files_dot=os.listdir(path_uniprot_par)       #files saved from the removing_parantheses() function
   
   os.makedirs(os.path.join(yaml_file_path,"text_representations/preprocess/data/uniprot_dot"), exist_ok=True)
   for j in (files_dot):
      old_file_dot=open(os.path.join(path_uniprot_par, j), "r")
      #breakpoint()   
      new_file_dot=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_dot" ,j), "w")
      for line in old_file_dot:
         new_file_dot.write(line.replace(' .', '.'))
         
   removing_spaces()
   
         
def removing_spaces():     #removing spaces

   #os.makedirs(yaml_file_path + "/text_representations/preprocess/data/uniprot_dot", exist_ok=True)
   os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_space"), exist_ok=True)

   for jk in os.listdir(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_dot")):
      #breakpoint()
      old_file_space=open(os.path.join(yaml_file_path,"text_representations/preprocess/data/uniprot_dot",jk), "r")
      new_file_space=open(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_space" , jk), "w")
      for line in old_file_space:
          new_file_space.write(line.replace('  ', ''))         


def main():
   removing_parentheses()       
         
