#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm

#extracting the information of the subsections in the General annotation (Comments) part of the xml files of the proteins
#subsections are function, cofactor, subunit, tissue specificity, induction, domain, PTM, disease
#Main function takes an input file handle and returns an iterator giving SeqRecord objects


def extract_relevant_info_from_uniprot(record,new_file_dir,error_file):
  #breakpoint()
    
  try:                         
            new_file = open( os.path.join(new_file_dir, record.id+ '.txt'),'w')
            
                #getting the uniprot id of each protein with the record.id and using it in the file name

            if ("comment_function" in record.annotations.keys()):
                functions = record.annotations['comment_function']      #assigning the information of function subsection to the function variable
                for i in range(0, len(functions)):                      #some subsections have more than one input, so we print their information to the related file in order
                    new_file.write(functions[i] + "\n\n")                

            if ("comment_cofactor" in record.annotations.keys()):
                cofactors = record.annotations['comment_cofactor']
                for i in range(0, len(cofactors)):
                    new_file.write(cofactors[i] + "\n\n")

            if ("comment_subunit" in record.annotations.keys()):
                subunits = record.annotations['comment_subunit']
                for i in range(0, len(subunits)):
                    new_file.write(subunits[i] + "\n\n")

            if ("comment_tissuespecificity" in record.annotations.keys()):
                tissuespecificities = record.annotations['comment_tissuespecificity']
                for i in range(0, len(tissuespecificities)):
                    new_file.write(tissuespecificities[i] + "\n\n")

            if ("comment_induction" in record.annotations.keys()):
                inductions = record.annotations['comment_induction']
                for i in range(0, len(inductions)):
                    new_file.write(inductions[i] + "\n\n")

            if ("comment_domain" in record.annotations.keys()):
                domains = record.annotations['comment_domain']
                for i in range(0, len(domains)):
                    new_file.write(domains[i] + "\n\n")

            if ("comment_PTM" in record.annotations.keys()):
                PTMs = record.annotations['comment_PTM']
                for i in range(0, len(PTMs)):
                    new_file.write(PTMs[i] + "\n\n")

            if ("comment_disease" in record.annotations.keys()):
                diseases = record.annotations['comment_disease']
                for i in range(0, len(diseases)):
                    new_file.write(diseases[0])
                    break               
                              
  except Exception as e:
        error_file.write(record.id)
        error_file.write("\n")
        error_file.write(str(e))
        error_file.write("\n\n")
        
def main():
  yaml_file_path=os.getcwd()
  #upload yaml file
  
  stream = open(os.path.join(yaml_file_path, 'Hoper_representation_generetor.yaml'), 'r') 
  data = yaml.safe_load(stream)

  #breakpoint()
  
  os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data"), exist_ok=True)
  
  os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/uniprot_subsections"), exist_ok=True)
  
  os.makedirs(os.path.join(yaml_file_path, "text_representations/preprocess/data/error_file"), exist_ok=True)
  
  uniprot_dir=data["parameters"]["uniprot_dir"]                              #downloading Uniprot database
  
  new_file_dir=os.path.join(yaml_file_path, 'text_representations/preprocess/data/uniprot_subsections')                  #creating directory to save output files        
  error_file = open( os.path.join(yaml_file_path, 'text_representations/preprocess/data/error_file',"exceptions.txt") ,'w')      #printing the exceptions


  with gzip.open(uniprot_dir, 'rb') as handle:    
    for uniprot_record in tqdm(SeqIO.UniprotIO.UniprotIterator(handle)):
        extract_relevant_info_from_uniprot(uniprot_record,new_file_dir,error_file)
