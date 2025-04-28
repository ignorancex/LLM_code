import os
#import glob
#import tqdm
#from sklearn.utils import shuffle

#import pickle
#mport imghdr
import sys

import subprocess
command = (f'pip install pyyaml')
os.system(command)
import yaml
#import pandas as pd

# Get the path of the conda executable from the system
conda_path = subprocess.getoutput("which conda")
# If the conda path is found
if conda_path:
    # Derive the base Anaconda path from the found conda path
    anaconda_base_path = os.path.dirname(os.path.dirname(conda_path))
    
    # Construct the path to conda.sh
    conda_sh_path = os.path.join(anaconda_base_path, 'etc', 'profile.d', 'conda.sh')

# upload yaml file
path = os.getcwd()
#sys.path.append(path + "/case_study/bin/")
 
stream = open(os.path.join(path, "Hoper_representation_generetor.yaml"), "r")
data = yaml.safe_load(stream)
module_name=data["parameters"]["module_name"]

isDirected =False
if "PPI" in data["parameters"]["choice_of_module"] :
  
  
  edge_f=data["parameters"]["interaction_data_path"][0]
  protein_id=data["parameters"]["protein_id_list"][0]
  node2vec_parameters=data["parameters"]["node2vec_module"]["parameter_selection"]
  d=node2vec_parameters["d"] 
  p=node2vec_parameters["p"]
  q=node2vec_parameters["q"]

  
  from ppi_representations import Node2vec
  from ppi_representations import HOPE
  if "Node2vec" in data["parameters"]["choice_of_representation_name"]:
    #d = ','.join(map(str, node2vec_parameters["d"]))
    #p = ','.join(map(str, node2vec_parameters["p"]))
    #q = ','.join(map(str, node2vec_parameters["q"]))
    command = f'bash -c "source {conda_sh_path} && conda activate hoper_PPI && conda info --envs && python ./ppi_representations/Node2vec.py {edge_f} {protein_id} {isDirected} {d} {p} {q}"'
    os.system(command)
  if "HOPE" in  data["parameters"]["choice_of_representation_name"]:
    #HOPE.hope_repesentation_call(edge_f,protein_id,isDirected,data["parameters"]["HOPE_module"]["parameter_selection"]["d"],data["parameters"]["HOPE_module"]["parameter_selection"]["beta"])
    command = f'bash -c "source {conda_sh_path} && conda activate hoper_PPI && conda info --envs && python ./ppi_representations/HOPE.py {edge_f} {protein_id} {isDirected} {data["parameters"]["HOPE_module"]["parameter_selection"]["d"]} {data["parameters"]["HOPE_module"]["parameter_selection"]["beta"]} "'
    os.system(command)
if "text" in data["parameters"]["choice_of_module"] :

  if "generate" in data["parameters"]["choice_of_process"]:
    choice_of_representation_type = data["parameters"]["generate_module"]["choice_of_representation_type"][0]
    uniprot_files_path = data["parameters"]["generate_module"]["uniprot_files_path"][0]
    pubmed_files_path = data["parameters"]["generate_module"]["pubmed_files_path"][0]
    model_download = data["parameters"]["generate_module"]["model_download"]
    command = (
    f'bash -c "source {conda_sh_path} && '
    f'conda activate HOPER_textrepresentations && '
    f'conda info --envs && '
    f'pip install sent2vec && '
    f'pip install -U nltk && '
    f'python -m nltk.downloader stopwords && '
    f'pip install fasttext && '
    f'python text_representations/representation_generation/createtextrep.py '
    f'--{choice_of_representation_type} -upfp {uniprot_files_path} -pmfp {pubmed_files_path} -mdw {model_download}"'
)
    
    #command = f'bash -c "source {conda_sh_path} && conda activate HOPER_textrepresentations && conda info --envs && pip install sent2vec && pip install -U nltk && python -m nltk.downloader stopwords && pip install fasttext && python text_representations/representation_generation/createtextrep.py --' + data["parameters"]["generate_module"]["choice_of_representation_type"][0] + ' -upfp ' + data["parameters"]["generate_module"]["uniprot_files_path"][0] + ' -pmfp ' + data["parameters"]["generate_module"]["pubmed_files_path"][0] + ' -mdw ' + data["parameters"]["generate_module"]["model_download"]"'
    os.system(command)
  if "visualize" in  data["parameters"]["choice_of_process"]:
    choice_of_visualization_type = data["parameters"]["visualize_module"]["choice_of_visualization_type"][0]
    result_files_path = data["parameters"]["visualize_module"]["result_files_path"][0]

# Construct the command
    command = (
    f'bash -c "source {conda_sh_path} && '
    f'conda activate HOPER_textrepresentations && '
    f'conda info --envs && '
    f'pip install sent2vec && '
    f'pip install -U nltk && '
    f'python -m nltk.downloader stopwords && '
    f'pip install fasttext && '
    f'python text_representations/result_visualization/visualize_results.py '
    f'-{choice_of_visualization_type} -rfp {result_files_path}"'
)
   
if "Preprocessing" in data["parameters"]["choice_of_module"] :
  
  command = f'bash -c "source {conda_sh_path} && conda activate hoper_preprocess && conda info --envs && python ./text_representations/preprocess/preprocess_main.py "'
  os.system(command)
  
if "fuse_representations" in data["parameters"]["choice_of_module"]:
  from utils import fuse_representation
  command = f'bash -c "source {conda_sh_path} && conda activate hoper_case_study_env && conda info --envs && python ./utils/fuse_representation.py {data["parameters"]["representation_files"]} {data["parameters"]["min_fold_number"]} {data["parameters"]["representation_names"] }"'   
  representation_df=os.system(command)
  

if "SimpleAe" in data["parameters"]["choice_of_module"] :

  #from multimodal_representations import simple_ae
  command = f'bash -c "source {conda_sh_path} && conda activate HoloProtRep-AE && conda info --envs && python ./multimodal_representations/simple_ae.py {data["parameters"]["representation_path"]} "'
  os.system(command)
    
  
