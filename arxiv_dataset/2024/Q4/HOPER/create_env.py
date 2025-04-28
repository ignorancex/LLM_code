import os

os.system("conda env create -f case_study/hoper_case_study_env.yml")
os.system("conda env create -f ppi_representations/hoper_PPI.yml")
os.system("conda env create -f text_representations/preprocess/hoper_preprocess.yml")
os.system("conda env create -f text_representations/text_representations.yml")
os.system("conda env create -f multimodal_representations/simple_ae_env.yml")