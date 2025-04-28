import pandas as pd
import os
path = os.path.join(os.getcwd(), "utils") 
from utils import RepresentationFusion

def make_fuse_representation(representation_files,min_fold_num,representation_names):
  representation_file_list=[]
    #path = os.path.dirname(os.getcwd())
  for rep_file in representation_files:
    directory, rep_file_name = os.path.split(rep_file)
    print("loading " + rep_file_name + "...")
    representation_file_list.append(
                    pd.read_csv(rep_file)
                )
  if min_fold_num!="None" and min_fold_num>0 and min_fold_num<=len(
                    representation_file_list
                ):
    min_fold_number=min_fold_num
  else:
    min_fold_number = len(
                    representation_file_list
                )
  
    #import pdb
    #pdb.set_trace()
  representation_dataframe = (
                RepresentationFusion.produce_fused_representations(
                    representation_file_list,
                    min_fold_number,
                    representation_names,
                )
            )
    #breakpoint()
  if "data" not in os.listdir(os.path.dirname(os.getcwd())):
    
    os.makedirs("./data", exist_ok=True)
  
  fuse_representation_path=os.path.join("data","_".join(
                [str(representation) for representation in representation_names])+"_binary_fused_representations_dataframe_multi_col.csv")
  pd.DataFrame(representation_dataframe).to_csv(
                fuse_representation_path,
                index=False,
            )
  return representation_dataframe

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python script.py param1 param2 param3")
    else:
        param1 = sys.argv[1]
        param2 = sys.argv[2]
        param3 = sys.argv[3]
        representation_df=make_fuse_representation(param1,param2,param3)
  
        
   