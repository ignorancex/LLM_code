import json
from tqdm import tqdm
import os
import json
# import seaborn as sns
import os
from soar.preprocess import get_dataset
import pickle
import copy
import numpy as np
from soar.llm_utils import merge_results, to_formated_list, merged_dic_results
from soar.llm_utils import get_number_of_solved_tasks_bis, get_info
from soar.post_process.filtering import filter_invalid_grid
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str,)
parser.add_argument('--path_folder', type=str)
parser.add_argument('--max_data', type=int, default=3000)
parser.add_argument('--split', type=str, default='train', help="split to process, 'train' or 'val'")
parser.add_argument("--use_repair_data",action=argparse.BooleanOptionalAction ,default=False,help="use_repair_data")
parser.add_argument("--arc2", action=argparse.BooleanOptionalAction, help="arc 2",default=False)

args = parser.parse_args()
train_data, val_data, test_data = get_dataset(args.base_path,arc_2=args.arc2)
data2test=copy.deepcopy(train_data)
data2test.update(val_data)
list_k_train=list(train_data.keys())
list_k_test=list(val_data.keys())
list_k_all=list_k_train+list_k_test

def compute_n_response(data):
    n_response=0
    for k in data.keys():
        n_response+=len(data[k])
    return n_response

def min_response(data):
    min_resp=np.inf
    for k,v in data.items():
        min_resp = min(len(v),min_resp)
    print("min response: ",min_resp)

def check_max_data(data):
    return max([len(v) for _,v in data.items()])

def check_max_data_refinement(data):
    list_len=[]
    for k,v in data.items():
        v_refined=[f for f in v if f["type"]=="refined"]
        list_len.append(len(v_refined))
    return max(list_len)

def keep_refine_data(data):
    data_refined={}
    for k,v in data.items():
        v_refined=[f for f in v if f["type"]=="refined"]
        data_refined[k]=v_refined
    return data_refined


def get_dataset_part(split, folder_path):
    all_data = []
    keep_file = []
    list_file=os.listdir(folder_path)
    if split == 'train':
        for filename in list_file:
            if filename.startswith('train_') and ".pkl" in filename and not "dedup" in filename and not "_sol_repair" in filename and not "_sol.pkl" in filename:
                keep_file.append(filename)
    elif split == 'val':
        for filename in list_file:
            if filename.startswith('val_') and ".pkl" in filename and not "dedup" in filename and not "_sol_repair" in filename and not "_sol.pkl" in filename:
                keep_file.append(filename)
    else:
        raise ValueError("split must be train or val")

    keep_path = keep_file
    for filename in tqdm(keep_path):
        # if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        try: 
            with open(file_path, 'rb') as f:
                data = to_formated_list(pickle.load(f))
                data_m = merge_results(data)
                if "gen-" in file_path:
                    check_file_path = file_path.split("gen-")[1]
                    if "/refinement/" in check_file_path:
                        data_m = keep_refine_data(data_m)

                max_data = check_max_data(data_m)
                data = to_formated_list(data_m)
                print(f"max data for {file_path}: {max_data}")
                all_data += data
        except Exception as e:
            print(f"error with {file_path} \nerror:",e)
    print("==="*10)
    print("stat initial response: ")
    merge_res = merge_results(all_data)
    for k,v in merge_res.items():
        if len(v) > args.max_data:
            
            # sample randomly max_data
            sample_idx = np.random.choice(len(v),args.max_data,replace=False)  
            merge_res[k] = [v[i] for i in sample_idx]

    print("number of solved task for given path:",folder_path)
    get_info(get_number_of_solved_tasks_bis(merge_res,mv=False))
    print("n initial rep: ",compute_n_response(merge_res))
    print("==="*10)
    return merge_res

def rm_transduction_resp(dic_,all_data=None):
    n_rm=0
    for k,v in tqdm(dic_.items()):
        try:
            flag_one =  any([(1,1)==np.shape(i["output"]) for i in all_data[k]["train"]])
        except:
            flag_one = False
        list_rm = []
        for idx in range(len(v)):
            if rm_transduction(v[idx],flag_one):
                list_rm.append(idx)
                n_rm += 1
        for i in list_rm[::-1]:
            del dic_[k][i]
        
    print("number of code removed",n_rm)
    return dic_


def rm_transduction(resp,flag_one = False):
    if max([len(i) for i in resp["code"].split("\n")]) > 200:
        return True
        
    if flag_one:
        def replace_characters(string):
            return string.replace(' ','')

    else:
        def replace_characters(string):
            return string.replace(' ','').replace("[","").replace("]","")
    list_str = [replace_characters(str(i)) for i in resp["predicted_train_output"]+resp["predicted_test_output"]]
    resp_code = replace_characters(resp["code"])
    for i in list_str:
        if i in resp_code:
            return True

    return False


split=args.split

try:
    merge_res = get_dataset_part(split,args.path_folder+"solution/")
except Exception as e:
    print("error with get_dataset_par solution:",e)
    print("check path_folder:",args.path_folder)
    print("===================")
    merge_res={}

if args.use_repair_data:
    print("==="*10)
    print("adding refinement data")
    print("==="*10)
    merge_res_ref = get_dataset_part(split,args.path_folder+"/refinement/")
    print("==="*10)
    print("n repair rep: ", compute_n_response(merge_res_ref))
    merge_res = merged_dic_results(merge_res,merge_res_ref)
    

print("n initial rep: ", compute_n_response(merge_res))
min_response(merge_res)

path_save= args.path_folder+"solution/" +f"{split}_sol.pkl"
if args.use_repair_data:
    path_save= args.path_folder+"solution/" +f"{split}_sol_repair.pkl"

# create dir if not exist
os.makedirs(os.path.dirname(path_save), exist_ok=True)

print("stat after rm invalid grid: ")
filtered_data = filter_invalid_grid(merge_res,data2test)
get_info(get_number_of_solved_tasks_bis(filtered_data,mv=False))
filtered_data = rm_transduction_resp(filtered_data,all_data=data2test)

with open(path_save, 'wb') as f:
    pickle.dump([{'dict_response':filtered_data}], f)
print("saved successfully:",path_save)

get_info(get_number_of_solved_tasks_bis(filtered_data,mv=True))
