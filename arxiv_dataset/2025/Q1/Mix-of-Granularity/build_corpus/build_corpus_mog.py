# import libraries
import os
import sys
from tqdm import tqdm

import json

medrag_path = "MoG/"
sys.path.insert(0, medrag_path)
from src.medrag import MedRAG
from eval_utils import QADataset
from config import config

corpus_name = "Wikipedia"
corpus_path = os.path.join(config["db_dir"], corpus_name.lower(), "chunk")
corpus_path_moe = os.path.join(config["db_moe_dir"])

# prepare output paths
half_path = os.path.join(config["db_moe_dir"], "wikipedia_half", "chunk")
single_path = os.path.join(config["db_moe_dir"], "wikipedia_1", "chunk")
double_path = os.path.join(config["db_moe_dir"], "wikipedia_2", "chunk")
quad_path = os.path.join(config["db_moe_dir"], "wikipedia_4", "chunk")
oct_path = os.path.join(config["db_moe_dir"], "wikipedia_8", "chunk")

all_path_list = [
    corpus_path_moe,
    half_path,
    single_path,
    double_path,
    quad_path,
    oct_path,
]
# loop over these paths inside all_path_list
# create the directories if they do not exist
for path in all_path_list:
    if not os.path.exists(path):
        os.makedirs(path)


# define functions
def break_json_objects(json_object):
    # process the json object here
    id = json_object["id"]
    contents = json_object["contents"]
    content = json_object["content"]

    # split the contents into 2 parts with almost equal length
    half = len(content) // 2

    # find the nearest space before the halfway point
    nearest_space = content.rfind(" ", 0, half)
    if nearest_space != -1:
        half = nearest_space
    content_first_half = content[:half]
    content_second_half = content[half:]

    # create new ids for these new json objects
    id_first_half = id + "-0"
    id_second_half = id + "-1"

    # create new json objects
    json_object_first_half = {
        "id": id_first_half,
        "title": json_object["title"],
        "content": content_first_half,
    }
    json_object_first_half["contents"] = (
        json_object_first_half["title"] + "." + json_object_first_half["content"]
    )
    json_object_second_half = {
        "id": id_second_half,
        "title": json_object["title"],
        "content": content_second_half,
    }
    json_object_second_half["contents"] = (
        json_object_second_half["title"] + "." + json_object_second_half["content"]
    )

    return json_object_first_half, json_object_second_half


def merge_json_objects(queue):
    id_list = [json_object["id"] for json_object in queue]
    content_list = [json_object["content"] for json_object in queue]
    new_id = "|".join(id_list)
    new_content = " ".join(content_list)
    json_object_new = {
        "id": new_id,
        "content": new_content,
        "title": queue[0]["title"],
        "contents": queue[0]["title"] + "." + new_content,
    }
    return json_object_new


# start processing
# attention, only keep the fields of "id" and "contents"


# first we generate half and single json objects
# read jsonl files
jsonl_files = [file for file in os.listdir(corpus_path) if file.endswith(".jsonl")]
for i in tqdm(range(len(jsonl_files)), desc="half,1 chunking"):
    file = jsonl_files[i]
    file_path = os.path.join(corpus_path, file)
    file_name = file.split(".")[0]
    half_list, single_list = [], []
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_object = json.loads(line)
            except:
                continue
            json_object_first_half, json_object_second_half = break_json_objects(
                json_object
            )

            # create a new single json object containing only id and contents
            # attention, its id should be the concatenation of the two half objects instead of the original one
            json_object_new = {
                "id": json_object_first_half["id"]
                + "|"
                + json_object_second_half["id"],
                "contents": json_object["contents"],
                "title": json_object["title"],
                "content": json_object["content"],
            }

            # append the new half json objects to the queues
            half_list.append(json_object_first_half)
            half_list.append(json_object_second_half)

            # append the new single json object to the single list
            single_list.append(json_object_new)

    # write the half json objects to the corresponding files
    half_file_path = os.path.join(half_path, file_name + ".jsonl")
    with open(half_file_path, "w") as f:
        for i in range(len(half_list)):
            j_object = half_list[i].copy()
            j_object["id"] = str(i + 1) + "#" + j_object["id"]
            f.write(json.dumps(j_object) + "\n")

    # write the single json objects to the corresponding files
    single_file_path = os.path.join(single_path, file_name + ".jsonl")
    with open(single_file_path, "w") as f:
        for i in range(len(single_list)):
            j_object = single_list[i].copy()
            j_object["id"] = str(i + 1) + "#" + j_object["id"]
            f.write(json.dumps(j_object) + "\n")

# now generate double, quad and oct json objects
# for scalability, we use temporary queues to store the json objects
# intialize the queues
double_q, quad_q, oct_q = [], [], []
double_list, quad_list, oct_list = [], [], []
for i in tqdm(range(len(jsonl_files)), desc="2,4,8 chunking"):
    file = jsonl_files[i]
    file_path = os.path.join(single_path, file)
    file_name = file.split(".")[0]
    with open(file_path, "r") as f:
        for line in f:
            json_object = json.loads(line)
            # for speed, keep only the id and contents filed
            json_id_wo_line_num = json_object["id"].split("#")[1]
            json_object = {
                "id": json_id_wo_line_num,
                "contents": json_object["contents"],
                "title": json_object["title"],
                "content": json_object["content"],
            }

            # append the json object to the queues
            double_q.append(json_object)
            quad_q.append(json_object)
            oct_q.append(json_object)

            # merge the json objects in the queues and write to the corresponding files
            if len(double_q) == 2:
                json_object_new = merge_json_objects(double_q)
                double_list.append(json_object_new)
                double_q = []

            if len(quad_q) == 4:
                json_object_new = merge_json_objects(quad_q)
                quad_list.append(json_object_new)
                quad_q = []

            if len(oct_q) == 8:
                json_object_new = merge_json_objects(oct_q)
                oct_list.append(json_object_new)
                oct_q = []
    # At the end of a file, merge the remaining json objects in the queues
    if len(double_q) > 0:
        json_object_new = merge_json_objects(double_q)
        double_list.append(json_object_new)
        double_q = []

    if len(quad_q) > 0:
        json_object_new = merge_json_objects(quad_q)
        quad_list.append(json_object_new)
        quad_q = []

    if len(oct_q) > 0:
        json_object_new = merge_json_objects(oct_q)
        oct_list.append(json_object_new)
        oct_q = []

    # write the double json objects to the corresponding files
    double_file_path = os.path.join(double_path, file_name + ".jsonl")
    with open(double_file_path, "w") as f:
        for i in range(len(double_list)):
            j_object = double_list[i].copy()
            old_id = j_object["id"]
            j_object["id"] = str(i + 1) + "#" + old_id
            f.write(json.dumps(j_object) + "\n")
    j_object = None
    # write the quad json objects to the corresponding files
    quad_file_path = os.path.join(quad_path, file_name + ".jsonl")
    with open(quad_file_path, "w") as f:
        for i in range(len(quad_list)):
            j_object = quad_list[i].copy()
            old_id = j_object["id"]
            j_object["id"] = str(i + 1) + "#" + old_id
            f.write(json.dumps(j_object) + "\n")
    j_object = None
    # write the oct json objects to the corresponding files
    oct_file_path = os.path.join(oct_path, file_name + ".jsonl")
    with open(oct_file_path, "w") as f:
        for i in range(len(oct_list)):
            j_object = oct_list[i].copy()
            old_id = j_object["id"]
            j_object["id"] = str(i + 1) + "#" + old_id
            f.write(json.dumps(j_object) + "\n")
    j_object = None
