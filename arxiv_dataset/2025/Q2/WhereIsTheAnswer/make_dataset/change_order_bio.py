import json
import pdb
import copy

data = [json.loads(line) for line in open("dataset/synth_bio_data.jsonl")]
for index in [3, 5, 7, 9]:
    with open(f"dataset/synth_bio_data_insert_{index}.jsonl", "w") as f:
        for line in data:
            new_line = copy.deepcopy(line)
            new_line["data_list"][index - 1] = line["data_list"][0]
            new_line["data_list"][0] = line["data_list"][index - 1]
            new_dict = {
                "name": line["name"],
                "description": "".join(new_line["data_list"]),
                "data_list": new_line["data_list"],
            }
            json.dump(new_dict, f)
            f.write("\n")
