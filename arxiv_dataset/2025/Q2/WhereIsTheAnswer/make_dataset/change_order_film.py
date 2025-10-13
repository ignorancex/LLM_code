import json
import pdb
import copy

data = [json.loads(line) for line in open("dataset/film_all.jsonl")]
for index in range(2, 6):
    with open(f"dataset/film_insert_{index}.jsonl", "w") as f:
        for line in data:
            new_line = copy.deepcopy(line)
            ## if the number of sentences is less than index,
            ## swap with the last sentence.
            if len(new_line["text"]) >= index:
                swap_id = index - 1
            else:
                swap_id = -1
            new_line["text"][swap_id] = line["text"][0]
            new_line["text"][0] = line["text"][swap_id]
            new_dict = {
                "title": line["title"],
                "genre": line["genre"],
                "text": new_line["text"],
            }
            json.dump(new_dict, f)
            f.write("\n")
