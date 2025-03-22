import jsonlines
import numpy as np
import json
def load_jsonl_list(path):
    myfile = open(path, "r")
    jsonl_list = []
    sets = []
    for i in range(9388):
        try:
            myline = myfile.readline()
        except:
            break
        try:
            y = json.loads(myline)
            a = str(y['choices'])
            if (a in sets):
                continue
            sets.append(a)
            jsonl_list.append(y)
        except:
            jsonl_list.append(None)
            print(myline)
            continue
    return jsonl_list

list1 = load_jsonl_list('./causal_prompting/GPT3/gpt3_result_setup_2_short_large_new.jsonl')
print(len(list1))
output_path = '...'
#a = np.array([[[i, ''] for i in list1+list2]])
#np.save(output_path, a)

