import json
llm_model = 'vicuna-7b-v1.3'
dataset = 'amz-new'
# dataset = 'ml-10m-new'
framework = 'KAR'
# framework = 'ONCE'
# framework = 'TRAWL'
# framework = 'RLMRec'
relaxed = True
# relaxed = False
if dataset == 'amz-new' and relaxed == True:
    src_file_name = 'recent_item_ans_0_40000_p0.1_k2.json'
    tgt_file_name = 'recent_item_ans_0_80000_p0.1_k2.json'
elif dataset == 'amz-new' and relaxed == False:
    src_file_name = 'recent_item_ans_0_40000_p0.0_k1.json'
    tgt_file_name = 'recent_item_ans_0_80000_p0.0_k1.json'
elif dataset == 'ml-10m-new' and relaxed == True:
    src_file_name = 'recent_item_ans_0_10000_p0.1_k2.json'
    tgt_file_name = 'recent_item_ans_0_20000_p0.1_k2.json'
elif dataset == 'ml-10m-new' and relaxed == False:
    src_file_name = 'recent_item_ans_0_10000_p0.0_k1.json'
    tgt_file_name = 'recent_item_ans_0_20000_p0.0_k1.json'
else:
    raise NotImplementedError

warmup_dir = f'../decoding/warmup_data/{framework}/{dataset}/{llm_model}/warmup_data_item_withid.json'
src_dir = f'../data/{dataset}/knowledge/{framework}/{llm_model}/{src_file_name}'
target_dir = f'../data/{dataset}/knowledge/{framework}/{llm_model}/{tgt_file_name}'

all_data = []
lines = open(warmup_dir, 'r').readlines()
all_data.extend(lines)
lines = open(src_dir, 'r').readlines()
all_data.extend(lines)
all_data = [line.strip() for line in all_data]
# all_data = [json.loads(d) for d in all_data]
# all_data = [json.dumps(d) for d in all_data]
print(len(all_data))
with open(target_dir, 'w') as f:
    f.write('\n'.join(all_data))
