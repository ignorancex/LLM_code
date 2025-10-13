import json
dataset = 'ml-10m-new'
# dataset = 'amz-new'
# model_name = 'chatglm2-6b'
# model_name = 'Qwen-7B-Chat'
model_name = 'vicuna-7b-v1.3'
data_type = 'history'
# data_type = 'item'
# framework = 'TRAWL'
# framework = 'ONCE'
framework = 'RLMRec'
# knowledge_type = 'p0.0_k1'
knowledge_type = 'p0.1_k2'
all_data = []
lines = open(f'../data/{dataset}/knowledge/{framework}/{model_name}/recent_{data_type}_ans_0_50000_{knowledge_type}.json').readlines()
all_data.extend(lines)
lines = open(f'../data/{dataset}/knowledge/{framework}/{model_name}/recent_{data_type}_ans_50000_70000_{knowledge_type}.json').readlines()
all_data.extend(lines)
# lines = open(f'{dataset}/{model_name}/warmup_data_{data_type}_40000_70000.json').readlines()
# all_data.extend(lines)
# lines = open(f'{dataset}/{model_name}/warmup_data_{data_type}_50000_70000.json').readlines()
# all_data.extend(lines)
all_data = [json.loads(d) for d in all_data]
all_data = [json.dumps(d) for d in all_data]
print(len(all_data))
with open(f'../data/{dataset}/knowledge/{framework}/{model_name}/recent_{data_type}_ans_0_70000_{knowledge_type}.json', 'w') as f:
    f.write('\n'.join(all_data))
