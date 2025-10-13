import json
dataset = 'ml-10m-new'
# dataset = 'amz-new'
# model_name = 'chatglm2-6b'
# model_name = 'Qwen-7B-Chat'
model_name = 'Qwen-1_8B-Chat'
# model_name = 'Mistral-7B-Instruct-v0.2'
# model_name = 'vicuna-7b-v1.3'
# model_name = 'vicuna-7b-v1.5'
data_type = 'history'
# data_type = 'item'
framework = 'KAR'
# framework = 'TRAWL'
# framework = 'ONCE'
# framework = 'RLMRec'
all_data = []
lines = open(f'{framework}/{dataset}/{model_name}/warmup_data_{data_type}_0_50000.json').readlines()
all_data.extend(lines)
lines = open(f'{framework}/{dataset}/{model_name}/warmup_data_{data_type}_50000_70000.json').readlines()
all_data.extend(lines)
# lines = open(f'{dataset}/{model_name}/warmup_data_{data_type}_40000_70000.json').readlines()
# all_data.extend(lines)
# lines = open(f'{dataset}/{model_name}/warmup_data_{data_type}_50000_70000.json').readlines()
# all_data.extend(lines)
# all_data = [json.loads(d) for d in all_data]
all_data = [json.dumps(d) for d in all_data]
print(len(all_data))
with open(f'{framework}/{dataset}/{model_name}/warmup_data_{data_type}_withid.json', 'w') as f:
    f.write('\n'.join(all_data))
