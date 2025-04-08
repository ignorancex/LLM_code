import json

merged_data = []

file_list = [
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_0_1000.json',
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_1000_2000.json',
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_2000_3000.json',
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_3000_4000.json',
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_4000_5000.json',
    './gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_5000_6000.json',
]

for file_name in file_list:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        merged_data.extend(data)

with open('./gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct_0_6000.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("JSON 文件合并完成")
