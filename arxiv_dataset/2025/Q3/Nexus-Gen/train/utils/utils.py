import os
import json
from tqdm import tqdm

def read_txt(file_path):
    datas = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            datas.append(line)
    return datas

def read_jsonl(file_path, num_samples=None):
    print(f"reading from {file_path}")
    data_list = []
    samples = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line.strip())
            data_list.append(data)
            samples += 1
            if num_samples is not None and samples >= num_samples:
                break
    print(f"read {len(data_list)} samples")
    return data_list

def save_jsonl(data, file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    print(f'saving to {file_path}')
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in tqdm(data, desc='Saving', total=len(data)):
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
    print(f'saved {len(data)} samples')

