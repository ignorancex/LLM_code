import os
import sys
import json
import random
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

output_path = 'train/train_data.json'
valid_path = output_path.replace('train_data', 'valid_data')
def merge_data(input_path):
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[]')

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    with open(output_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())

    selected_data = []
    for item in data:
        tmp = {}
        if item.get('good_questions') == None:
            continue
        if len(item['good_questions']) >= 1:
            tmp['instruction'] = f'Determine whether the following question is too simple to be enhanced:\nQuestion: {item["question"]}\nChoices: {item["choices"]}'
            tmp['input'] = ''
            tmp['output'] = 'Yes'
        elif len(item['good_questions']) < 1:
            tmp['instruction'] = f'Determine whether the following question is too simple to be enhanced:\nQuestion: {item["question"]}\nChoices: {item["choices"]}'
            tmp['input'] = ''
            tmp['output'] = 'No'

        selected_data.append(tmp)

    train_data.extend(selected_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_data, indent=4, ensure_ascii=False))

def split_data():
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    random.shuffle(data)
    valid_data = data[:int(len(data)*0.1)]
    train_data = data[int(len(data)*0.1):]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_data, indent=4, ensure_ascii=False))

    with open(valid_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(valid_data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    dir_path = 'enhanced/gpt-4o-mini'
    for file in os.listdir(dir_path):
        if file.endswith('.json'):
            merge_data(os.path.join(dir_path, file))
    split_data()