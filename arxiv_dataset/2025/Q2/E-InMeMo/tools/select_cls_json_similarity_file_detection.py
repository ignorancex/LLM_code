import json


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def read_txt_file(file_path):
    label_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            id_, label = line.strip().split('__')
            label_dict[id_] = label
    return label_dict


def filter_json_data(json_data, label_dict):
    filtered_data = {}
    for key, values in json_data.items():
        key_label = label_dict.get(key.split('_')[0] + '_' + key.split('_')[1])
        if key_label:
            filtered_values = [val for val in values if val.split('_')[0] + '_' + val.split('_')[1] in label_dict and label_dict[val.split('_')[0] + '_' + val.split('_')[1]] == key_label]
            if filtered_values:
                filtered_data[key] = filtered_values
    return filtered_data


def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

for split in ['train', 'val']:
    json_file_path = f'pascal-5i/VOC2012/features_vit-dinov2_fmlr_{split}_detection/top_all-similarity.json'
    txt_file_path = 'evaluate/splits/pascal/trn/all_trn_data4det_class.txt'
    output_json_file_path = f'pascal-5i/VOC2012/features_vit-dinov2_fmlr_{split}_detection/folder_cls_top_all-similarity.json'

    json_data = read_json_file(json_file_path)
    label_dict = read_txt_file(txt_file_path)
    filtered_json_data = filter_json_data(json_data, label_dict)
    save_json_file(filtered_json_data, output_json_file_path)

print("Complete the generation of class-based json file.")
