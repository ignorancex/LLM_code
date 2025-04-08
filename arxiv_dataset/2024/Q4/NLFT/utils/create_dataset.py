# 用于创建Negative：Positive=1:2的数据集

import json
import random
import os
import argparse

from helper import extract_answer_from_output


def main():
    random.seed(42)
    parser = argparse.ArgumentParser(description='create dataset for SFT, NLFT, ReFT')
    parser.add_argument('--input_json', type=str, required=True, help='path to input json')
    parser.add_argument('--dataset_dir', type=str, required=True, help='path to dataset dir')
    parser.add_argument('--n_total', type=int, default=800, help='total number of data') # 默认 800 条，可设 1200 条
    parser.add_argument('--ratio', type=str, default=None, choices=['correct', '1:3', '1:2', '1:1', '2:1', '3:1', 'wrong', None], help='ratio of correct and wrong') # 对比错的数目比例

    args = parser.parse_args()
    input_file_path = args.input_json
    dataset_dir = args.dataset_dir
    n_total = args.n_total
    ratio = args.ratio

    sft_dir = os.path.join(dataset_dir, 'SFT')
    nlft_dir = os.path.join(dataset_dir, 'NLFT')
    reft_dir = os.path.join(dataset_dir, 'ReFT')
    os.makedirs(sft_dir, exist_ok=True)
    os.makedirs(nlft_dir, exist_ok=True)
    os.makedirs(reft_dir, exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"从 {input_file_path} 加载了 {len(data)} 条数据")

    if ratio is None:
        # 只要顺序取前若干个就可以了，因为已经打乱过
        nlft_style_dataset = data[:n_total]
    else:
        n_correct = 0
        n_wrong = 0
        if ratio == 'correct':
            n_correct, n_wrong = n_total, 0
        elif ratio == 'wrong':
            n_correct, n_wrong = 0, n_total
        else:
            temp_correct_ratio, temp_wrong_ratio = ratio.split(':')
            if n_total % (int(temp_correct_ratio) + int(temp_wrong_ratio)) != 0:
                print('[WARNING] n_total 按照比例除不尽')
            correct_ratio = int(temp_correct_ratio) / (int(temp_correct_ratio) + int(temp_wrong_ratio))
            n_correct = int(round(n_total * correct_ratio, 2))
            n_wrong = n_total - n_correct

        data_with_judgment = [item for item in data if item.get('judgment') is not None]
        data_without_judgment = [item for item in data if item.get('judgment') is None]

        assert n_correct < len(data_without_judgment)
        assert n_wrong < len(data_with_judgment)
        print('正确数目: {}, 错误数目: {}'.format(n_correct, n_wrong))
        selected_data_without_judgment = random.sample(data_without_judgment, n_correct)
        selected_data_with_judgment = random.sample(data_with_judgment, n_wrong)

        nlft_style_dataset = selected_data_without_judgment + selected_data_with_judgment
        assert len(nlft_style_dataset) == n_total
        random.shuffle(nlft_style_dataset)

    def get_sft_style_dataset(dataset):
        return [{"prompt": f"{item['instruction']}\n{item['input']}", "completion": item['answer']} for item in dataset]

    sft_style_dataset = get_sft_style_dataset(nlft_style_dataset)

    def get_reft_style_dataset(dataset):
        return [{"item_id": f"gsm8k_{item['id']}", "question": item['input'], "answer_cot": item['answer'], "answer_value": extract_answer_from_output(item['answer'])} for item in dataset]
    
    reft_style_dataset = get_reft_style_dataset(nlft_style_dataset)

    nlft_filename = "{}NLFT{}.json".format(ratio.replace(':', '_') + '_' if ratio else '', n_total)
    sft_filename = "{}SFT{}.json".format(ratio.replace(':', '_') + '_' if ratio else '', n_total)
    reft_filename = "{}ReFT{}.json".format(ratio.replace(':', '_') + '_' if ratio else '', n_total)

    nlft_path = os.path.join(nlft_dir, nlft_filename)
    sft_path = os.path.join(sft_dir, sft_filename)
    reft_path = os.path.join(reft_dir, reft_filename)
    
    with open(nlft_path, 'w', encoding='utf-8') as nlft_file:
        json.dump(nlft_style_dataset, nlft_file, ensure_ascii=False, indent=4)
    # print(f"已保存NLFT数据到 {nlft_path}")

    with open(sft_path, 'w', encoding='utf-8') as sft_file:
        json.dump(sft_style_dataset, sft_file, ensure_ascii=False, indent=4)
    # print(f"已保存SFT数据到 {sft_path}")

    # 将SFT JSON转换为JSON Lines格式
    jsonl_path = sft_path.replace('.json', '.jsonl')
    try:
        with open(sft_path, 'r', encoding='utf-8') as sft_file, open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            data_sft = json.load(sft_file)
            for item in data_sft:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        # print(f"已将 {sft_path} 转换为 JSON Lines 格式，保存到 {jsonl_path}")
    except Exception as e:
        print(f"错误: 转换 {sft_path} 为 JSON Lines 失败。详细信息: {e}")
    
    with open(reft_path, 'w', encoding='utf-8') as nlft_file:
        json.dump(reft_style_dataset, nlft_file, ensure_ascii=False, indent=4)
    # print(f"已保存ReFT数据到 {reft_path}")

    print("\n所有数据处理完成。")


if __name__ == '__main__':
    main()
