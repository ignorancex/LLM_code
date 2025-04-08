import argparse
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert parquet to json, and shuffle')
    parser.add_argument('--input_parquet', type=str, default='./train-00000-of-00001.parquet', help='path to parquet')
    parser.add_argument('--output_json', type=str, required=True, help='path to output json') # default: gsm8k_train.json
    parser.add_argument('--shuffle', type=bool, default=True, help='whether or not to shuffle')

    args = parser.parse_args()
    file_path = args.input_parquet
    df = pd.read_parquet(file_path)

    df['id'] = df.index
    cols = ['id'] + [col for col in df.columns if col != 'id']
    df = df[cols]

    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    json_data = df.to_json(orient='records', force_ascii=False)

    output_json_path = args.output_json
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    print(f'JSON文件已保存到: {output_json_path}')
