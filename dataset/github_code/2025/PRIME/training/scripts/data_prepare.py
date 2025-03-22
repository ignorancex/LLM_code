import os

from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm

short_system_prompt="""
When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.

[ASSESS]

[ADVANCE]

[VERIFY]

[SIMPLIFY]

[SYNTHESIZE]

[PIVOT]

[OUTPUT]

You should strictly follow the format below:

[ACTION NAME]

# Your action step 1

# Your action step 2

# Your action step 3

...

Next action: [NEXT ACTION NAME]

"""

task2ability={'Coding':'code',
              'Math':'math'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str, default='PRIME-RL/Eurus-2-RL-Data')
    parser.add_argument('--output', type=str, default='dataset/prime')
    args=parser.parse_args()

    input_dataset = load_dataset(args.input)
    for split in ['train','validation']:
        output_dataset=[]
        cur_dataset = input_dataset[split]
        for data_entry in tqdm(cur_dataset):
            cur_data = {
                "data_source": data_entry['source'],
                "prompt": [
                    {
                        "role": "system",
                        "content": short_system_prompt
                    },
                    {
                        "role": "user",
                        "content": data_entry['instruction']
                    }],
                "ability": task2ability[data_entry['task']],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry['reference'],
                },
                "extra_info": {
                    'split': 'dummy',
                    'index': 0
                }
            }
            output_dataset.append(cur_data)
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
