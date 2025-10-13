import json
import os
from json import JSONDecodeError

import pandas as pd
from tqdm.auto import tqdm


OUTPUT_PATH = '../run_vqa/b2t_vqa_output'

presence_dict = {
    'yes': 1.0,
    'no': 0.0
}

if os.path.exists(os.path.join(OUTPUT_PATH, 'all_outputs.feather')):
    df = pd.read_feather(os.path.join(OUTPUT_PATH, 'all_outputs.feather'))
else:
    df_list = []
    tc_dirs = sorted([d for d in os.listdir(OUTPUT_PATH) if not d.endswith('.feather')])
    for target_class in tqdm(tc_dirs):
        for keyword in sorted(os.listdir(os.path.join(OUTPUT_PATH, target_class))):
            records = []
            for file in sorted(os.listdir((os.path.join(OUTPUT_PATH, target_class, keyword)))):
                with open(os.path.join(OUTPUT_PATH, target_class, keyword, file), 'r') as f:
                    try:
                        record = json.load(f)
                    except JSONDecodeError:
                        print(os.path.join(OUTPUT_PATH, target_class, keyword, file))
                        continue
                record['image'] = file.split('.')[0] + '.JPEG'
                record['clean answer'] = record['raw answer'].lower().strip().split(' ')[0].replace('.', '').replace(',', '')
                try:
                    record['present'] = presence_dict[record['clean answer']]
                except KeyError:
                    continue
                records.append(record)

            df = pd.DataFrame.from_records(records)
            df['keyword'] = keyword
            df['target class'] = target_class
            df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)
    df.to_feather(os.path.join(OUTPUT_PATH, 'all_outputs.feather'))

df.to_feather('b2t_vqa_answers.feather')
