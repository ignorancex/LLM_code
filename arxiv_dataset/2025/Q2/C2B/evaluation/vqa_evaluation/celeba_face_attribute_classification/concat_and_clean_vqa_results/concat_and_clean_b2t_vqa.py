import json
import os

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
    kw_dirs = sorted([d for d in os.listdir(OUTPUT_PATH) if not d.endswith('.feather')])

    for keyword in tqdm(kw_dirs):
        records = []
        for file in sorted(os.listdir((os.path.join(OUTPUT_PATH, keyword)))):
            with open(os.path.join(OUTPUT_PATH, keyword, file), 'r') as f:
                record = json.load(f)
            record['image'] = str(int(file.split('.')[0]) + 1) + '.jpg'
            record['clean answer'] = record['raw answer'].lower().strip().split(' ')[0].replace('.', '').replace(',', '')
            record['present'] = presence_dict[record['clean answer']]
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df['keyword'] = keyword
        df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)
    df.to_feather(os.path.join(OUTPUT_PATH, 'all_outputs.feather'))

df.to_feather('b2t_vqa_answers.feather')
