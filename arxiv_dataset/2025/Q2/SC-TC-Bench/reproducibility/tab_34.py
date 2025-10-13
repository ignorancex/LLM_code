import pandas as pd
from collections import Counter
import argparse
from utils import set_random_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)   
    args = parser.parse_args()
    set_random_seed(args.seed)

    regional_names = pd.read_csv('source_data/regional_name_and_characteristics.csv')

    firstnames = Counter([x[1:] for x in regional_names['name'].tolist()])
    firstname_dict = []
    for fisrtname in firstnames:
        if firstnames[fisrtname] >= 2:
            firstname_dict.append(fisrtname)

    regional_names.loc[:, 'firstname'] = [x[1:] for x in regional_names['name'].tolist()]
    counts = regional_names.groupby(['firstname', 'decile', 'region']).size().reset_index(name='count')
    valid_combinations = counts[counts['count'] >= 2][['firstname', 'decile', 'region']]
    # Merge back to the original DataFrame to keep only valid rows
    filtered_df = regional_names.merge(valid_combinations, on=['firstname', 'decile', 'region'], how='inner').reset_index(drop=True)
    
    combo = []
    for i in range(len(filtered_df)):
        combo.append(filtered_df['region'][i] + '#' + str(filtered_df['decile'][i]) )
    filtered_df.loc[:, 'combo'] = combo

    for firstname in filtered_df['firstname'].unique():
        tem_df = filtered_df[filtered_df['firstname']==firstname]

        for tem_combo in tem_df['combo'].unique():
            big_name_list = tem_df[tem_df['combo']==tem_combo]['name'].tolist()
            if len(big_name_list) >= 2:
                print(big_name_list)
