import pandas as pd 
from tqdm import tqdm
from openai import OpenAI

data = pd.read_csv('../data/comb_df_gpt_labels.csv', index_col=0)

client = OpenAI(
    api_key = 'your_api_key',
  organization='your_org_id',
)



def get_response(prefix: str, statement: str, label: str):

    prompt = prefix + ' ' + statement
    print(prompt)
    response =  client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": prompt}
    ]
    ).model_dump()


    response = response['choices'][0]['message']['content']
    if label == 'topic':
        response = response.lower()
        if 'cultural' in response or 'economic' in response:
            label = 'cultural' if 'cultural' in response else 'economic'
            return label
        else:
            return pd.NA

    elif label == 'politics':
        response = response.lower()
        if 'right' in response or 'left' in response:
            label = 'right' if 'right' in response else 'left'
            return label
        else:
            return pd.NA
    else: 
        return response
    


var_names =  ['topic_label_gpt',
'pol_label_gpt',
'pol_opposite_gpt',
'pol_reformulation_gpt']

prefixes = ['Please indicate whether the following statement is about economic or cultural issues by returning "economic" or "cultural".', 
'Please indicate whether the following statement is attributable to the right or left side of the political spectrum by returning "right" or "left".',
'The following statement is attributable to the right or left side of the political spectrum. Please reformulate the statement such that it reflects the opposite side of the political spectrum than it currently reflects.',
'Please reformulate the following statement such that the meaning of the statement does not change, but the wording does.']

labels = ['topic', 'politics', None, None]


tqdm.pandas()

for i in range(4):
    data[var_names[i]]  = data['statement'].progress_apply(lambda x: get_response(prefixes[i], x, labels[i]))    

data.to_csv("../data/comb_df_gpt_labels.csv")