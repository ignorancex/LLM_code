import random
import datasets

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'code', 'math']


def load_code(cache_dir='cache', sub_dir='python', split='train'):
    dataset = datasets.load_dataset(f'code_search_net', sub_dir, split=split, cache_dir=cache_dir)
    datas = []
    for item in dataset:
        datas.append(item['whole_func_string'])
        if len(datas) > 5000:
            break
    return datas


def load_math(cache_dir='cache', sub_dir='python', split='train'):
    import os 
    import json
    dataset = []
    path = "/n/holylabs/LABS/barak_lab/Lab/hanlinzhang/casl/yf/few-shot-learning-llama-master/data/math/test"
    subjects = ['algebra'] # , 'counting_and_probability', 'geometry', 'number_theory', 'precalculus', 'prealgebra', 'intermediate_algebra'] # DEBUG
    for subject in subjects:
        data_path = os.path.join(path, subject)
        json_files = os.listdir(data_path)
        selected_files = random.sample(json_files, 1000)
        for file in selected_files:
            file_path = os.path.join(data_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                datum = data['problem'] + ' ' + data['solution']
                dataset.append(datum)
    return dataset

def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = f'{cache_dir}/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
    
if __name__ == '__main__':
    def strip_newlines(text):
        return ' '.join(text.split())
    cache_dir = 'cache'
    dataset = 'writing'
    data = load(dataset, cache_dir)
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]