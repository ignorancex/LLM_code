import logging
from collections import defaultdict, Counter
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd
from promptsource.templates import Template
import yaml
import json
from src.common.log_util import prepare_global_logging
from src.preprocessors.empty import EmptyPreprocessor

PROJECT_ROOT = Path.cwd()
if 'scripts' in str(PROJECT_ROOT):
    while 'scripts' in str(PROJECT_ROOT):
        PROJECT_ROOT = PROJECT_ROOT.parent

PROMPT_TASK_NAME_MAPPING = {
    "app_reviews"                            : "AppReviews",
    "imdb"                                   : "IMDB",
    "super_glue/rte"                         : "RTE",
    "super_glue/cb"                          : "CB",
    "super_glue/wic"                         : "WiC",
    "super_glue/copa"                        : "COPA",
    "anli"                                   : "ANLI",
    "aqua_rat/raw"                           : "AQuA",
    "math_qa"                                : "MathQA",
    "sem_eval_2010_task_8"                   : "SemEval2010",
    "BIG-BENCH"                              : "BIG-BENCH",
    "common_gen"                             : "CommonGen",
    "craffel/openai_lambada"                 : "LAMBADA",
    "numer_sense"                            : "NumerSense",
    "multi_x_science_sum"                    : "Multi-XSci",
    "xsum"                                   : "XSum",
    "zest"                                   : "ZEST",
    "adversarial_qa"                         : "AdversarialQA",
    "financial_phrasebank/sentences_allagree": "FinNews",
    "BAREBONES"                              : "No Prompt",
    "craigslist_bargains"                    : "Craigslist",
    "yelp"                                   : "Yelp"
}
GROUP_NAME_MAPPING = {
    "WordsinContext[validation]"              : "WiC",
    "RecognizingTextualEntailment[validation]": "RTE",
    "craigslist_bargains[validation]"         : "Craigslist",
    "anli[dev_r1]"                            : "ANLI R1",
    "anli[dev_r2]"                            : "ANLI R2",
    "anli[dev_r3]"                            : "ANLI R3",
    "CommitmentBank[validation]"              : "CB",
    "AQuA[validation]"                        : "AQuA"
}


def make_latex_table(df):
    ranks = df.pop('Rank')
    rank_best = ranks.idxmax()
    rank_worst = ranks.idxmin()

    best = df.idxmax()
    worst = df.idxmin()
    out = ['\\toprule', '{} & ' + " & ".join(df.columns) + " & Rank\\\\", '\\midrule']
    for idx, row in df.iterrows():
        row_str = f'{idx}'
        for i, v in row.items():
            cell = ""
            if idx != 'No Prompt':
                if best[i] == idx:
                    cell = "\\cellcolor{green!25}"
                elif worst[i] == idx:
                    cell = "\\cellcolor{red!25}"

            val_formated = "{:0.2f}".format(v)
            if idx != "No Prompt" and (i == idx or ((i == "CB" or 'ANLI' in i) and idx == "ANLI")):
                cell += "\\textbf{" + val_formated + "}"
            else:
                cell += val_formated

            row_str += " & " + cell

        rank_val = f"{ranks[idx]:.2f}"
        if idx == rank_best:
            rank_val = "\\cellcolor{red!25}" + rank_val
        elif idx == rank_worst:
            rank_val = "\\cellcolor{green!25}" + rank_val
        row_str += " & " + rank_val
        if idx == 'No Prompt':
            out = out[:2] + ['\\midrule', row_str + "\\\\"] + out[2:]
        else:
            out.append(row_str + "\\\\")
    out.append('\\bottomrule')
    return '\n'.join(out)


def get_rank(df, dont_update_names=False):
    if not dont_update_names:
        df['group'] = df['group'].apply(lambda g: GROUP_NAME_MAPPING[g])
        df['prompt_task'] = df['prompt_task'].apply(lambda g: PROMPT_TASK_NAME_MAPPING[g])
    df = df.drop(df[df['prompt_task'] == "BIG-BENCH"].index)
    df.loc[df['prompt_task'] == 'NumerSense', 'task_mode'] = "COMPLETION"
    df[f'accuracy_rank'] = df.groupby('group')["accuracy"].rank(ascending=False)
    df[f'f1_rank'] = df.groupby('group')["f1"].rank(ascending=False)
    return df


def add_prompt_length(df):
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    processor = EmptyPreprocessor()
    all_templates = yaml.load(
        PROJECT_ROOT.joinpath('prompts', 'general_fixed_choice.yaml').open('r'),
        yaml.Loader
    )['templates']

    data_point = processor({}, 0)
    prompt_token_ids = {
        prompt.id: ['input_ids']
        for prompt in all_templates.values()
    }
    prompt_tokens = {}
    for k, prompt in all_templates.items():
        str_tokens = []
        offset = 0
        prev_had_wordpiece = False
        prev_is_wordpiece = False
        tokens = tokenizer.tokenize(prompt.apply(data_point)[0], add_special_tokens=False)
        prompt_token_ids[k] = tokenizer.convert_tokens_to_ids(tokens)
        for i, t in enumerate(tokens):
            if t.startswith("▁"):
                if t == "▁":
                    offset += 1
                    prev_is_wordpiece = True
                    prev_had_wordpiece = True
                    continue
                else:
                    str_tokens.append(t.replace("▁", ""))
            else:
                if i == 0 or len(str_tokens) == 0 or prev_is_wordpiece:
                    str_tokens.append(t)
                elif "##" in str_tokens[i - offset - 1] or prev_had_wordpiece:
                    str_tokens.append("##" + t)
                else:
                    str_tokens.append(t)

            prev_had_wordpiece = t.startswith("▁") if not prev_is_wordpiece else True
            prev_is_wordpiece = False
        prompt_tokens[k] = str_tokens

    with PROJECT_ROOT.joinpath('data', 'prompt_tokens.json').open('w', encoding='utf-8') as f:

        json.dump(prompt_tokens, f, indent=True)
    unigram_corr_df = []
    bigram_corr_df = []
    trigram_corr_df = []

    median_rank = df.groupby('prompt_id').median()

    for prompt_id, tokens in prompt_token_ids.items():
        try:
            record = {
                "prompt_id": prompt_id,
                "rank"     : median_rank.loc[prompt_id]['accuracy_rank']
            }
        except:
            continue
        bigram_bow = Counter()
        for j in range(1, len(tokens), 2):
            bigram = ' '.join(map(str, [tokens[j - 1], tokens[j]]))
            if not bigram:
                continue
            bigram_bow[bigram] += 1

        trigram_bow = Counter()
        for j in range(2, len(tokens), 3):
            trigram = ' '.join(map(str, [tokens[j - 2], tokens[j - 1], tokens[j]]))
            if not trigram:
                continue
            trigram_bow[trigram] += 1

        unigram_corr_df.append({**Counter(tokens), **record})
        bigram_corr_df.append({**bigram_bow, **record})
        trigram_corr_df.append({**trigram_bow, **record})

    unigram_corr_df = pd.DataFrame.from_records(unigram_corr_df).fillna(0)
    bigram_corr_df = pd.DataFrame.from_records(bigram_corr_df).fillna(0)
    trigram_corr_df = pd.DataFrame.from_records(trigram_corr_df).fillna(0)

    df['prompt_tokens'] = df['prompt_id'].apply(lambda x: len(prompt_token_ids[x]))

    return df


def get_crosstask_data(df, key_use):
    grouped = defaultdict(dict)
    ranks = defaultdict(dict)
    for key, group in df.groupby(["group", "prompt_task"]):
        medians = group[["accuracy", "f1"]].median()
        grouped[key[0]][key[1]] = medians[key_use]
        ranks[key[1]][key[0]] = group[f'{key_use}_rank'].tolist()
    grouped = pd.DataFrame.from_dict(grouped, orient='index').T

    new_ranks = {}
    for k, v in ranks.items():
        k_ranks = [r for i in v.values() for r in i]
        new_ranks[k] = pd.Series(k_ranks).median()

    grouped['Rank'] = pd.Series(new_ranks)

    df.to_csv(PROJECT_ROOT.joinpath('data', f'{key_use}_run_results.csv'), index=False)

    return grouped


def get_diff_crosstask_with_text(df):
    df['group'] = df['group'].apply(lambda g: GROUP_NAME_MAPPING[g])
    df['prompt_task'] = df['prompt_task'].apply(lambda g: PROMPT_TASK_NAME_MAPPING[g])
    df = df.drop(df[df['prompt_task'] == "BIG-BENCH"].index)
    col_save = [
        'prompt_id', 'group', 'run_name', 'prompt_task', 'accuracy', 'f1', 'choices_in_prompt',
        'training_task', 'name'
    ]
    df = df[col_save]

    base = get_rank(df[df['run_name'] == 'CTBase'].copy(),True)
    base['key'] = base['prompt_id'] + "|" + base['group'] + "|" + base['prompt_task']
    no_text = get_rank(df[df['run_name'] == 'CTNoText'].copy(),True)
    no_text['key'] = no_text['prompt_id'] + "|" + no_text['group'] + "|" + no_text['prompt_task']

    joined = base.append(no_text)

    return joined


def create_visualization_data():
    out_path = PROJECT_ROOT.joinpath('data')
    run_csv_path = out_path.joinpath('runs.csv')

    if not PROJECT_ROOT.joinpath('logs').exists():
        PROJECT_ROOT.joinpath('logs').mkdir()
    else:
        # Clear logs
        with PROJECT_ROOT.joinpath('logs', 'create_visualization_data.log').open('w') as _:
            pass

    prepare_global_logging(PROJECT_ROOT.joinpath("logs"), log_name='create_visualization_data')
    logger = logging.getLogger('create_visualization_data')
    summary_keys = [
        line
        for line in out_path.joinpath('summary_keys.txt').read_text('utf-8').splitlines(True)
        if line
    ]

    logger.info(f"{len(summary_keys)} summary keys found")
    runs_df: pd.DataFrame = pd.read_csv(run_csv_path)
    logger.info(f"{len(runs_df)} runs found in {run_csv_path}")
    logger.debug(f"Columns found for runs: {', '.join(runs_df.columns)}")

    baseline_run_names = [
        "T0", "T5", "T0LenNorm", "T5LenNorm"
    ]
    logger.debug(f"Baseline run names: {baseline_run_names}")

    baseline_runs: pd.DataFrame = runs_df[
        runs_df.run_name.str.contains("|".join(baseline_run_names))]
    logger.info(f"{len(baseline_runs)} baseline runs")
    cross_task_data = runs_df[runs_df.run_name == 'CTBase'].copy()
    logger.info(f"{len(cross_task_data.groupby(['name']).size())} unique cross task "
                f"prompts")
    logger.info("Getting Ranks")
    cross_task_data = get_rank(cross_task_data)
    logger.info("Getting Prompt Lengths")
    cross_task_data = add_prompt_length(cross_task_data)

    logger.info("Saving baselines")
    baseline_runs.to_csv(out_path.joinpath('baselines.csv'), index=False)

    logger.info("Saving Crosstask data")
    cross_task_data.to_csv(
        PROJECT_ROOT.joinpath('data', 'cross_task.csv'),
        index=False
    )

    logger.info("Saving latex tables")
    acc_table_path = PROJECT_ROOT.joinpath('data', 'cross_task_accuracy_table.text')
    with acc_table_path.open('w', encoding='utf-8') as f:
        f.write(make_latex_table(get_crosstask_data(
            cross_task_data.copy(),
            'accuracy'
        )))
    f1_table_path = PROJECT_ROOT.joinpath('data', 'cross_task_f1_table.text')
    with f1_table_path.open('w', encoding='utf-8') as f:
        f.write(make_latex_table(get_crosstask_data(
            cross_task_data.copy(),
            'f1'
        )))

    logger.info("Saving differences")
    diff_df = get_diff_crosstask_with_text(
        runs_df[runs_df.run_name.isin(['CTBase', 'CTNoText'])].copy()
    )

    diff_df.to_csv(PROJECT_ROOT.joinpath('data', 'diff.csv'), index=False)


if __name__ == '__main__':

    create_visualization_data()
