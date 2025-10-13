from pathlib import Path
import pandas as pd
from tqdm import tqdm
from get_llms_results import best_result_prompt_template
from llms import llm_dict
import re
from typing import Dict, Optional
import jsonlines
from pydantic import BaseModel

# DEFINE HOW THIS BASELINE WAS COMPUTED
BASELINE_F1SCORE = 0.7994505494505495

# Create relevant datasets
metrics_path = Path('metrics')
metrics_path.mkdir(exist_ok=True)

# Reload the new datasets
results_df = pd.read_csv(Path('data','llms_results.csv'))
results_df = results_df.fillna('')
eval_df = pd.read_csv(Path('data','eval_dataset.csv'), index_col=0)
eval_df['relevant_results'] = eval_df['relevant_results'].fillna('')
llms_results_df = pd.read_csv(Path('data','llms_results.csv'))

# Make sure there are no duplicates in the query column
eval_df = eval_df.drop_duplicates(subset="query")
results_df = results_df.drop_duplicates(subset="query")

# Extract model columns dynamically (excluding metadata and timedelta columns)
model_columns = [col for col in results_df.columns 
                 if "top-" in col and 
                 "-timedelta" not in col and 
                 "-response-metadata" not in col and
                 "tokens" not in col
]

# Top-ks to evaluate
top_ks = [10, 20, 50, 100, 200]

# Models to evaluate
models = [
    'gpt-4o-baseline',
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.5-preview',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4.1',
    'gpt-4.5-preview',
    'o1',
    'o1-mini',
    'o3',
    'o3-mini',
    'o4-mini',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-pro-exp-02-05',
    'gemini-2.5-pro-exp-03-25',
    'gemma-3-4b-it',
    'gemma-3-27b-it',
    'gemma-2-27b-it',
    'DeepSeek-V3',
    'DeepSeek-R1',
    'DeepSeek-R1-Distill-Qwen-1.5B',
    'DeepSeek-R1-Distill-Qwen-7B',
    'Llama-3.3-70B-Instruct',
    'Llama-3.2-3B-Instruct',
    'Llama-3.2-1B-Instruct',
    'Llama-3.1-405B-Instruct',
    'Llama-3.1-70B-Instruct',
    'Llama-3-70B-Instruct',
    'Llama-4-Maverick-Instruct-Basic',
    'Llama-4-Scout-Instruct-Basic',
    'sabia-3',
    'sabiazinho-3',
    'QwQ-32B',
]

# Define helper function to extract models' code selections from xml tags
def extract_answer(text):
    right_answer_format = False
    try:
        match = re.search(r"<answer>(.*?)</answer>", text)
        selected_code = match.group(1)
        right_answer_format = True
        return selected_code, right_answer_format
    except:
        return "", right_answer_format

# Load valid ICPC-2 codes
icpc_df = pd.read_csv(Path('data/icpc-2_partial.csv'), index_col=0)
valid_icpc_codes = icpc_df['code'].to_list()

# Define object to store relevant metadata related to each llm prediction
class EvalResult(BaseModel):
    model_top_k_name: str
    model: str
    top_k: int
    query: str
    prediction: str
    selected_code: str
    empty_answer: bool
    right_answer_format: bool
    is_valid_icpc_2_code: bool
    selected_code_is_in_relevant_codes: bool
    selected_code_is_in_search_results: bool
    no_relevant_code: bool
    relevant_code_is_in_search_results: bool
    relevant_code_indices_in_search_results: Optional[list[int]] = None
    true_positive: bool
    false_positive: bool
    true_negative: bool
    false_negative: bool
    ideal_scn_true_positive: bool
    ideal_scn_false_positive: bool
    ideal_scn_true_negative: bool
    ideal_scn_false_negative: bool

# Define helper functions that evaluate model generations and returns an EvalResult objects
def eval_model_prediction(
        model: str, 
        top_k: int,
        query: str,
        prediction: str, 
        relevant_codes: list[str], 
        top_k_search_results: list[str]) -> EvalResult:
    """
    Evaluate the model prediction against the relevant codes and search results.
    
    Args:
        prediction (str): The predicted code from the model.
        relevant_codes (list[str]): A list of codes that are relevant to the query.
        search_results (list[str]): A list of search results for the query.
        top_k (int): The number of search results to consider.

    Returns:
        dict[bool]: A dictionary containing the evaluation results.
    """
    # Relevant information
    empty_answer = False
    is_valid_icpc_2_code = False
    selected_code_is_in_relevant_codes = False
    selected_code_is_in_search_results = False
    no_relevant_code = False
    relevant_code_is_in_search_results = False
    relevant_code_indices_in_search_results = None

    # Check if there is only one or zero selected code
    selected_code, right_answer_format = extract_answer(prediction)
    if selected_code == "":
        empty_answer = True

    # If there is a code, check if it is a valid ICPC-2 code
    if selected_code in valid_icpc_codes:
        is_valid_icpc_2_code = True

    # If it is valid, check if it is in the relevant codes
    if (
        len(relevant_codes)>0 and 
        empty_answer == False and 
        selected_code in relevant_codes
        ):
        selected_code_is_in_relevant_codes = True
    
    # Check if there are any relevant codes
    if len(relevant_codes) == 0:
        no_relevant_code = True

    # If it is valid, check if it is in the search results
    if selected_code != "" and selected_code in top_k_search_results:
        selected_code_is_in_search_results = True

    # If there is a relevant code, check if it is in the search results
    if len(relevant_codes)>0:
        for code in relevant_codes:
            if code in top_k_search_results:
                relevant_code_is_in_search_results = True
    
    # If there are relevant codes in the search results, store the first occurrence of each relevant code present in the search results
    if relevant_code_is_in_search_results:
        idx = []
        for code in relevant_codes:
            if code in top_k_search_results:
                idx.append(top_k_search_results.index(code))
        relevant_code_indices_in_search_results = idx

    # Contingency table
    true_positive, true_negative, false_positive, false_negative = False, False, False, False
    if empty_answer == False and selected_code_is_in_relevant_codes:
        true_positive = True
    elif empty_answer == True and no_relevant_code == True:
        true_negative = True
    elif (
        (empty_answer == False and no_relevant_code == True) or 
        (empty_answer == False and no_relevant_code == False and selected_code_is_in_relevant_codes == False)):
        false_positive = True
    elif (
        empty_answer == True and no_relevant_code == False
    ):
        false_negative = True

    assert sum([true_positive, true_negative, false_positive, false_negative])==1, "Something wrong with correctness rules..."

    # Contingency table with a perfect retriever scenario in which:
    # - when there is a relevant code to select, it is present in the search results
    # - when there is no relevant code to select, we keep as it is
    ideal_scn_true_positive, ideal_scn_true_negative, ideal_scn_false_positive, ideal_scn_false_negative = False, False, False, False
    if empty_answer == False and selected_code_is_in_relevant_codes:
        ideal_scn_true_positive = True
    elif empty_answer == True and no_relevant_code == True:
        ideal_scn_true_negative = True
    elif (
        (empty_answer == False and no_relevant_code == True) or 
        (empty_answer == False and no_relevant_code == False and selected_code_is_in_relevant_codes == False and relevant_code_is_in_search_results)):
        ideal_scn_false_positive = True
    elif (
        empty_answer == True and no_relevant_code == False
    ):
        ideal_scn_false_negative = True

    assert sum([ideal_scn_true_positive, ideal_scn_true_negative, ideal_scn_false_positive, ideal_scn_false_negative])<=1, "Something wrong with correctness rules..."


    return EvalResult(
        model_top_k_name=f'{model}-top-{top_k}',
        model=model,
        top_k=top_k,
        query=query,
        prediction=prediction,
        selected_code=selected_code,
        empty_answer=empty_answer,
        right_answer_format=right_answer_format,
        is_valid_icpc_2_code=is_valid_icpc_2_code,
        selected_code_is_in_relevant_codes=selected_code_is_in_relevant_codes,
        selected_code_is_in_search_results=selected_code_is_in_search_results,
        no_relevant_code=no_relevant_code,
        relevant_code_is_in_search_results=relevant_code_is_in_search_results,
        relevant_code_indices_in_search_results=relevant_code_indices_in_search_results,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        ideal_scn_true_positive=ideal_scn_true_positive,
        ideal_scn_true_negative=ideal_scn_true_negative,
        ideal_scn_false_positive=ideal_scn_false_positive,
        ideal_scn_false_negative=ideal_scn_false_negative,
    )

def get_query_top_k_codes(query: str, top_k: int):
    top_k_results = eval(eval_df[eval_df['query']==query]['search_engine_results_top_200'].values[0])[:top_k]
    return [r['code'] for r in top_k_results]

def eval_model_predictions(
        model: str, 
        top_k: int,
        ) -> None:
    
    predictions_data = []
    for row in results_df.to_dict('records'):
        relevant_results = eval_df[eval_df['query']==row['query']]['relevant_results'].values[0]
        predictions_data.append(
            eval_model_prediction(
                model=model,
                top_k=top_k,
                query=row['query'],
                prediction=row[f'{model}-top-{top_k}'],
                relevant_codes=relevant_results.split('|') if relevant_results != '' else [],
                top_k_search_results=get_query_top_k_codes(row['query'], top_k)
            ).model_dump()
        )
    
    with jsonlines.open(Path(f'metrics/{model}-top-{top_k}.jsonl'), mode='w') as writer:
        for obj in predictions_data:
            writer.write(obj)


def get_reasoning_tokens(x):
            if (
                'token_usage' in x 
                and 
                'completion_tokens_details' in x['token_usage'] 
                and 
                x['token_usage']['completion_tokens_details'] is not None
                and
                'reasoning_tokens' in x['token_usage']['completion_tokens_details']
                ):
                return x['token_usage']['completion_tokens_details']['reasoning_tokens']
            else:
                return None
            
def extract_token_usage(s: str) -> Dict[str, int]:
    """
    Extracts completion_tokens, prompt_tokens and total_tokens from a string
    containing something like:
      ... ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=324, total_tokens=332) ...

    Returns a dict:
      {
        'completion_tokens': 8,
        'prompt_tokens': 324,
        'total_tokens': 332
      }

    Raises ValueError if the pattern isn’t found.
    """
    
    pattern = re.compile(
        r"completion_tokens\s*=\s*(\d+)\s*,\s*"
        r"prompt_tokens\s*=\s*(\d+)\s*,\s*"
        r"total_tokens\s*=\s*(\d+)"
    )
    m = pattern.search(s)
    if not m:
        return eval(s)
    return {'token_usage':{
        'completion_tokens': int(m.group(1)),
        'prompt_tokens': int(m.group(2)),
        'total_tokens': int(m.group(3)),
    }
    }




if __name__ == '__main__':
    # Get evaluation data from models' generations
    metrics_df = pd.DataFrame()
    for model_name in tqdm(model_columns, desc="Processing models' predictions..."):
        model, top_k = model_name.split('-top-')
        eval_model_predictions(model=model, top_k=int(top_k))

    # Load data into a dataframe
    metrics_df = pd.DataFrame()
    for model_name in tqdm(model_columns, desc="Gathering evaluation data..."):
        model, top_k = model_name.split('-top-')
        metrics_df = pd.concat([metrics_df, pd.read_json(Path(f'metrics/{model}-top-{top_k}.jsonl'), lines=True)], axis=0)

    # save predictions data to disk
    metrics_df.to_csv(Path('metrics/predictions_data.csv'))

    # Compute relevant metrics
    metrics = []
    eps = 1e-16

    for model_name in tqdm(model_columns, desc="Computing relevant metrics..."):
        model, top_k = model_name.split('-top-')
        results = {
            'model': model,
            'top_k': top_k,
            'model_top_k_name': f'{model}-top-{top_k}',
        }

        # Rate of answers with right format
        results['right_answer_format'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')]['right_answer_format'].mean()
        
        # Rate of answers with valid ICPC-2 codes
        results['valid_icpc_2_code_rate'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}') & (metrics_df['empty_answer']==False)]['is_valid_icpc_2_code'].mean()

        # Rate of relevant code present in the search results
        results['relevant_code_is_in_search_results'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')]['relevant_code_is_in_search_results'].mean()

        # Rate of non empty answers in which the selected code was in the search results
        results['selected_code_is_in_search_results'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}') & (metrics_df['empty_answer']==False)]['selected_code_is_in_search_results'].mean()

        # Rate of selected codes that are in the relevant codes considering empty answers and empty relevant codes
        results['selected_code_is_in_relevant_codes'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')]['selected_code_is_in_relevant_codes'].mean()

        # Rate of selected codes that are in the relevant codes considering empty answers and empty relevant codes when the relevant codes are in the search results
        results['selected_code_is_in_relevant_codes_and_in_search_results'] = metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')&(metrics_df['relevant_code_is_in_search_results']==True)]['selected_code_is_in_relevant_codes'].mean()

        # Precision
        true_positives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')
                                        &(metrics_df['true_positive']==True)])
        
        false_positives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')
                                        &(metrics_df['false_positive']==True)])
        
        results['precision'] = true_positives / (true_positives + false_positives + eps)

        # Recall
        false_negatives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')&(metrics_df['false_negative']==True)])
        results['recall'] = true_positives / (true_positives + false_negatives + eps)

        # F1 score
        results['f1_score'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + eps)
        
        # Ideal retriever scenario
        # Precision
        ideal_scn_true_positives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')
                                        &(metrics_df['ideal_scn_true_positive']==True)])
        
        ideal_scn_false_positives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')
                                        &(metrics_df['ideal_scn_false_positive']==True)])
        
        results['ideal_scn_precision'] = ideal_scn_true_positives / (ideal_scn_true_positives + ideal_scn_false_positives + eps)

        # Recall
        ideal_scn_false_negatives = len(metrics_df[(metrics_df['model_top_k_name']==f'{model}-top-{top_k}')&(metrics_df['ideal_scn_false_negative']==True)])
        results['ideal_scn_recall'] = ideal_scn_true_positives / (ideal_scn_true_positives + ideal_scn_false_negatives + eps)

        # F1 score
        results['ideal_scn_f1_score'] = 2 * results['ideal_scn_precision'] * results['ideal_scn_recall'] / (results['ideal_scn_precision'] + results['ideal_scn_recall'] + eps)

        metrics.append(results)


    computed_metrics_df = pd.DataFrame.from_records(metrics).sort_values('f1_score', ascending=False)
    computed_metrics_df.to_csv(Path('metrics/computed_metrics.csv'))

    # Generate latex tables
    # Detailed F1-scores table
    detailed_f1_table = computed_metrics_df
    detailed_f1_table['top_k'] = detailed_f1_table['top_k'].astype(int)
    detailed_f1_table = detailed_f1_table.sort_values(by=['model','top_k'], ascending=True).reset_index(drop=True)
    detailed_f1_table[['model', 'top_k', 'precision', 'recall', 'f1_score']].to_latex(Path('latex_tables','detailed_f1_table.tex'), index=False, float_format="%.4f")

    # Detailed F1-scores table with ideal retriever
    detailed_f1_table = computed_metrics_df
    detailed_f1_table['top_k'] = detailed_f1_table['top_k'].astype(int)
    detailed_f1_table = detailed_f1_table.sort_values(by=['model','top_k'], ascending=True).reset_index(drop=True)
    detailed_f1_table[['model', 'top_k', 'ideal_scn_precision', 'ideal_scn_recall', 'ideal_scn_f1_score']].to_latex(Path('latex_tables','detailed_ideal_f1_table.tex'), index=False, float_format="%.4f")

    # Mean and max F1-scores table
    piv_df = computed_metrics_df.pivot(index='model', columns='top_k', values='f1_score')
    piv_df['mean_f1_score'] = piv_df.apply(lambda x: x.mean(), axis=1)
    piv_df['max_f1_score'] = piv_df.apply(lambda x: x.max(), axis=1)
    summ_piv_df = piv_df.sort_values('max_f1_score', ascending=False)[['mean_f1_score', 'max_f1_score']]
    summ_piv_df.columns = ['Mean F1 score', 'Max F1 score']
    summ_piv_df.index.name = 'Model'
    summ_piv_df.to_latex(Path('latex_tables','llm_summary_f1_scores.tex'), index=True, float_format="%.4f")

    # Mean and max F1-scores table with ideal retriever
    ideal_piv_df = computed_metrics_df.pivot(index='model', columns='top_k', values='ideal_scn_f1_score')
    ideal_piv_df['mean_ideal_f1_score'] = ideal_piv_df.apply(lambda x: x.mean(), axis=1)
    ideal_piv_df['max_ideal_f1_score'] = ideal_piv_df.apply(lambda x: x.max(), axis=1)
    summ_ideal_piv_df = ideal_piv_df.sort_values('max_ideal_f1_score', ascending=False)[['mean_ideal_f1_score', 'max_ideal_f1_score']]
    summ_ideal_piv_df.columns = ['Mean ideal F1 score', 'Max ideal F1 score']
    summ_ideal_piv_df.index.name = 'Model'
    summ_ideal_piv_df.to_latex(Path('latex_tables','llm_summary_ideal_f1_scores.tex'), index=True, float_format="%.4f")

    # Complete metrics summary table
    full_piv_df = computed_metrics_df.pivot(index='model', columns='top_k', values='f1_score')
    full_piv_df['mean_f1_score'] = full_piv_df.apply(lambda x: x.mean(), axis=1)
    full_piv_df['max_f1_score'] = full_piv_df.apply(lambda x: x.max(), axis=1)
    full_piv_df.fillna('').sort_values('max_f1_score', ascending=False).to_latex(Path('latex_tables','llm_summary_full.tex'), index=True, float_format="%.4f", longtable=True)

    # Complete metrics summary table with ideal retriever
    ideal_full_piv_df = computed_metrics_df.pivot(index='model', columns='top_k', values='ideal_scn_f1_score')
    ideal_full_piv_df['mean_f1_score'] = ideal_full_piv_df.apply(lambda x: x.mean(), axis=1)
    ideal_full_piv_df['max_f1_score'] = ideal_full_piv_df.apply(lambda x: x.max(), axis=1)
    ideal_full_piv_df.fillna('').sort_values('max_f1_score', ascending=False).to_latex(Path('latex_tables','llm_summary_full_ideal.tex'), index=True, float_format="%.4f", longtable=True)

    # Extract token usage from each model
    for model in tqdm(models, desc="Gathering models' token usage..."):
        for top_k in top_ks:
            if f'{model}-top-{top_k}-response-metadata' not in llms_results_df.columns: continue
            # get objects in the right format

            if 'token_usage' not in llms_results_df[f'{model}-top-{top_k}-response-metadata'].iloc[0]:
                llms_results_df[f'{model}-top-{top_k}-response-metadata'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(lambda x: {'token_usage': eval(x)} if type(x)==str else {'token_usage': x})
            
            elif 'ChatCompletionOutputUsage' in llms_results_df[f'{model}-top-{top_k}-response-metadata'].iloc[0]:
                llms_results_df[f'{model}-top-{top_k}-response-metadata'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(extract_token_usage)
            
            else:
                llms_results_df[f'{model}-top-{top_k}-response-metadata'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(lambda x: eval(x) if type(x)==str else x)

            if model in ['gemini-2.0-pro-exp-02-05', 'gemini-2.5-pro-exp-03-25', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']:
                # input tokens
                # print(model, top_k, llms_results_df[f'{model}-top-{top_k}-response-metadata'].iloc[0])
                llms_results_df[f'{model}-top-{top_k}-input_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage'].get('input_tokens', None)
                )

                # output tokens
                llms_results_df[f'{model}-top-{top_k}-output_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage'].get('output_tokens', None)
                )

                # total tokens
                llms_results_df[f'{model}-top-{top_k}-total_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage'].get('total_tokens', None)
                )
            else:
                # input tokens
                # print(model, top_k, llms_results_df[f'{model}-top-{top_k}-response-metadata'].iloc[0])
                llms_results_df[f'{model}-top-{top_k}-input_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage']['prompt_tokens']
                )

                # output tokens
                llms_results_df[f'{model}-top-{top_k}-output_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage']['completion_tokens']
                )

                # total tokens
                llms_results_df[f'{model}-top-{top_k}-total_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(
                    lambda x: x['token_usage']['total_tokens']
                )
            
            # reasoning tokens
            llms_results_df[f'{model}-top-{top_k}-reasoning_tokens'] = llms_results_df[f'{model}-top-{top_k}-response-metadata'].apply(get_reasoning_tokens)

    # Update llms results csv
    llms_results_df.to_csv(Path('data','llms_results.csv'), index=False)