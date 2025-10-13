from llms import llm_dict
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from llms_chat_models import HuggingFaceLocalChatModel
from tqdm import tqdm
from datetime import datetime as dt
import jsonlines
from log import get_logger

# Configure logging
logger = get_logger(__name__)

eval_df = pd.read_csv(Path('data/eval_dataset.csv'), index_col=0)

# Load llm results file
if Path('data/llms_results.csv').is_file():
    llms_results = pd.read_csv(Path('data/llms_results.csv'), index_col=0)
else:
    llms_results = pd.DataFrame(columns=['query'])
    llms_results['query'] = eval_df['query']
    llms_results.to_csv('data/llms_results.csv')

# Define prompt for best result selection
best_result_prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful medical coder and expert in the International Classification of Primary Care. You will receive a query and a list of results from an ICPC search engine. Your  task is to select the result that best matches the query. Your response should be a single ICPC code between the XML tags <answer>selected_code</answer>. If there is no result good enough to match the given query, return an empty answer: <answer></answer>."),
    ("user", "Query: {query}\n\nSearch engine results: {search_engine_results}"),
])

best_result_prompt_template_for_o_models = ChatPromptTemplate([
    ("user", "You are a helpful medical coder and expert in the International Classification of Primary Care. You will receive a query and a list of results from an ICPC search engine. Your  task is to select the result that best matches the query. Your response should be a single ICPC code between the XML tags <answer>selected_code</answer>. If there is no result good enough to match the given query, return an empty answer: <answer></answer>."),
    ("user", "Query: {query}\n\nSearch engine results: {search_engine_results}"),
])

best_result_prompt_template_for_gemma_2_models = ChatPromptTemplate([
    ("user", "You are a helpful medical coder and expert in the International Classification of Primary Care. You will receive a query and a list of results from an ICPC search engine. Your  task is to select the result that best matches the query. Your response should be a single ICPC code between the XML tags <answer>selected_code</answer>. If there is no result good enough to match the given query, return an empty answer: <answer></answer>.\n\nQuery: {query}\n\nSearch engine results: {search_engine_results}"),
])


def get_llms_results(
        models_list: list[str],
        top_k_list: list[int],
        overwrite: list[str] = [],
):
    logger.warning("Getting results for LLMs...")
    for llm_model_name in models_list:
        for top_k in top_k_list:
            results_column_name = f"{llm_model_name}-top-{top_k}"
            if results_column_name in llms_results.columns and llm_model_name not in overwrite:
                logger.warning(f"Skipping results for {results_column_name}...")
                continue
            llm = llm_dict[llm_model_name]
            if isinstance(llm, HuggingFaceLocalChatModel):
                llm.load_model()
            llms_results[results_column_name] = None

            seen_queries = []
            results = []
            time_to_results = []
            response_metadatas = []

            # Check if there are already partial results for this model and top k
            if Path(f'data/{llm_model_name}-top-{top_k}.jsonl').is_file():
                logger.warning(f"Loading partial results for {llm_model_name} and top {top_k}...")
                with jsonlines.open(f'data/{llm_model_name}-top-{top_k}.jsonl', mode='r') as reader:
                    for obj in reader:
                        results.append(obj['response'])
                        time_to_results.append(obj['timedelta'])
                        response_metadatas.append(obj['response_metadata'])
                        seen_queries.append(obj['query'])
            else:
                Path(f'data/{llm_model_name}-top-{top_k}.jsonl').touch()

            # 
            for query, search_engine_results in tqdm(
                zip(eval_df['query'].tolist(), eval_df['search_engine_results_top_200'].tolist()),
                total=eval_df.shape[0],
                desc=f"Getting results for {llm_model_name} and top {top_k}..."):

                if query in seen_queries:
                    continue

                # Define prompt
                if llm_model_name in ['o1-mini', 'o1', '03-mini']:
                    prompt_template = best_result_prompt_template_for_o_models

                elif llm_model_name in ['gemma-2-27b-it', 'gemma-3-4b-it']:
                    prompt_template = best_result_prompt_template_for_gemma_2_models

                else:
                    prompt_template = best_result_prompt_template
                
                prompt = prompt_template.invoke({
                    "query": query,
                    "search_engine_results": eval(search_engine_results)[:top_k]
                })
                    
                t0 = dt.now()
                llm_response = llm.invoke(prompt)
                t1 = dt.now()
                time_to_results.append(t1-t0)
                results.append(llm_response.content)
                response_metadatas.append(llm_response.response_metadata)
                with jsonlines.open(f'data/{llm_model_name}-top-{top_k}.jsonl', mode='a') as writer:
                    writer.write({
                        'query': query,
                        'search_engine_results': eval(search_engine_results)[:top_k],
                        'response': llm_response.content,
                        'response_metadata': llm_response.response_metadata if 'gemini' not in llm_model_name else llm_response.usage_metadata,
                        'timedelta': (t1-t0).total_seconds()
                    })

            llms_results[results_column_name] = results
            llms_results[f"{results_column_name}-timedelta"] = time_to_results
            llms_results[f"{results_column_name}-response-metadata"] = response_metadatas
            llms_results.to_csv('data/llms_results.csv')



if __name__ == '__main__':    
    # Get results for all models
    top_results_to_evaluate = [10, 20, 50, 100]
    models_list = list(llm_dict.keys())
    get_llms_results(
        models_list, 
        top_results_to_evaluate,
        overwrite=[
            ]
        )

    # Additional analysis for best performing models
    best_performing_models = [
        'gpt-4.5-preview',
        'gemini-2.0-pro-exp-02-05',
        'DeepSeek-V3',
        'o3-mini',
        'DeepSeek-R1',
        'QwQ-32B',
        'gemini-2.5-pro-exp-03-25'
    ]

    additional_top_k = [
        200
    ]
    get_llms_results(best_performing_models, additional_top_k, overwrite=['gemini-2.0-pro-exp-02-05', 'gemini-2.5-pro-exp-03-25'])