# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse
from agentic_reason.config import parse_args
from agentic_reason.data_loader import load_dataset
from agentic_reason.cache import CacheManager
from agentic_reason.models import initialize_model, get_output_dir
from agentic_reason.search import process_search_query
from agentic_reason.utils import parse_steps, extract_between, replace_recent_steps
from agentic_reason.generation import generate_webpage_to_reasonchain_batch, run_generation
from agentic_reason.prompt_manager import get_instruction_and_prompt, prepare_input_prompts

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tools.run_code import code_agent
from tools.run_search import search_agent
from nano_graphrag import GraphRAG, QueryParam

from agentic_ds import agentic_ds

from tools.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from evaluate import (
    run_evaluation, 
    extract_answer
)
from agentic_reason.config import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_CODE_QUERY,
    END_CODE_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    BEGIN_CODE_RESULT,
    END_CODE_RESULT,
    BEGIN_MIND_MAP_QUERY,
    END_MIND_MAP_QUERY,
    BEGIN_MIND_MAP_RESULT,
    END_MIND_MAP_RESULT,
)
from agentic_reason.utils import extract_reasoning_context

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key


    # ---------------------- Caching Mechanism ----------------------
      # Initialize cache manager
    cache_manager = CacheManager('./cache')
    search_cache = cache_manager.search_cache
    url_cache = cache_manager.url_cache
    search_cache_path = cache_manager.search_cache_path
    url_cache_path = cache_manager.url_cache_path
    # ---------------------- Set Max Tokens ----------------------
    max_tokens = 8192

    # ---------------------- Model Loading ----------------------
    llm, tokenizer = initialize_model(args)
    # ---------------------- Data Loading ----------------------
    filtered_data = []
    if args.dataset_name:
        filtered_data = load_dataset(args.dataset_name, args.split)
        dataset_name = args.dataset_name
    else:# Get user input
        print("\nEnter your query (or 'quit' to exit):")
        query = input("> ")
        filtered_data.append({'Question': query})
        dataset_name = 'gpqa'


    # ---------------------- Preparation of Input Prompts ----------------------
        # Prepare input prompts and active sequences
    input_list, active_sequences = prepare_input_prompts(
        filtered_data=filtered_data,
        dataset_name=dataset_name,
        model_path=args.model_path,
        tokenizer=tokenizer,
        max_search_limit=args.max_search_limit,
        subset_num=args.subset_num
    )

    if args.dataset_name:
        output_name = args.dataset_name
    else:
        output_name = 'interactive'
    output_dir = get_output_dir(args, output_name, args.max_search_limit, args.top_k)

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # define tools
    code_tool = code_agent(model_name=args.remote_model)
    search_tool = search_agent(llm, tokenizer, bing_subscription_key, bing_endpoint, top_k = args.top_k, use_jina = use_jina, jina_api_key = jina_api_key, max_doc_len = max_doc_len, max_tokens = max_tokens, coherent = True, MAX_SEARCH_LIMIT = MAX_SEARCH_LIMIT, MAX_TURN = MAX_TURN)

    if args.deep_research:
        tools = [code_tool, search_tool]
        agentic_ds(topic=query, model_name=args.remote_model, search_engine=args.search_engine, tools=tools)
        return
    # Main loop until all sequences are finished or maximum turns reached
    while True:
        # Identify sequences that need generation
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]
        print("sequences_needing_generation: ", sequences_needing_generation)

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            outputs = run_generation(
                sequences_needing_generation, 
                llm,
                tokenizer,
                temperature,
                top_p,
                top_k_sampling,  # This is top_k
                repetition_penalty,
                max_tokens
            )
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []
            batch_code_results = []
            batch_mind_map_results = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                print("the outputs are: ", out.outputs)
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                                # Extract coding query
                code_query = extract_between(text, BEGIN_CODE_QUERY, END_CODE_QUERY)
                mind_map_query = extract_between(text, BEGIN_MIND_MAP_QUERY, END_MIND_MAP_QUERY)

                # If a search query is present and needs to be executed
                print("the search query is: ", search_query)
                print("the output is: ", seq['output'])

                if args.mind_map:
                    if not args.mind_map_path:
                        raise ValueError("mind_map_path must be provided if mind_map is True")
                    mind_map = GraphRAG(working_dir=args.mind_map_path)
                else:
                    mind_map = None

                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    
                    if args.mind_map:
                        text_to_insert = seq['output'].split(BEGIN_SEARCH_QUERY)[0]
                        mind_map.insert(text_to_insert)

                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                results = bing_web_search(search_query, bing_subscription_key, bing_endpoint, market='en-US', language='en')
                                search_cache[search_query] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = {}
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:top_k]
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        # get reasoning con
                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = extract_reasoning_context(all_reasoning_steps, mind_map=mind_map)

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")


                # If a code query is present and needs to be executed
                elif code_query and seq['output'].rstrip().endswith(END_CODE_QUERY):

                    if args.mind_map:
                        text_to_insert = seq['output'].split(BEGIN_CODE_QUERY)[0]
                        mind_map.insert(text_to_insert)
                                            # get reasoning con
                    all_reasoning_steps = seq['output']
                    all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                    truncated_prev_reasoning = extract_reasoning_context(all_reasoning_steps, mind_map=mind_map)
                
                    code_query = code_query.strip()
                    code_result = code_tool.generate_code(code_query, truncated_prev_reasoning)
                    batch_code_results.append(code_result)

                elif mind_map_query and seq['output'].rstrip().endswith(END_MIND_MAP_QUERY):
                    if args.mind_map:
                        text_to_insert = seq['output'].split(BEGIN_MIND_MAP_QUERY)[0]
                        mind_map.insert(text_to_insert)
                    mind_map_query = mind_map_query.strip()
                    mind_map_result = mind_map.query(mind_map_query)
                    batch_mind_map_results.append(mind_map_result)

                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")
            
            # Batch fetch all URLs at once to optimize speed
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=use_jina,
                        jina_api_key=jina_api_key,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    llm=llm,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    coherent=True,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis, code_result, mind_map_result in zip(batch_sequences, webpage_analyses, batch_code_results, batch_mind_map_results):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    
                    if isinstance(code_result, str):
                        append_text = f"\n\n{BEGIN_CODE_RESULT}{code_result}{END_CODE_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], code_result)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

                    if isinstance(mind_map_result, str):
                        append_text = f"\n\n{BEGIN_MIND_MAP_RESULT}{mind_map_result}{END_MIND_MAP_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], mind_map_result)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break

    total_time = time.time() - start_time

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    if args.dataset_name:
        # Run evaluation
        run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split)

        # ---------------------- Update Search and URL Cache ----------------------
        print('Updating Search and URL Cache...')
        # Load existing caches or initialize empty dictionaries
        if os.path.exists(search_cache_path):
            with open(search_cache_path, 'r', encoding='utf-8') as f:
                search_cache_new = json.load(f)
        else:
            search_cache_new = {}

        if os.path.exists(url_cache_path):
            with open(url_cache_path, 'r', encoding='utf-8') as f:
                url_cache_new = json.load(f)
        else:
            url_cache_new = {}

        search_cache.update(search_cache_new)
        url_cache.update(url_cache_new)

        cache_manager.save_caches()

        print("Process completed.")
    else:
            # Print final output
            print("\nFinal Response:")
            print(output_list[0])

if __name__ == "__main__":
    main()
