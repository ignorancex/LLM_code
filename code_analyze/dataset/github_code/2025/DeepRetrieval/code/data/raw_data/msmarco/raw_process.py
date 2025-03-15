#!/usr/bin/env python3

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

def read_qrels(qrels_path):
    """
    Read qrels file and return a dictionary mapping query IDs to document IDs.
    
    Args:
        qrels_path (str): Path to the qrels file
        
    Returns:
        dict: Dictionary mapping query IDs to sets of document IDs
    """
    query_to_docs = defaultdict(set)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  # Ensure we have at least 3 columns
                query_id = parts[0]
                doc_id = parts[2]
                query_to_docs[query_id].add(doc_id)
    return query_to_docs

def process_queries(queries_path, query_to_docs, output_path):
    """
    Process queries file and output jsonl where each line is {"question": "xxx", "docs_id": [...]}.
    Only queries that are in query_to_docs dictionary are included.
    
    Args:
        queries_path (str): Path to the queries file
        query_to_docs (dict): Dictionary mapping query IDs to sets of document IDs
        output_path (str): Path to save the processed queries as jsonl
    
    Returns:
        int: Number of queries kept
    """
    kept_count = 0
    with open(queries_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # Ensure we have at least 2 columns
                query_id = parts[0]
                question = parts[1]
                
                if query_id in query_to_docs:
                    # Create JSON object
                    json_obj = {
                        "question": question,
                        "docs_id": list(query_to_docs[query_id])
                    }
                    
                    # Write to output file
                    f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    kept_count += 1
    return kept_count

def main():
    parser = argparse.ArgumentParser(description='Process queries files to keep only those with doc mappings')
    parser.add_argument('--qrels_train', type=str, default='qrels.train.tsv', help='Path to qrels.train.tsv')
    parser.add_argument('--qrels_dev', type=str, default='qrels.dev.tsv', help='Path to qrels.dev.tsv')
    parser.add_argument('--domains', type=str, nargs='+', default=['all', 'health', 'science', 'tech'], 
                        help='List of domains to process (e.g., all health science tech)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for processed files (default: same as input)')
    
    args = parser.parse_args()
    
    # Read qrels to get query IDs with document mappings
    train_query_to_docs = read_qrels(args.qrels_train)
    dev_query_to_docs = read_qrels(args.qrels_dev)
    
    print(f"Found {len(train_query_to_docs)} unique query IDs in {args.qrels_train}")
    print(f"Found {len(dev_query_to_docs)} unique query IDs in {args.qrels_dev}")
    
    # Process each domain
    for domain in args.domains:
        domain_dir = f"msmarco_{domain}"
        
        # Process train queries
        train_path = os.path.join(domain_dir, "queries.train.tsv")
        if not os.path.exists(train_path):
            print(f"Warning: {train_path} does not exist, skipping")
            continue
            
        output_dir = args.output_dir if args.output_dir else domain_dir
        os.makedirs(output_dir, exist_ok=True)
        
        train_output = os.path.join(output_dir, "train.jsonl")
        kept_train = process_queries(train_path, train_query_to_docs, train_output)
        
        # Process dev queries
        dev_path = os.path.join(domain_dir, "queries.dev.tsv")
        if os.path.exists(dev_path):
            dev_output = os.path.join(output_dir, "dev.jsonl")
            kept_dev = process_queries(dev_path, dev_query_to_docs, dev_output)
            print(f"Domain {domain}: Kept {kept_train}/{os.path.getsize(train_path)/1024:.1f}KB train queries and {kept_dev}/{os.path.getsize(dev_path)/1024:.1f}KB dev queries")
        else:
            print(f"Domain {domain}: Kept {kept_train}/{os.path.getsize(train_path)/1024:.1f}KB train queries (dev file not found)")
            
        # Process eval queries
        eval_path = os.path.join(domain_dir, "queries.eval.tsv")
        if os.path.exists(eval_path):
            eval_output = os.path.join(output_dir, "eval.jsonl")
            kept_eval = process_queries(eval_path, train_query_to_docs, eval_output)
            print(f"Domain {domain}: Kept {kept_eval}/{os.path.getsize(eval_path)/1024:.1f}KB eval queries")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()