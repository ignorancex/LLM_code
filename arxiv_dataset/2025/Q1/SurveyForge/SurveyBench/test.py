import requests
import json
import os
import time
import re
import argparse
from datetime import datetime
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate citation coverage for surveys')
    
    # Evaluation settings
    parser.add_argument('--is_human_eval', 
                    action='store_true',
                    help='True for human survey evaluation, False for generated surveys')

    parser.add_argument('--num_generations', type=int, default=1,
                        help='Number of generated surveys per topic')
    
    # Path settings
    parser.add_argument('--generated_surveys_ref_dir', type=str, default='./generated_surveys_ref',
                        help='Directory path to generated surveys')
    parser.add_argument('--benchmark_refs_dir', type=str, default='./ref_bench',
                        help='Directory path to benchmark references')
    parser.add_argument('--human_surveys_ref_dir', type=str, default='./human_written_ref',
                        help='Directory path to human written surveys')
    parser.add_argument('--topic_list_path', type=str, default='topics.txt',
                        help='Path to topics list file')
    
    config = parser.parse_args()
    return config

def parse_arxiv_date(arxiv_id):
    """
    Parse date and sequence number from arXiv ID
    Returns: tuple of (datetime, int) or (None, None) if parsing fails
    """
    pattern_match = re.match(r'(\d{2})(\d{2})\.(\d{4,5})', arxiv_id)
    if pattern_match:
        year, month, seq_number = pattern_match.groups()
        try:
            paper_date = datetime.strptime(f"20{year}-{month}", "%Y-%m")
            return paper_date, int(seq_number)
        except ValueError:
            return None, None
    return None, None

def compute_citation_coverage(target_refs, benchmark_refs):
    """
    Compute citation coverage between target references and benchmark references
    Args:
        target_refs: List of target reference IDs to evaluate
        benchmark_refs: List of benchmark reference sets
    Returns:
        tuple: (citations_count, coverage_ratio, matched_reference_ids)
    """

    # Process target references
    target_paper_dates = {}
    for paper_id in target_refs:
        clean_paper_id = re.sub(r'v\d+$', '', paper_id)
        date, seq_num = parse_arxiv_date(clean_paper_id)
        if date is not None:
            target_paper_dates[clean_paper_id] = (date, seq_num)

    # Process benchmark references
    benchmark_paper_dates = {}
    for ref_set in benchmark_refs:
        for paper_id in ref_set:
            clean_paper_id = re.sub(r'v\d+$', '', paper_id)
            date, seq_num = parse_arxiv_date(clean_paper_id)
            if date is not None:
                benchmark_paper_dates[clean_paper_id] = (date, seq_num)

    latest_bench_date, latest_bench_seq = max(benchmark_paper_dates.values(), key=lambda x: (x[0], x[1]))

    # Filter target papers by date criteria
    valid_target_ids = {
        paper_id for paper_id, (date, seq_num) in target_paper_dates.items() 
        if (date < latest_bench_date) or (date == latest_bench_date and seq_num < latest_bench_seq)
    }

    # Calculate coverage statistics
    matched_paper_ids = valid_target_ids.intersection(benchmark_paper_dates.keys())
    citation_count = len(matched_paper_ids)
    total_papers = len(valid_target_ids)
    coverage_ratio = citation_count / total_papers if total_papers > 0 else 0
    return citation_count, coverage_ratio, matched_paper_ids

def evaluate_domain_references(domain_name, survey_title, config):
    """
    Evaluate references for a given domain
    Returns: tuple of (citation_count, coverage_ratio, matched_paper_ids)
    """
    # Load benchmark references
    bench_file_path = os.path.join(config.benchmark_refs_dir, f"{domain_name}_bench.json")
    with open(bench_file_path, 'r', encoding='utf') as f:
        benchmark_data = [json.load(f)]

    if config.is_human_eval:
        human_file_path = os.path.join(config.human_surveys_ref_dir, f"{survey_title}.json")
        with open(human_file_path, "r") as f:
            human_refs = json.load(f)
        return compute_citation_coverage(human_refs.keys(), [refs.keys() for refs in benchmark_data])
    
    # Process auto-generated evaluations
    total_citation_count = total_coverage_ratio = 0
    matched_papers_list = []
    for exp_num in range(1, config.num_generations + 1):
        refs_file_path = os.path.join(config.generated_surveys_ref_dir, domain_name, f"exp_{exp_num}/", "ref.json")
        with open(refs_file_path, "r") as f:
            generated_refs = json.load(f)
        citations, coverage, matched = compute_citation_coverage(
            generated_refs.keys(), 
            [refs.keys() for refs in benchmark_data]
        )
        total_citation_count += citations
        total_coverage_ratio += coverage
        matched_papers_list.append(matched)
    
    avg_citation_count = total_citation_count / config.num_generations
    avg_coverage_ratio = total_coverage_ratio / config.num_generations
    return avg_citation_count, avg_coverage_ratio, matched_papers_list

def get_survey_title_mapping():
    """Return mapping of topics to human-written survey titles"""
    return {
        "3D Gaussian Splatting": "A Survey on 3D Gaussian Splatting",
        "3D Object Detection in Autonomous Driving": "3D Object Detection for Autonomous Driving: A Comprehensive Survey",
        "Evaluation of Large Language Models": "A Survey on Evaluation of Large Language Models",
        "LLM-based Multi-Agent": "A survey on large language model based autonomous agents",
        "Generative Diffusion Models": "A survey on generative diffusion models",
        "Graph Neural Networks": "Graph neural networks: Taxonomy, advances, and trends",
        "Hallucination in Large Language Models": "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models",
        "Multimodal Large Language Models": "A Survey on Multimodal Large Language Models",
        "Retrieval-Augmented Generation for Large Language Models": "Retrieval-augmented generation for large language models: A survey",
        "Vision Transformers": "A survey of visual transformers"
    }

def main():
    # Parse arguments
    config = parse_args()
    
    # Get survey titles mapping
    survey_titles = get_survey_title_mapping()

    # Load research topics
    with open(config.topic_list_path, "r") as f:
        research_topics = [line.strip() for line in f if line.strip()]

    # Evaluate each domain
    coverage_ratios = []
    for topic in research_topics:
        _, coverage_ratio, _ = evaluate_domain_references(
            topic, 
            survey_titles[topic],
            config
        )
        coverage_ratios.append(coverage_ratio)

    # Print results
    for topic, ratio in zip(research_topics, coverage_ratios):
        print(f"{topic} citation coverage: {round(ratio, 3)}")
    print(f"Average Coverage Across Topics: {np.mean([round(x, 3) for x in coverage_ratios])}")

if __name__ == "__main__":
    main()
