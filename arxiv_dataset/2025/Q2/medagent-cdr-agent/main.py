import argparse
import random
import json
import csv
import os
from config import OpenaiConfig
from llm_utils import OpenaiEngine
from medagent import MedAgent
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--core_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--num_candidate_cdrs", type=int, default=2)
    parser.add_argument("--anomaly_cdr", action="store_true", help="activate to select cdr based on anomaly detection")
    parser.add_argument("--cdr_config", type=str, default="var_meaning", help='var_meaning, var_only, rule_text')
    parser.add_argument("--irr_info_robustness", action="store_true", help="activate to enhance the robustness against irrelevant information in clinical note")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Data preparation (CHANGE THIS PART FOR OTHER DATASETS)
    clinical_data = []
    with open(os.path.join(args.data_path, 'cdr_bench_merged_diverse_labels.csv'), 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data = {
                'id': row['id'],
                'note': row['note'],
                'cdr_label': row['diverse_labels']
            }
            clinical_data.append(data)

    # Openai settings
    openai_config = OpenaiConfig(args.core_model)
    if openai_config["api_key"] == "<YOUR_API_KEY>":
        raise Exception(
            f"Please set your GPT API key first.")
    else:
        core_model = OpenaiEngine(openai_config)

    # Initialize agent
    agent = MedAgent(
        core_model=core_model,
        cdr_info_path='cdr_info_w_io.json',
        silence=False,
        num_candidate_cdrs=args.num_candidate_cdrs,
        anomaly_cdr=args.anomaly_cdr,
        irr_info_robustness=args.irr_info_robustness,
        cdr_config=args.cdr_config
    )

    # Test agent on the dataset
    logs = []
    for data in clinical_data:
        decisions = agent.process(data['note'])
        logs.append({
            'id': data['id'],
            'clinical notes': data['note'],
            'cdr label': data['cdr_label'],
            'decision ground truth': None,
            'clinical decision': decisions
        })
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
    if args.anomaly_cdr:
        logs_file = os.path.join(args.logs_path, 'log_{}_{}_{}.json'.format(args.core_model, args.cdr_config, datetime.now().strftime('%Y%m%d%H%M%S')))
    else:
        logs_file = os.path.join(args.logs_path,
                                 'log_{}_{}cdrs_{}_{}.json'.format(args.core_model, args.num_candidate_cdrs,
                                                                   args.cdr_config,
                                                                   datetime.now().strftime('%Y%m%d%H%M%S')))
    with open(logs_file, 'w') as file:
        json.dump(logs, file, indent=4)


if __name__ == "__main__":
    main()
