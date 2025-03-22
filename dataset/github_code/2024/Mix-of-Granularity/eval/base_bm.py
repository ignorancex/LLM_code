'''
Unified script for benchmarking

Zijie 9 April 2024

'''    

import subprocess
import argparse
import traceback

parser = argparse.ArgumentParser(description="Benchmark_job_parser")
parser.add_argument("--llm", type=str, default="internlm")
parser.add_argument("--corpus_name", type=str, default="textbooks")
parser.add_argument("--retriever_name", type=str, default="bm25")
parser.add_argument("--prediction_folder_name", type=str, default="prediction")
parser.add_argument("--exp_option", type=str, default=None)
parser.add_argument("--rag_k", type=int, default=3)
parser.add_argument("--do_cot", 
action="store_true", default=False,
                    help="set to inference with cot")
parser.add_argument("--do_rag", 
action="store_true", default=False,
                    help="set to inference with rag")


args = parser.parse_args()

llm, rag_k, corpus_name, retriever_name, prediction_folder_name = args.llm, args.rag_k, args.corpus_name.lower(), args.retriever_name.lower(), args.prediction_folder_name

bash_command_cot = f'python MoG/src/evaluate.py --results_dir {prediction_folder_name} --llm_name {llm} --corpus_name {corpus_name} --retriever_name {retriever_name}'

bash_command_rag = f'python MoG/src/evaluate.py --results_dir {prediction_folder_name} --llm_name {llm} --corpus_name {corpus_name} --retriever_name {retriever_name} --rag --k {rag_k} --exp_option {args.exp_option}'

if args.do_cot:
    print(f"[In progress] Evaluating the llm {llm} in the setting of CoT...\n")
    try:
        output = subprocess.check_output(bash_command_cot, shell=True, executable="/bin/bash")
        print(output.decode())
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
    print("\n[Done] Benchmarking with CoT ends.\n")

if args.do_rag:
    print(f"[In progress] Evaluating the llm {llm} in the setting of RAG...\n")
    try:
        output = subprocess.check_output(bash_command_rag, shell=True, executable="/bin/bash")
        print(output.decode())
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
    print("\n[Done] Benchmarking with RAG ends.\n")