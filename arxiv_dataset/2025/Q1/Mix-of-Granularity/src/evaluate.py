import os
import json
import argparse
import re

from eval_utils import QADataset, locate_answer, locate_answer4pub_llama
from sklearn.metrics import accuracy_score
import numpy as np
import statistics


def evaluate(
    dataset, save_dir, split="test", locate_fun=locate_answer, test_limit=None
):
    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}

    if test_limit is not None:
        total_len = min(test_limit, len(dataset))
    else:
        total_len = len(dataset)

    # for i, fpath in enumerate(sorted([f for f in os.listdir(save_dir) if f.endswith(".json")])[:total_len]):

    total_cnt = 0
    no_answer_count = 0
    for q_idx in range(min(len(dataset), total_len)):
        fpath = os.path.join(save_dir, split + "_" + dataset.index[q_idx] + ".json")
        answers = []
        try:
            json_object = json.load(open(fpath))
        except:
            print("---> Error encoutnered in this file:")
            print(fpath)

        for it in json_object[:1]:
            if isinstance(it, list):
                it = "".join(it)
            answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        answers = [ans for ans in answers if ans != "NA"]
        for ans in answers:
            if ans == "No":
                no_answer_count += 1
            total_cnt += 1
        if len(answers) == 0:
            pred.append(-1)
            continue
        ans = statistics.mode(answers)
        if ans in answer_list:
            pred.append(answer_list.index(ans))
        else:
            pred.append(-1)

    print("no answer rate: ", no_answer_count / total_cnt)

    truth = [answer2idx[item["answer"]] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[: len(pred)]
        flag = True

    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1 - acc) / len(truth))
    return acc, std, flag


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="OpenAI/gpt-35-turbo-16k")
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--corpus_name", type=str, default="textbooks")  # "medcorp"
    parser.add_argument("--retriever_name", type=str, default="bm25")  # "RRF-4"
    parser.add_argument("--results_dir", type=str, default="./prediction")
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--exp_option", type=str, default=None)

    args = parser.parse_args()

    llm_name = args.llm_name
    rag = False if args.rag is None else True
    k = args.k
    corpus_name = args.corpus_name
    retriever_name = args.retriever_name
    results_dir = args.results_dir
    test_limit = args.test_limit

    dataset_names = ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]
    datasets = {key: QADataset(key) for key in dataset_names}

    scores = []
    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="")
        split = "test"
        if dataset_name == "medmcqa":
            split = "dev"
        if rag:
            if args.exp_option == "None":
                save_dir = os.path.join(results_dir, dataset_name, "rag_"+str(k), llm_name, corpus_name, retriever_name)
            else:
                save_dir = os.path.join(results_dir, dataset_name, f"{args.exp_option}_router_rag_"+str(k), llm_name, corpus_name, retriever_name)
        
    
        else:
            save_dir = os.path.join(results_dir, dataset_name, "cot", llm_name)
        if os.path.exists(save_dir):
            if "pmc_llama" in llm_name.lower():
                acc, std, flag = evaluate(
                    datasets[dataset_name],
                    save_dir,
                    split,
                    locate_answer4pub_llama,
                    test_limit=test_limit,
                )
            else:
                acc, std, flag = evaluate(
                    datasets[dataset_name], save_dir, split, test_limit=test_limit
                )
            scores.append(acc)
            print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
            if flag:
                print(" (NOT COMPLETED)")
            else:
                print("")
        else:
            print(f"NOT STARTED. The path [{save_dir}] does not exist.")
            # scores.append(0)

    if len(scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))
