"""
Organize the results in a json to facilitate the manual screening

Zijie 20 May 2024
"""

# LIBRARIES
import sys

root_path = "***"
sys.path.insert(0, f"{root_path}/MoG/src")
import os
import json
from tqdm import tqdm
from eval_utils import locate_answer_new as locate_answer
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import numpy as np
import argparse

# PATHS
retriever = "bm25"

qa_dataset_list = ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]
# llm_list = ["glm", "gpt", "internlm", "llama3", "qwen"]
llm_list = ["qwen"]  ## temp change

parser = argparse.ArgumentParser(description="Prediction_job_parser")
parser.add_argument("--test_limit", type=str, default=None)

parser.add_argument("--exp_num", type=int, default=None)
parser.add_argument("--cot_exp_num", type=int, default=None)
parser.add_argument("--rag_k", type=int, default=None)
parser.add_argument("--corpus", type=str, default="textbooks_5")
parser.add_argument(
    "--FALL_BACK_NO",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--FALL_BACK_NOT_RELEVANT",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--FALL_BACK_NO_SNIP",
    action="store_true",
    default=False,
)
args = parser.parse_args()

exp_num = str(args.exp_num)
cot_exp_num = str(args.cot_exp_num)
corpus = args.corpus
rag_k = args.rag_k

test_limit = args.test_limit
FALL_BACK_NO = args.FALL_BACK_NO
FALL_BACK_NOT_RELEVANT = args.FALL_BACK_NOT_RELEVANT
FALL_BACK_NO_SNIP = args.FALL_BACK_NO_SNIP

fall_back_list = []
if FALL_BACK_NO:
    fall_back_list.append("no_ans")
if FALL_BACK_NOT_RELEVANT:
    fall_back_list.append("not_rel")
if FALL_BACK_NO_SNIP:
    fall_back_list.append("no_snip")

if len(fall_back_list) == 0:
    fall_back_list = "no_fallback"
else:
    fall_back_list = "-".join(fall_back_list)


def is_not_relevant(
    ans,
    sw=[
        "not mentioned",
        "not relevant",
        "not mention",
        "no mention",
        "no specific mention",
        "not provide",
        "not among",
        "not related",
        "no relevant information",
        "irrelevant information",
        "not included",
        "no information provided",
        "information missing",
        "not available",
        "not applicable",
        "not include",
        "no details",
        "no specific details",
        "information not found",
        "no reference",
        "no related content",
        "unrelated information",
        "no useful information",
        "nothing relevant",
        "not address",
        "not cover",
        "not addressed",
        "not discussed",
        "not found",
        "information absent",
        "no pertinent information",
        "no data provided",
        "no evidence",
        "no mention of relevance",
        "missing information",
        "no relevant content",
        "not contain relevant information",
        "not contain",
        "not clear",
        "not sure",
        "not determine",
        "not specify",
        "not a source of information",
    ],
):
    # Add regex patterns for phrases
    patterns = [
        re.compile(r"none of .* is .* mentioned", re.IGNORECASE),
        re.compile(r"none of .* are .* mentioned", re.IGNORECASE),
    ]

    # sw: stop words
    for w in sw:
        if w in ans:
            return True
    for pattern in patterns:
        if pattern.search(ans):
            return True
    return False


# ans = """{
#             "step_by_step_thinking": "The quadruple marker test is described as a test that measures maternal serum AFP, hCG, unconjugated estriol, and dimeric inhibin. The text does not mention ss-hCG, but it does mention hCG, which is likely a typo. Therefore, the correct answer is B. ss-hCG.",
#             "answer_choice": "B"
#         }"""
# print("is not relevant?")
# print(is_not_relevant(ans))
# quit()


# Helper function to load json files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Helper function to save jsonl files
def save_jsonl(data, file_path):
    with open(file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item, indent=4) + "\n")


# Create a figure with 5x5 subplots
fig, axs = plt.subplots(5, 5, figsize=(20, 20))

# Adjust vertical spacing
fig.subplots_adjust(hspace=0.8)

acc_dict = {}
for i1 in tqdm(range(len(llm_list)), desc="llm list"):
    llm = llm_list[i1]
    print(f"Processing llm {llm}")
    llm_acc_dict = {}
    for j1 in tqdm(range(len(qa_dataset_list)), desc="qa_dataset list"):
        qa_dataset = qa_dataset_list[j1]
        print(f"Processing qa_dataset {qa_dataset}")

        # DEFINE THE PATHS
        if qa_dataset == "medmcqa":
            splitter = "dev"
        else:
            splitter = "test"
        cot_path = f"{root_path}/MoG/prediction_results/exp{cot_exp_num}/{qa_dataset}/cot/{llm}/"
        # under cot_path are all dev/test_*.josnl files of final prediction
        mog_path = f"{root_path}/MoG/prediction_results/exp{exp_num}/{qa_dataset}/exp{exp_num}_router_rag_{rag_k}/{llm}/{corpus}/{retriever}/"
        # under mog_path are all dev/test_*.josnl files + folders for each prediction result of final prediction

        bm_path = f"{root_path}/MoG/eval/benchmark.json"  # benchmark path

        jsonl_path = f"{root_path}/MoG/visualization/exp{exp_num}_{fall_back_list}/res_organized/exp{exp_num}_{llm}_{qa_dataset}_res.jsonl"
        png_path = f"{root_path}/MoG/visualization/exp{exp_num}_{fall_back_list}/exp{exp_num}_flag_plot.png"
        eval_res_path = f"{root_path}/MoG/visualization/exp{exp_num}_{fall_back_list}/exp{exp_num}_eval_after_viz.json"

        # Load benchmark.json
        with open(bm_path, "r") as f:
            benchmark = json.load(f)
        print("Benchmark file loaded.")
        benchmark = benchmark[qa_dataset]
        print(f"{len(benchmark)} samples in {qa_dataset}")
        print(sorted(list(benchmark.keys()))[0])
        print(benchmark[sorted(list(benchmark.keys()))[0]])

        # Load the cot_results

        cot_files = [
            os.path.join(cot_path, f)
            for f in os.listdir(cot_path)
            if f.endswith(".json")
        ]
        if test_limit:
            cot_files = cot_files[0:test_limit]
        cot_results = {}
        for i in tqdm(range(len(cot_files)), desc="cot"):
            file = cot_files[i]
            ques_id = (file.split("/")[-1]).split(f"{splitter}_")[-1].split(".json")[0]
            try:
                cot_results[ques_id] = load_json(file)
            except Exception as e:
                print("ques_id_err ", ques_id)
                print("file")
                print(file)
                raise (e)
        print("CoT results loaded.")
        print(sorted(list(cot_results.keys()))[0])
        print(type(cot_results[sorted(list(cot_results.keys()))[0]][0]))
        print(cot_results[sorted(list(cot_results.keys()))[0]][0])

        # Load the mog_results
        mog_files = [
            os.path.join(mog_path, f)
            for f in os.listdir(mog_path)
            if f.endswith(".json")
        ]
        if test_limit:
            mog_files = mog_files[0:test_limit]
        mog_results = {}
        for i in tqdm(range(len(mog_files)), desc="mog"):
            file = mog_files[i]
            ques_id = (file.split("/")[-1]).split(f"{splitter}_")[-1].split(".json")[0]
            mog_results[ques_id] = load_json(file)
        print("MoG results loaded.")
        print(sorted(list(mog_results.keys()))[0])
        print(type(mog_results[sorted(list(mog_results.keys()))[0]][0]))
        print(mog_results[sorted(list(mog_results.keys()))[0]][0])

        # Load the snippets retrieved

        snip_dict = {}
        for i in tqdm(range(len(mog_files)), desc="snippets"):
            file = mog_files[i]
            ques_id = (file.split("/")[-1]).split(f"{splitter}_")[-1].split(".json")[0]
            snip_file = os.path.join(mog_path, f"{splitter}_{ques_id}", "snippets.json")
            # load the
            snip_json = load_json(snip_file)
            snip = snip_json[0][0][0]
            snip_dict[ques_id] = snip
        print("Snippets loaded.")
        print(sorted(list(snip_dict.keys()))[0])
        print(snip_dict[sorted(list(snip_dict.keys()))[0]])

        # Combine the results in one jsonl file
        combined_results = []
        q_id_list = sorted(list(cot_results.keys()))
        for i in tqdm(range(len(q_id_list)), desc="combine"):
            cot_skip_flag, mog_skip_flag = False, False
            q_id = q_id_list[i]
            # read info
            label = benchmark[q_id]["answer"]
            question = benchmark[q_id]["question"]
            snippet = snip_dict[q_id]

            # locate results
            try:
                cot_res = locate_answer(
                    json.loads(cot_results[q_id][0])["answer_choice"]
                ).upper()
            except Exception:
                try:
                    cot_res = locate_answer(str(cot_results[q_id][0]))
                except Exception as e:
                    print(f"CoT result parsing error in {q_id}")
                    cot_skip_flag = True
                    cot_res = cot_results[q_id][0]
            try:
                mog_res = locate_answer(
                    json.loads(mog_results[q_id][0])["answer_choice"]
                ).upper()
                raw_mog_res = json.loads(mog_results[q_id][0])
                if FALL_BACK_NO:
                    if mog_res.lower() == "no":
                        mog_res = cot_res
                if FALL_BACK_NOT_RELEVANT:
                    if is_not_relevant(str(raw_mog_res)):
                        mog_res = cot_res
                if FALL_BACK_NO_SNIP:
                    if "NO_TEXT_RETRIEVED" in snippet:
                        mog_res = cot_res
            except Exception:
                try:
                    mog_res = locate_answer(str(mog_results[q_id][0]))
                    raw_mog_res = str(mog_results[q_id][0])
                    if FALL_BACK_NO:
                        if mog_res.lower() == "no":
                            mog_res = cot_res
                    if FALL_BACK_NOT_RELEVANT:
                        if is_not_relevant(str(raw_mog_res)):
                            mog_res = cot_res
                    if FALL_BACK_NO_SNIP:
                        if "NO_TEXT_RETRIEVED" in snippet:
                            mog_res = cot_res
                except Exception as e:
                    print(f"MoG result parsing error in {q_id}")
                    mog_skip_flag = True
                    mog_res = mog_results[q_id][0]

            improved_flag, degraded_flag, remain_correct, remain_wrong = (
                False,
                False,
                False,
                False,
            )

            if (not cot_skip_flag) and (not mog_skip_flag):
                if cot_res == label and mog_res != label:
                    degraded_flag = True
                if cot_res != label and mog_res == label:
                    improved_flag = True
                if cot_res == label and mog_res == label:
                    remain_correct = True
                if cot_res != label and mog_res != label:
                    remain_wrong = True

            j_res = {
                "q_id": q_id,
                "question": question,
                "label": label,
                "snippets": snippet,
                "cot_err": cot_skip_flag,
                "cot_res": cot_res,
                "mog_err": mog_skip_flag,
                "mog_res": mog_res,
                "improved": improved_flag,
                "degraded": degraded_flag,
                "raw_mog_res": raw_mog_res,
                "remain_correct": remain_correct,
                "remain_wrong": remain_wrong,
            }

            combined_results.append(j_res)

        print("combined_results created.")
        if len(combined_results) > 0:
            print(combined_results[0])

        # Create jsonl_path directory if it does not exist
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        # Save the combined results
        save_jsonl([combined_results], jsonl_path)

        # Count the four flags
        flag_counts = defaultdict(int)
        for result in combined_results:
            if result["improved"]:
                flag_counts["improved"] += 1
            if result["degraded"]:
                flag_counts["degraded"] += 1
            if result["remain_correct"]:
                flag_counts["remain_correct"] += 1
            if result["remain_wrong"]:
                flag_counts["remain_wrong"] += 1
            if result["cot_err"]:
                flag_counts["cot_err"] += 1
            if result["mog_err"]:
                flag_counts["mog_err"] += 1

        # Data for plotting
        categories = [
            "improved",
            "degraded",
            "remain_correct",
            "remain_wrong",
            "cot_err",
            "mog_err",
        ]
        counts = [flag_counts[cat] for cat in categories]

        # Plotting the bar chart
        bar = axs[i1, j1].bar(
            categories,
            counts,
            color=["#A8D08F", "#F4B184", "#8FABDB", "#BFBFBF", "#0D0D0D", "#806000"],
        )

        # Add values on top of the bars
        for rect in bar:
            height = rect.get_height()
            axs[i1, j1].text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        if i1 == len(categories) - 1:
            axs[i1, j1].set_xlabel("Categories", fontsize=20)
        if j1 == 0:
            axs[i1, j1].set_ylabel(
                f"{llm}\nCounts", multialignment="center", fontsize=20
            )
        if i1 == 0:
            axs[i1, j1].set_title(f"{qa_dataset}", fontsize=20)

        # Tilt the x-axis labels
        axs[i1, j1].set_xticklabels(categories, rotation=45, ha="right")

        plt.savefig(png_path)
        print(f"\n[Done] Plot saved in {png_path}")

        # Calculate the acc
        truth = [res["label"] for res in combined_results]
        cot_pred = [res["cot_res"] for res in combined_results]
        cot_acc = (np.array(truth) == np.array(cot_pred)).mean()
        cot_std = np.sqrt(cot_acc * (1 - cot_acc) / len(truth))

        mog_pred = [res["mog_res"] for res in combined_results]
        mog_acc = (np.array(truth) == np.array(mog_pred)).mean()
        mog_std = np.sqrt(mog_acc * (1 - mog_acc) / len(truth))

        llm_acc_dict[f"{qa_dataset}"] = {
            "cot_acc": cot_acc,
            "cot_std": cot_std,
            "mog_acc": mog_acc,
            "mog_std": mog_std,
        }
    acc_dict[f"{llm}"] = llm_acc_dict

fig.suptitle(f"Exp {exp_num} Comparison of MoG with respect to CoT\n", fontsize=24)
plt.tight_layout()
plt.savefig(png_path)
print(f"\n[Done] Plot saved in {png_path}")
plt.close()


# print the acc and std results
def format_floats(d, precision=4):
    if isinstance(d, dict):
        return {k: format_floats(v, precision) for k, v in d.items()}
    elif isinstance(d, float):
        return round(d, precision)
    elif isinstance(d, list):
        return [format_floats(i, precision) for i in d]
    else:
        return d


acc_dict = format_floats(acc_dict)
print("Eval results: ")
print(json.dumps(acc_dict, indent=4))
# also save to file
with open(eval_res_path, "w") as f:
    json.dump(acc_dict, f, indent=4)
