import os
import glob
from operator import itemgetter
import re
# Initializing list
import argparse
from evaluations.judge_model import load_evaluator

parser = argparse.ArgumentParser()

############ Evaluator model parameters ##########
parser.add_argument(
    "--evaluator-model",
    default="gpt-4-turbo",
    help="Name of evaluator model.",
    choices=["gpt-3.5-turbo", 
                "gpt-4", 
                "gpt-4-turbo", 
                "gpt-4-1106-preview", 
                "no-evaluator"]
)
parser.add_argument(
    "--evaluator-max-n-tokens",
    type = int,
    default = 10,
    help = "Maximum number of tokens for the evaluator."
)
parser.add_argument(
    "--evaluator-temperature",
    type=float,
    default=0,
    help="Temperature to use for evaluator."
)

args = parser.parse_args()


all_output = glob.glob('./results_hyper/s_10.0_e_0.4/*')
file_prefix = './results_hyper/s_10.0_e_0.4'
all_files = [os.path.join(file_prefix, 'res{}.txt'.format(ele)) for ele in range(50)]


def retrieve_question_index_and_response(file):
    f = open(file)
    content = f.readlines()
    question = content[2].split(':')[1].strip()
    content = ' '.join(content)
    content_list = content.split('Iter:')
    content_list_filter = [ele for ele in content_list if 'Generated String:' in ele]
    content_list_index = [int(ele.split(',')[0]) for ele in content_list_filter]
    content_list_response = [ele.split('Generated String: ')[1] for ele in content_list_filter]
    content_list_response = [ele.replace('\r', ' ').replace('\n', ' ') for ele in content_list_response]
    return question, content_list_index, content_list_response

big_check_list = []

for ii, file in enumerate(all_files):
    question, content_list_index, content_list_response = retrieve_question_index_and_response(file)
    if len(content_list_index)>200:
        content_list_index = content_list_index[:200]
        content_list_response = content_list_response[:200]
    args.goal = question
    evaluator_llm = load_evaluator(args)
    question_list_repeat = [question]*len(content_list_response)
    judge_scores = evaluator_llm.judge_score(question_list_repeat, content_list_response)
    if 10 in judge_scores:
        minimum_index = judge_scores.index(10)
        budget = content_list_index[minimum_index]
    else:
        budget = 1000000000
    big_check_list.append(budget+1)
    print('Current Q ID: {}, Question{}, Budget: {}'.format(ii, question, budget+1))

with open('./output_gpt.txt', 'w') as f:
    for line in big_check_list:
        f.write(f"{line}\n")

