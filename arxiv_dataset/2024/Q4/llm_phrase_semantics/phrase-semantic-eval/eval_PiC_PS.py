import csv
import json
import os, sys
import string

from utils.openaimodel import OpenAIModel

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import sklearn
import argparse
from tqdm import tqdm
from torch import mode, nn
from scipy.stats.stats import pearsonr
import torch
from utils.utils import load_model
from utils.glove_utils import get_phrase_emb
from config.model_path import MODEL_PATH

# The following function is modified from Yu and Ettinger, ACL 2020 
# (https://github.com/yulang/phrasal-composition-in-transformers/tree/master/src)


def load_PiC_PS_data(path):
    rows = []
    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in csvreader:
            # print(', '.join(row))
            rows.append(row)
    return rows


fewshot_examples=[
        {
            "phrase1": "attached agency",
            "phrase2": "accompanying business that represents actors",
            "sentence1": "Carty attended a Saturday morning acting class at Phildene Stage School from the age of four, which had an attached agency.",
            "sentence2": "Carty attended a Saturday morning acting class at Phildene Stage School from the age of four, which had an accompanying business that represents actors.",
            "label": "positive",
            "explanation": "Both phrases describe a business connected to the Stage School, with the primary function of representing actors.",
            "idx": 4
        },
        {
            "phrase1": "first colony",
            "phrase2": "original settlement",
            "sentence1": "after theo's apparent death, she decides to leave first colony and ends up traveling with the apostles.",
            "sentence2": "after theo's apparent death, she decides to leave original settlement and ends up traveling with the apostles.",
            "label": "negative",
            "explanation": "Both phrases refer to an initial establishment or habitation, but \"first colony\" carries specific connotations of being part of a series of colonies or a larger colonization effort, whereas \"original settlement\" is a more neutral term for any first habitation.",
            "idx": 0
        },
        {
            "phrase1": "next course",
            "phrase2": "following level",
            "sentence1": "Each contender that finished last couldn't advance to the next course.",
            "sentence2": "Each contender that finished last couldn't advance to the following level.",
            "label": "positive",
            "explanation": "In the context of contenders not being able to advance after finishing last, both \"next course\" and \"following level\" essentially mean the next stage or step in the competition or challenge.",
            "idx": 5
        },
        {
            "phrase1": "bank's network",
            "phrase2": "bank's locations",
            "sentence1": "The guard assigned to Vivian leaves her to prevent the robbery, allowing her to connect to the bank's network.",
            "sentence2": "The guard assigned to Vivian leaves her to prevent the robbery, allowing her to connect to the bank's locations.",
            "label": "negative",
            "explanation": "\"Bank's network\" and \"bank's locations\" refer to fundamentally different aspects of the bank - one is abstract/digital, and the other is concrete/physical. The actions Vivian could take and the kind of connection she could establish would differ significantly depending on whether it's with the bank's network or its physical locations.",
            "idx": 1
        },
        {
            "phrase1": "innate ability",
            "phrase2": "natural capacity",
            "sentence1": "This is to say that neither direct transdifferentiation nor mitotic division have the innate ability to restore hair cells.",
            "sentence2": "This is to say that neither direct transdifferentiation nor mitotic division have the natural capacity to restore hair cells.",
            "label": "positive",
            "explanation": "Given the context of the sentences, which discusses the capabilities of biological processes (direct transdifferentiation and mitotic division) in hair cell restoration, the difference in nuance between \"innate ability\" and \"natural capacity\" is minimal. In this specific context, they serve the same function in the sentence, both denoting an inherent capability of a biological process.",
            "idx": 7
        },
        {
            "phrase1": "public exchange",
            "phrase2": "free forum",
            "sentence1": "Two days later Louis XVI banished Necker by a \"lettre de cachet\" for his very public exchange of pamphlets.",
            "sentence2": "Two days later Louis XVI banished Necker by a \"lettre de cachet\" for his very free forum of pamphlets.",
            "label": "negative",
            "explanation": "",
            "idx": 2
        },
        {
            "phrase1": "one filter",
            "phrase2": "single straining device",
            "sentence1": "It is advisable for anyone who likes to make coffee often to have more than one filter.",
            "sentence2": "It is advisable for anyone who likes to make coffee often to have more than single straining device.",
            "label": "positive",
            "explanation": "",
            "idx": 10
        },
        {
            "phrase1": "slower form",
            "phrase2": "different type",
            "sentence1": "This suggests that synaptotagmin-7 is responsible for mediating a slower form of Ca(2+)-triggered release while the faster release is induced by synaptotagmin-1.",
            "sentence2": "This suggests that synaptotagmin-7 is responsible for mediating a different type of Ca(2+)-triggered release while the faster release is induced by synaptotagmin-1.",
            "label": "negative",
            "explanation": "",
            "idx": 3
        },
        {
            "phrase1": "maximum growth",
            "phrase2": "optimal development",
            "sentence1": "MAPPS was founded in order to organize the Advanced Placement college preparatory courses in a way to facilitate maximum growth for student achievement at Mosley.",
            "sentence2": "MAPPS was founded in order to organize the Advanced Placement college preparatory courses in a way to facilitate optimal development for student achievement at Mosley.",
            "label": "positive",
            "explanation": "",
            "idx": 11
        },
        {
            "phrase1": "perfect body",
            "phrase2": "ultimate soul",
            "sentence1": "She had such a perfect body that I took a very soft doe skin, we stretched it on her and tied it together with thongs.\"",
            "sentence2": "She had such a ultimate soul that I took a very soft doe skin, we stretched it on her and tied it together with thongs.\"",
            "label": "negative",
            "explanation": "",
            "idx": 6
        },
        {
            "phrase1": "stable balance",
            "phrase2": "proper equilibrium",
            "sentence1": "In the 1940s, Hungary introduced a new technique called the eggbeater kick that enables goalkeepers to maintain a stable balance in the water.",
            "sentence2": "In the 1940s, Hungary introduced a new technique called the eggbeater kick that enables goalkeepers to maintain a proper equilibrium in the water.",
            "label": "positive",
            "explanation": "",
            "idx": 16
        },
        {
            "phrase1": "second duty",
            "phrase2": "extra assignment",
            "sentence1": "The second duty, in his words, is to \"put on the new man\" according to the image of God (Ephesians 4:24).",
            "sentence2": "The extra assignment, in his words, is to \"put on the new man\" according to the image of God (Ephesians 4:24).",
            "label": "negative",
            "explanation": "",
            "idx": 8
        },
        {
            "phrase1": "enough presence",
            "phrase2": "adequate existence",
            "sentence1": "In soil and water, photodegradation is again predominant mechanism if there is enough presence of sunlight.",
            "sentence2": "In soil and water, photodegradation is again predominant mechanism if there is adequate existence of sunlight.",
            "label": "positive",
            "explanation": "",
            "idx": 18
        },
        {
            "phrase1": "third sheet",
            "phrase2": "third flat layer",
            "sentence1": "Only two folios of the notebook were dated, the third sheet 5 December 1594 and the 32nd 27 January 1595 (1596).",
            "sentence2": "Only two folios of the notebook were dated, the third flat layer 5 December 1594 and the 32nd 27 January 1595 (1596).",
            "label": "negative",
            "explanation": "",
            "idx": 9
        },
        {
            "phrase1": "party support",
            "phrase2": "group assistance",
            "sentence1": "Incense Mages can also focus on party support with area control spells or aura buffs, which grow more powerful with more party members.",
            "sentence2": "Incense Mages can also focus on group assistance with area control spells or aura buffs, which grow more powerful with more party members.",
            "label": "positive",
            "explanation": "",
            "idx": 21
        },
        {
            "phrase1": "larger pool",
            "phrase2": "Broader circle",
            "sentence1": "It was this mandate which kept healthcare costs down under the PPACA by promoting cost sharing over a larger pool.",
            "sentence2": "It was this mandate which kept healthcare costs down under the PPACA by promoting cost sharing over a Broader circle.",
            "label": "negative",
            "explanation": "",
            "idx": 22
        },
]

def openai_conduct_PiCPS_test(model, test_data, k_shot=0, cot=False):
    """
    Classify whether a pair of phrases has the same semantic meaning in the same context.
    A data example:
        {"phrase1": "air position",
        "phrase2": "posture while jumping",
        "sentence1": "In 1990, Petit accepted a full-time overnight on air position at gospel radio station WYLD-AM.",
        "sentence2": "In 1990, Petit accepted a full-time overnight on posture while jumping at gospel radio station WYLD-AM.",
        "label": "negative",}
    """
    prompt_template = "Given a pair of phrases sharing the same context, you are tasked to answer whether the two phrases have the same semantic meaning.\n" \
                      "{FEWSHOT_EXAMPLES}" \
                      "\nNow here is the test, in the two sentences below: \n\t- {SENTENCE1}\n\t- {SENTENCE2}\n"
    if cot:
         prompt_template += "Please answer whether the phrase pair \"{PHRASE1}\" and \"{PHRASE2}\" has the same semantic meaning. Let's think step by step, briefly explain the similarity/difference between the two phrases, and end the respond with only Yes or No).\n"
    else:
         prompt_template += "Please answer whether the phrase pair \"{PHRASE1}\" and \"{PHRASE2}\" has the same semantic meaning? Please only respond with Yes or No.\n"

    fewshot_prompt = ""
    if k_shot > 0:
        fewshot_prompt = f"Here are {k_shot} examples:\n"
        if cot:
            fewshot_template = ("Example#{EXAMPLE_ID}, in the two sentences below: \n\t- {SENTENCE1}\n\t- {SENTENCE2}\n"
                                "Whether the phrase pair \"{PHRASE1}\" and \"{PHRASE2}\" has the same semantic meaning? "
                                "Let's think step by step: {EXPLANATION}\n"
                                "So the answer is: {ANSWER}\n")
        else:
            fewshot_template = ("Example#{EXAMPLE_ID}, in the two sentences below: \n\t- {SENTENCE1}\n\t- {SENTENCE2}\n"
                                "{EXPLANATION}"
                                "The phrase pair \"{PHRASE1}\" and \"{PHRASE2}\" has the same semantic meaning is: {ANSWER}\n")
        for i in range(k_shot):
            ex_dict = fewshot_examples[i]
            fewshot_prompt += fewshot_template.format(EXAMPLE_ID=i + 1, SENTENCE1=ex_dict["sentence1"],SENTENCE2=ex_dict["sentence2"],
                                                   PHRASE1=ex_dict["phrase1"], PHRASE2=ex_dict["phrase2"],
                                                   EXPLANATION=ex_dict["explanation"],
                                                   ANSWER="Yes" if ex_dict["label"] == 'positive' else "No")
    num_correct = 0
    pred_details = []
    for idx, ex_dict in tqdm(enumerate(test_data), desc=f"Evaluating {model.model_name}"):
        prompt = prompt_template.format(SENTENCE1=ex_dict["sentence1"],SENTENCE2=ex_dict["sentence2"],
                                        PHRASE1=ex_dict["phrase1"], PHRASE2=ex_dict["phrase2"], FEWSHOT_EXAMPLES=fewshot_prompt)
        raw_pred = model.completions(prompt=prompt, n=1, temperature=0.0, top_p=1, max_tokens=512 if cot else 4)
        if cot:
            lines = [l for l in raw_pred[0].split("\n") if l.strip()]
            pred = lines[-1].lower().strip(string.punctuation)
            if pred.endswith("yes"):
                pred = "yes"
            elif pred.endswith("no"):
                pred = "no"
            else:
                pred = "INVALID"
        else:
            pred = raw_pred[0].lower().strip(string.punctuation)
        target = "yes" if ex_dict["label"] == 'positive' else "no"
        correct = (pred.endswith(target))
        if correct:
            num_correct += 1
        detail = {
            'target': target,
            'pred': pred,
            'correct': correct,
            'label': ex_dict["label"],
            'raw_pred': raw_pred,
            'data': ex_dict,
            'prompt': prompt,
        }
        pred_details.append(detail)
        # print(detail)
        print(f'num={(idx + 1)}, acc={num_correct / (idx + 1)}')
        # if idx == 5: break

    accuracy = num_correct / len(test_data)
    print(f'Accuracy on PiC-PS = {accuracy}')
    return accuracy, pred_details


def main_ps(args):
    # read data
    test_data = json.load(open(args.input_path, "r"))['data']

    # iterate through each model to be tested
    model_scores = {}
    for model_name, model_path in MODEL_PATH.items():
        print(model_name)
        if 'openai' in model_name:
            openai_model = OpenAIModel(model_name=model_path, api_key=args.openai_key)
            acc, pred_details = openai_conduct_PiCPS_test(openai_model, test_data, k_shot=args.k_shot, cot=args.cot)
        else:
            print(f'Only OpenAI model is supported, but {model_name} if given')
        score_dict = {
            'core_metric': acc,
            'accuracy': acc,
        }
        model_scores[model_name] = score_dict
        os.makedirs(args.output_dir, exist_ok=True)
        file_pattern = f"{model_name}-{args.k_shot}shot{'-cot' if args.cot else ''}"
        print(f'Output {model_name} score to {file_pattern}.score.json')
        with open(os.path.join(args.output_dir, f'{file_pattern}.score.json'), 'w') as score_output:
            score_output.write(json.dumps(score_dict, indent=4))
        print(f'Output {model_name} pred details to {file_pattern}.pred.json')
        with open(os.path.join(args.output_dir, f'{file_pattern}.pred.json'), 'w') as pred_output:
            pred_output.write(json.dumps(pred_details, indent=4))

    print('Done with PiC-PS')
    return model_scores


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/PiC-PS/test-hard-v2.0.1.json')
    parser.add_argument('--output_dir', type=str, default='output/PiC-PS/')
    parser.add_argument('--k_shot', type=int, default=0)
    parser.add_argument("--cot", default=False, type=str2bool, help="Use COT or not.")
    parser.add_argument('--openai_key', type=str, default='')
    args = parser.parse_args()
    main_ps(args)

    