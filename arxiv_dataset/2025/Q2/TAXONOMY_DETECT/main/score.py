import json
import re
import pandas as pd

from prompt.generate_prompt import generate_prompt



def read_jsonl(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    line = f.readline()
    ret_list = []
    while line:
        ret_list.append(json.loads(line.strip()))
        line = f.readline()
    return ret_list


def split_list(lst, chunk_size=span, type=span_type):
    if type == "static":  # dynamic or static
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    elif type == "dynamic":
        ret = [[]]
        tmp = []
        b_phrase = ''
        for rel in lst:
            if b_phrase != rel[0]:
                b_phrase = rel[0]
                if len(ret[-1]) + len(tmp) < chunk_size:
                    ret[-1].extend(tmp)
                else:
                    ret.extend([tmp])
                tmp = []
            tmp.append(rel)
        final_result = []
        for chunk in ret:
            while len(chunk) > chunk_size:
                final_result.append(chunk[: chunk_size])
                chunk = chunk[chunk_size:]
            if chunk:
                final_result.append(chunk)

        return final_result


def score0_prompt(phrases_list, out_file):
    """a group of concepts"""
    prompt_list = []
    for phrases in phrases_list:
        prompt = generate_prompt([str(phrases)], '../prompt/prompt_4_score_0.txt')
        prompt_list.append(prompt)

    df = pd.DataFrame({'id': range(len(prompt_list)), 'question': prompt_list})
    df.to_excel(out_file, index=False)


def score1_prompt(relation_chunks, out_file):
    """a group of [hypernym, hyponym]"""
    prompt_list = []
    for relations in relation_chunks:
        phrases = []
        for relation in relations:
            phrases.append({"hypernym": relation[0], "hyponym": relation[1]})
        prompt = generate_prompt([str(phrases)], '../prompt/prompt_4_score_1.txt')
        prompt_list.append(prompt)

    df = pd.DataFrame({'id': range(len(prompt_list)), 'question': prompt_list})
    df.to_excel(out_file, index=False)


def score2_prompt(relation_list, out_file):
    """a group of concepts"""
    prompt_list = []
    for relation in relation_list:
        prompt = generate_prompt([str(relation)], '../prompt/prompt_4_score_2.txt')
        prompt_list.append(prompt)

    df = pd.DataFrame({'id': range(len(prompt_list)), 'question': prompt_list})
    df.to_excel(out_file, index=False)


if __name__ == '__main__':


    # Set up argument parser
    parser = argparse.ArgumentParser(description='LITE: LLM-Impelled Taxonomy Evaluation')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='File containing taxonomy data (e.g., data/taxonomy.jsonl)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save evaluation results (e.g., results)')
    parser.add_argument('--span_type', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='Subtree generation method (dynamic|static)')
    parser.add_argument('--span_size', type=int, default=5,
                       help='Number of relationships per subtree when span_type=static')
    
    args = parser.parse_args()


    relation_file = args.input_file
    save_dir = args.output_dir
    span_type = args.span_type
    span = args.span_size

    relation_list = read_jsonl(relation_file)
    relation_chunks = split_list(relation_list, chunk_size=span)
    phrases_list = []
    for relations in relation_chunks:
        phrases = []
        for relation in relations:
            phrases.extend(relation)
        phrases_list.append(list(set(phrases)))

    relation_list2 = []
    for relations in relation_chunks:
        relation_head2tail = {}
        relation_head2tail_list = []
        for relation in relations:
            if relation_head2tail.get(relation[0]) is None:
                relation_head2tail[relation[0]] = []
            relation_head2tail[relation[0]].append(relation[1])
        for node1, node2 in relation_head2tail.items():
            relation_head2tail_list.append(
                {"hypernym": node1,
                 "list of hyponym": node2}
            )
        relation_list2.append(relation_head2tail_list)

    score0_prompt(phrases_list, '../result/task_socre0.xlsx')
    score1_prompt(relation_chunks, '../result/task_socre1.xlsx')
    score2_prompt(relation_list2, '../result/task_socre2.xlsx')