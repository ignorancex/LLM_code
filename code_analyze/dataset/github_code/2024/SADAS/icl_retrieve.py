import json
import random
from sentence_transformers import SentenceTransformer, util
import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print("device is {}".format(device))

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
print(embedder)

import pandas as pd

ICL_EXAMPLE_NUMBER = 8

def write_json(data, savedir):
    with open(savedir, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def get_all_examples():
    file_path = 'data/train_bargaining.csv'

    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)
    list_ = df['text'].tolist()
    conversation_set = set()

    dict_list = []
    for x in list_:
        splits = x.split('</s>')
        dialogue, remediation = splits[0], splits[1]
        remediation = remediation.replace('<s>Assistant: ', '').strip()
        splits = dialogue.replace('<s>Human: 你是一个对话助手，你的任务是改写对话中最后一句话，使其符合社会规范。\n在谈判游戏中，谈判的对话如下：\n', '').split('在这个对话中，最后一句话')
        conversation, last_sentence = splits[0].replace('人物A', 'buyer').replace('人物B', 'seller').strip(), splits[1].replace('违反了社会规范，请对这句话进行改写:', '').replace('\"', '').strip()
        temp_ = conversation.split('\n')
        temp_[-1] = temp_[-1].replace(': ', ':【violation】')
        conversation = '\n'.join(temp_).strip()
        conversation = '# Dialogue:\n' + conversation
        remediation = '# Remediation:\n' + remediation
        if conversation not in conversation_set:
            conversation_set.add(conversation)
            dict_ = {'conversation': conversation, 'last_sentence': last_sentence, 'remediation': remediation}
            dict_list.append(dict_)

    return dict_list

def select_icl_examples(dict_list):
    # Number of random elements to select
    N = ICL_EXAMPLE_NUMBER

    # Select N random items from the dictionary
    random_items = random.sample(dict_list, N)
    return random_items

def assemble_icl_examples(random_items):
    prompt = ''
    for x in random_items:
        prompt += x['conversation'] + '\n'
        prompt += x['remediation'] + '\n\n'
    return prompt.strip()

def obtain_icl_dialogues(dict_list):
    incontext_dialogues = []
    for x in dict_list:
        str_ = '\n'.join(x['conversation'].split('\n')[4:]).strip()
        incontext_dialogues.append(str_)
    return incontext_dialogues

def return_top_k(query, k, corpus_embeddings):
    query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=k)

    indices = []
    scores = []
    for score, idx in zip(top_results[0], top_results[1]):
        indices.append(int(idx))
        scores.append(float(score))
    return indices, scores

def main(icl_method='random', suffix_prompt=''):
    dict_list = get_all_examples()
    # write_json(dict_list, 'data/all_icl_example.json')

    if icl_method == 'retrieve':
        incontext_dialogues = obtain_icl_dialogues(dict_list)
        dialogues_embeddings = embedder.encode(incontext_dialogues, convert_to_tensor=True, device=device)
        indices, cos_scores = return_top_k(suffix_prompt, ICL_EXAMPLE_NUMBER, dialogues_embeddings)
        icl_examples = [dict_list[index] for index in indices]
    elif icl_method == 'random':
        icl_examples = select_icl_examples(dict_list)
    prompt = assemble_icl_examples(icl_examples)
    print(prompt)
    print(len(prompt))

def convert_messages():
    list_ = [{'role': 'user', 'content': '你好，请问贵公司有一种特殊的工业商品吗？'}, {'role': 'assistant', 'content': '你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。'}, {'role': 'user', 'content': '你会考虑以每单元$30的价格进行出售吗？'}, {'role': 'assistant', 'content': '很抱歉，$30的价格对于我们来说太低了。我们的工业商品是高质量且独特的，我们无法以$30的价格出售。我们希望能以每单元不低于$45的价格与您达成交易。'}, {'role': 'user', 'content': '明白了，$45的价格对于您来说是最低限度了。那么，请问您能否提供更多关于这种工业商品的详细信息？例如生产能力、质量标准等。'}, {'role': 'assistant', 'content': '当然，我很愿意提供更多关于我们工业商品的详细信息。我们的工业商品具有先进的生产能力，每月能生产1000个单位。质量标准非常严格，经过严密的测试和检验，确保符合高品质标准。我们还拥有专业的品控团队和先进的设备，以确保产品质量的稳定性和可靠性。这些特点使我们的工业商品在市场上具有很大的竞争力。请问您还有其他关于工业商品的问题吗？'}]
    prompt = 'seller:【violation】' + list_[-1]['content'] + '\n'
    count = 0
    for i in range(len(list_) - 2, -1, -1):
        if count % 2 == 0:
            str_ = 'buyer: ' + list_[i]['content'] + '\n'
        else:
            str_ = 'seller: ' + list_[i]['content'] + '\n'
        count += 1
        prompt = str_ + prompt
    prompt = '# Dialogue:\n' + prompt + '# Remediation:\n'
    return prompt

def produce_prompt_for_retrieve():
    list_ = [{'role': 'user', 'content': '你好，请问贵公司有一种特殊的工业商品吗？'}, {'role': 'assistant',
                                                                                     'content': '你好，我们公司产能充足，能够以一次性的方式向贵司出售所需的工业商品。这种工业商品的单元价格是$50。'},
             {'role': 'user', 'content': '你会考虑以每单元$30的价格进行出售吗？'}, {'role': 'assistant',
                                                                                   'content': '很抱歉，$30的价格对于我们来说太低了。我们的工业商品是高质量且独特的，我们无法以$30的价格出售。我们希望能以每单元不低于$45的价格与您达成交易。'},
             {'role': 'user',
              'content': '明白了，$45的价格对于您来说是最低限度了。那么，请问您能否提供更多关于这种工业商品的详细信息？例如生产能力、质量标准等。'},
             {'role': 'assistant',
              'content': '当然，我很愿意提供更多关于我们工业商品的详细信息。我们的工业商品具有先进的生产能力，每月能生产1000个单位。质量标准非常严格，经过严密的测试和检验，确保符合高品质标准。我们还拥有专业的品控团队和先进的设备，以确保产品质量的稳定性和可靠性。这些特点使我们的工业商品在市场上具有很大的竞争力。请问您还有其他关于工业商品的问题吗？'}]

    list_ = list_[3:]
    prompt = ''
    count = 0
    for i in range(len(list_)-1):
        if count % 2 == 0:
            str_ = 'seller: ' + list_[i]['content'] + '\n'
        else:
            str_ = 'buyer: ' + list_[i]['content'] + '\n'
        count += 1
        prompt = prompt + str_
    prompt += 'seller:【violation】' + list_[-1]['content'] + '\n'
    return prompt


if __name__ == "__main__":
    icl_method = 'retrieve'
    suffix_prompt = 'seller:【violation】$40？这个价格有点太低了吧。我们的产品质量有保证，我们也对自己的产品有信心，所以我们希望以不低于每单元$45的价格进行出售。'
    main(icl_method, suffix_prompt)
    # print(convert_messages())

    # produce_prompt_for_retrieve()
