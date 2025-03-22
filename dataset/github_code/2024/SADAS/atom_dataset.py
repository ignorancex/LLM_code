import json
import pandas as pd


def get_conversation(dict_):
    history = dict_['seller_dialog_history'][5:]
    conversation = ''
    for i, sentence in enumerate(history):
        if i % 2 == 0:
            conversation += '人物A: ' + sentence['content'].replace('【violation】', '') + '\n'
        else:
            conversation += '人物B: ' + sentence['content'].replace('【violation】', '') + '\n'
    conversation = conversation.strip()
    last_sentence = history[-1]['content'].replace('【violation】', '').replace('violation', '').replace('[violation]', '')
    remediation = dict_['remediation']
    return conversation, last_sentence, remediation

def create_dataset(paths, output_path):
    instances = []

    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)

        for x in data:
            conversation, last_sentence, remediation = get_conversation(x)
            instance = "<s>Human: 你是一个对话助手，你的任务是改写对话中最后一句话，使其符合社会规范。\n在谈判游戏中，谈判的对话如下：\n$CONVERSATION\n在这个对话中，最后一句话\"$SENTENCE\"违反了社会规范，请对这句话进行改写:\n</s><s>Assistant: $REMEDIATION\n</s>".replace(
                '$CONVERSATION', conversation).replace("$SENTENCE", last_sentence).replace("$REMEDIATION", remediation)
            instances.append(instance)
    df = pd.DataFrame({'text': instances})
    df.to_csv(output_path, index=False)

# file_paths = ['outputs/run_ternary_industrial_seller_social_norm_explicit_labeling_cn_withgoals_buyer_cn_withgoals_20231212-195710_prefer_lists.json', 'outputs/run_ternary_industrial_seller_social_norm_explicit_labeling_cn_withgoals_buyer_cn_withgoals_20231213-055107_prefer_lists.json', 'outputs/run_ternary_industrial_seller_social_norm_explicit_labeling_cn_withgoals_buyer_cn_withgoals_20231213-104901_prefer_lists.json', 'outputs/run_ternary_industrial_seller_social_norm_explicit_labeling_cn_withgoals_buyer_cn_withgoals_20240729-223303_prefer_lists.json']
file_paths = ['outputs/run_ternary_industrial_seller_social_norm_explicit_labeling_cn_withgoals_buyer_cn_withgoals_20240729-223303_prefer_lists.json']
output_path = "data/train_bargaining1.csv"
create_dataset(file_paths, output_path)

