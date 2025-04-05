import json
import random
import re
import time

import numpy as np
import openai
import torch
from retry.api import retry_call

from sentence_transformers import SentenceTransformer, util
from torch import nn
from transformers import AutoTokenizer, HoulsbyConfig

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print("device is {}".format(device))


embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)


def get_value(key, my_dict):
    if key in my_dict:
        return my_dict['key']
    else:
        return 'other'

# Corpus with example sentences
def obtain_stratgies(json_data_path):
    json_data = json.load(open(json_data_path))
    rewriting_strategies = []
    incontext_dialogues = []
    for item in json_data:
        topic = item['topic']

        social_distance = item['social distance']
        social_relation = item['social relation']
        formality = item['formality']
        location = item['location']
        chinese_topic = topic_mapping[topic]
        chinse_social_distance = social_distance_mapping[social_distance]
        chinse_social_relation = social_relation_mapping[social_relation]
        chinse_formality = formality_mapping[formality]
        chinese_location = location_mapping[location]
        dialogue = ["Speaker FLE:" + elem['utterance'].strip() if idx%2 == 0 else "Speaker SME:" + elem['utterance'].strip() for idx, elem in enumerate(item['dialogue'])]
        if isinstance(chinse_social_relation,tuple):
            prompt = "有一个对话场景，场景中有两个人。其中一个人是" + chinse_social_relation[0] + "，另外一个人是" + chinse_social_relation[1] + "，他们是" + chinse_social_distance + "的关系，他们在" \
                     + chinese_location + "进行了一次" + chinse_formality + "的对话，对话的内容与"+chinese_topic+"有关。以下是他们的对话：\n\"" + "\n".join(dialogue) \
                     + "\"\n 对话中，最后一句话\"" + dialogue[-1].split(':')[1] +"\"违反了社会规范。更好的说法应该是：\"" + item['Remediation'] +"\""
        else:
            prompt = "有一个对话场景，场景中有两个人。两个人是" + chinse_social_relation + "，并且他们是" + chinse_social_distance + "的关系，他们在" \
                     + chinese_location + "进行了一次" + chinse_formality + "的对话，对话的内容与" + chinese_topic + "有关。以下是他们的对话：\n\"" + "\n".join(dialogue) \
                     + "\"\n 对话中，最后一句话\"" + dialogue[-1].split(':')[1] + "\"违反了社会规范。更好的说法应该是：\"" + item[
                         'Remediation'] + "\""
        incontext_dialogues.append("\n".join(dialogue))
        rewriting_strategies.append(prompt)

    return rewriting_strategies, incontext_dialogues


def return_top_k(query, k, corpus_embeddings, strategies):
    query_embedding = embedder.encode(query, convert_to_tensor=True, device=device)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    if len(strategies) <= 10:
        inner_scores = util.cos_sim(corpus_embeddings, corpus_embeddings).mean(axis=1)
        print(inner_scores)
    else:
        inner_scores = 0

    top_results = torch.topk((cos_scores+inner_scores)/2, k=k)

    selected_strategies = []
    for score, idx in zip(top_results[0], top_results[1]):
        selected_strategies.append(strategies[idx])
    return selected_strategies, (cos_scores+inner_scores)/2



def select_text_between_quotes(text):
    # Define the regular expression pattern to match text between quotes
    pattern = r'"([^"]*)"'

    # Use re.findall to extract all matches of the pattern in the text
    matches = re.findall(pattern, text)

    # Return the list of matches
    return matches


json_data_path = 'human_written_dialogues.json'




formality_mapping = {"formal": "正式", "informal":"非正式"}
# [ '办公室','家','警察局','难民营','学校', '餐馆','商场','酒店','网络']
location_mapping = {"home":"家", "office":"办公室", "police-station":"警察局",
                    "refugee-camp":"难民营", "school":"学校", "restaurant":"餐馆", "store":"商场", "hotel":"酒店",
                    "online":"网络", 'open-area':"露天场合", "disaster-relief-site":"灾民安置点","military-camp":"军营"}
topic_mapping = {'counter-terrorism':'反对恐怖主义', 'life-trivial':'生活琐事', 'office-affairs':'办公事务',
                 'school-life':'学校生活', 'farming':'农业发展', 'poverty-assistance':'扶贫',
                 'child-missing':'失踪儿童', 'sale':'推销（产品或者服务）',
                 'police-corruption':'警察腐败', 'food':'食品', 'partnership-establishment':'建立合作关系',
                 'public-image-improvement':'提升公共形象', 'support-supply':'提供(食品，药品等)支持',
                 'trading':'交易', 'disaster-relief':'灾后救援', 'soldier-recruitment-interview':'招兵面试',
                 'soldier-training':'士兵训练',"tourism":"旅游","refugee":"难民"}
social_relation_mapping = {'elder-junior':('年长者','年轻人'), 'peer-peer':'平辈',
                           'student-professor':('学生','教授'),
                           'chief-subordinate':'上下级',
                           'commander-soldier':('指挥官','士兵'),
                           'partner-partner':'一对恋人',
                           'customer-server':('顾客','服务生'),
                           'mentor-mentee':('导师','学员'),
                           'patient-doctor':('病人','医生')}
social_distance_mapping = {'family':'家人', 'working':'工作', 'romantic':'恋爱', 'friend':'朋友',
                           'stranger':'陌生人','neighborhood':'邻居'}
norm_mapping = {'greeting': '打招呼', 'apology':'道歉', 'persuasion':'劝说', 'criticism':'批评', 'request':'请求',
                'thanks':'感谢', 'taking-leave':'告别'}

#dialogue = ["Speaker FLE: 你们提供的食品太少了，我们需要更多的帮助。",
#                     "Speaker SME: 我们会尽力提供帮助，但是资源有限。",
#                    "Speaker FLE: 那么你们为什么不能调动更多的资源？"]

norm_category = "apology"

topic = "life-trivial"

social_distance = "friend"

social_relation = "peer-peer"

formality = "informal"

location = "open-area"

rewriting_strategies, incontext_dialogues = obtain_stratgies(json_data_path)

rewriting_strategies_embeddings = embedder.encode(incontext_dialogues, convert_to_tensor=True, device=device)

chinese_topic = topic_mapping[topic]
chinse_social_distance = social_distance_mapping[social_distance]
chinse_social_relation = social_relation_mapping[social_relation]
chinse_formality = formality_mapping[formality]
chinese_location = location_mapping[location]

in_context_topk = 8

#if isinstance(chinse_social_relation, tuple):
    #suffix_prompt = "当前有一个对话场景，场景中有两个人。其中一个人是" + chinse_social_relation[0] + "，另外一个人是" + \
    #         chinse_social_relation[1] + "，他们是" + chinse_social_distance + "的关系，他们在" \
    #         + chinese_location + "进行了一次" + chinse_formality + "的对话，对话的内容与" + chinese_topic + "有关。以下是他们的对话：\n\"" + "\n".join(dialogue) \
    #         + "\" 对话中，最后一句话\"" + dialogue[-1].split(':')[1] + "\"违反了社会规范。"
#else:
    #suffix_prompt = "当前有一个对话场景，场景中有两个人。两个人是" + chinse_social_relation + "，并且他们是" + chinse_social_distance + "的关系，他们在" \
    #         + chinese_location + "进行了一次" + chinse_formality + "的对话，对话的内容与" + chinese_topic + "有关。以下是他们的对话：\n\"" + "\n".join(dialogue) \
    #         + "\" 对话中，最后一句话\"" + dialogue[-1].split(':')[1] + "\"违反了社会规范。"

# def chatgpt_remediation_generation(dialogue, norm_category):
#     print('dialogue: {}'.format(dialogue))
#
#     suffix_prompt = "当前有一个对话场景，场景中有两个人FLE和SME。以下是他们的对话：\n\"" + "\n".join(dialogue) + "\" 对话中，最后一句话\"" + dialogue[-1].split(':')[1] + "\"违反了社会规范。"
#     in_context_examples = return_top_k(suffix_prompt, in_context_topk, rewriting_strategies_embeddings,
#                                           rewriting_strategies)
#
#
#     prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n"+ "\n".join(in_context_examples) +'\n\n'+ \
#              suffix_prompt + "请根据我们的提示，把这段对话中的\"" + dialogue[-1].split(':')[1] + "\"修改成更礼貌的符合社会规范的说法，不要改变原句的意思，修改后的对话也要保持流畅，并且解释为何这样说符合社会规范。"
#
#     # print(prompt)
#
#
#     keys = ['sk-AAAAA']
#
#     completion = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo",
#       messages=[{"role": "user", "content": prompt}],
#      api_key=random.choice(keys)
#     )
#
#     chatgpt_message = completion['choices'][0]['message']['content']
#     remediation = select_text_between_quotes(chatgpt_message)[-1]
#     justification = chatgpt_message.split("\""+remediation+"\"")[1]
#     print('remediation: {}'.format(remediation))
#     print('justification: {}'.format(justification))
#     return remediation, justification

def chatgpt_remediation_generation_with_norm(dialogue, norm_category, prompt=None):
    # print(dialogue)
    if prompt is not None:
        #if prompt == 'Speaker SME本来内心想直接开始工作，由于Speaker FLE多次邀请Speaker SME晚餐，Speaker SME决定参加晚餐，而且不表露内心最初的想法。':
        norm_category = prompt

    if norm_category not in norm_mapping:
        chinese_norm_category = "对话"
    else:
        chinese_norm_category = norm_mapping[norm_category]
    suffix_prompt = "当前有一个对话场景，场景中有两个人FLE和SME，他们在进行对话。以下是他们的对话：\n\"" + "\n".join(
        dialogue) + "\"\n对话中，" + dialogue[-1].split(':')[
                        0] + "的最后一句话违反了" + chinese_norm_category + "的方式。"

    incontext_start_time = time.time()

    in_context_examples, sim_scores = return_top_k(suffix_prompt, in_context_topk, rewriting_strategies_embeddings,
                                                   rewriting_strategies)


    print("in context query time is : {}".format(str(time.time()-incontext_start_time)))
    if prompt == 'Speaker SME本来内心想直接开始工作，由于Speaker FLE多次邀请Speaker SME晚餐，Speaker SME决定参加晚餐，而且不表露内心最初的想法。':
        if len(dialogue) > 1:
            prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n" + "\n".join(
                in_context_examples) + '\n\n' + \
                     suffix_prompt + "在中文语境下，" + dialogue[-1].split(':')[
                         0] + chinese_norm_category + "的方式需要符合社会规范。为了避免冒犯他人，请根据我们例子中的提示，把这段对话中，" + \
                     dialogue[-1].split(':')[0] + "的话\"" + dialogue[-1].split(':')[
                         1] + "\"修改成更礼貌的符合社会规范的一句话。"+prompt+"请严格保留原句的意思，并且解释修改后的句子为何符合中国社会规范并且如何保留了原句的意思。"
        else:
            prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n" + "\n".join(
                in_context_examples) + '\n\n' + \
                     suffix_prompt + "在中文语境下，" + dialogue[-1].split(':')[
                         0] + chinese_norm_category + "的方式需要符合社会规范。为了避免冒犯他人，请根据我们例子中的提示，把这段对话中，" + \
                     dialogue[-1].split(':')[0] + "的话\"" + dialogue[-1].split(':')[
                         1] + "\"修改成更礼貌的符合社会规范的一句话。"+prompt+"请严格保留原句的意思，并且解释修改后的句子为何符合中国社会规范并且如何保留了原句的意思。"

    else:
        if len(dialogue) > 1:
            prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n" + "\n".join(
                in_context_examples) + '\n\n' + \
                     suffix_prompt + "在中文语境下，" + dialogue[-1].split(':')[
                         0] + chinese_norm_category + "的方式需要符合社会规范。为了避免冒犯他人，请根据我们例子中的提示，把这段对话中，" + \
                     dialogue[-1].split(':')[0] + "的话\"" + dialogue[-1].split(':')[
                         1] + "\"修改成更礼貌的符合社会规范的一句话。请严格保留原句的意思，并且解释修改后的句子为何符合中国社会规范并且如何保留了原句的意思。"
        else:
            prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n" + "\n".join(
                in_context_examples) + '\n\n' + \
                     suffix_prompt + "在中文语境下，" + dialogue[-1].split(':')[
                         0] + chinese_norm_category + "的方式需要符合社会规范。为了避免冒犯他人，请根据我们例子中的提示，把这段对话中，" + \
                     dialogue[-1].split(':')[0] + "的话\"" + dialogue[-1].split(':')[
                         1] + "\"修改成更礼貌的符合社会规范的一句话。请严格保留原句的意思，并且解释修改后的句子为何符合中国社会规范并且如何保留了原句的意思。"

    # 用来回答" + dialogue[-2].split(':')[0] + "的话\"" + \
    #                  dialogue[-2].split(':')[1] + "\"
    # no_context_prompt = "以下列出的不同对话中，均有人违反了社会规范，我们根据对话场景，提供了更好的说法。\n\n" + "\n".join(
    #        in_context_examples) + '\n\n' + "当前有一个对话场景，场景中有两个人FLE和SME。对话中，" + dialogue[-1].split(':')[0] + "的最后一句话\"" + dialogue[-1].split(':')[
    #                    1] + "\"违反了" + chinese_norm_category + "的方式。在中文语境下，" + dialogue[-1].split(':')[
    #                        0] + chinese_norm_category + "的方式需要符合社会规范。为了避免冒犯他人，请根据我们例子中的提示，把这段对话修改成更礼貌的符合社会规范的一句话。不要改变原句的意思，并且解释为何这样说符合社会规范。"

    no_context_prompt = "为了避免冒犯他人，请把这句话\"" + dialogue[-1].split(':')[1] + "\"修改成符合中国社会规范的语句，" \
                                                                                       "保留原句的意思，并且解释修改后的句子为何符合中国社会规范并且如何保留了原句的意思。输出答案格式请严格如下:\n" \
                                                                                       "\"修改后的句子\"\n解释"

    # print(prompt)
    # print(no_context_prompt)

    chatgpt_query_time = time.time()
    #print(prompt)

    # # todo: add keys
    # # for instance:
    # keys = [
    #         'sk-aaa',
    #         'sk-bbb']
    # todo: add retry
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model":"gpt-3.5-turbo",
            "messages":[{"role": "user", "content": prompt}],
            "api_key":random.choice(keys),
            "n":1,"temperature":1.0, "request_timeout":30}, tries=3, delay=1, jitter=1)
        #completion = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo",
        #    messages=[{"role": "user", "content": prompt}],
        #    api_key=random.choice(keys),
        #    n=4,
        #    temperature=0.25,
        #)
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices':[{'message':{'content':'Error'}}]}
        return [], []
    """
    nocontext_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": no_context_prompt}],
        api_key=random.choice(keys),
        n=3,
        temperature=0.25,
    )
    """
    print("chatgpt query time is : {}".format(str(time.time()-chatgpt_query_time)))

    #print(f'{completion["usage"]["prompt_tokens"]} prompt tokens used.')
    #print(f'{nocontext_completion["usage"]["prompt_tokens"]} prompt tokens used.')
    remediation_list = []
    justification_list = []
    # print(completion['choices'])
    for choice in completion['choices']:
        # print("====================================")
        chatgpt_message = choice['message']['content']
        print('chatgpt_message is: {}'.format(chatgpt_message))
        matches = select_text_between_quotes(chatgpt_message)
        if len(matches) == 1:
            remediation = matches[0]
            justification = chatgpt_message.split(remediation)[1][1:].replace("|||", "")
        else:
            # print("no quotes found！！！！！！！！！！")
            return [], []
            #remediation = "非常抱歉，我刚才没有听清您的问题，请问您能重复一遍刚才的问题么？"
            #justification = "当没有听清对方的问题时，礼貌地表达\"希望对方重复问题\"是礼貌并且符合社会规范的。"
        if not remediation.strip() in remediation_list:
            remediation_list.append(remediation.strip())
            justification_list.append(justification.strip())


    return remediation_list, justification_list
