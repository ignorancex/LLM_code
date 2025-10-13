import numpy as np
import pandas as pd
import random
import torch
from openai import OpenAI
from time import sleep
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import chinese_converter
import warnings
warnings.filterwarnings('ignore')

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LLMEngine(object):
    def __init__(self, llm):
        self.llm = llm
        self.get_hyperpara()

        if llm == 'gpt4o':
            self.engine = 'gpt-4o-2024-05-13'
        elif llm == 'gpt4':
            self.engine = 'gpt-4'
        elif llm == 'gpt3.5':
            self.engine = 'gpt-3.5-turbo'
        elif llm == 'ds':
            self.engine = 'deepseek-r1'
        elif llm in ['qwen', 'baichuan2', 'chatglm2', 'breeze', 'taiwan-llm', 'llama3-70b', 'llama3-8b']:
            if llm == 'qwen':
                self.model_label = "Qwen/Qwen2.5-7B-Instruct"
            elif llm == 'baichuan2':
                self.model_label = "baichuan-inc/Baichuan2-7B-Chat"
            elif llm == 'chatglm2':
                self.model_label = "THUDM/chatglm2-6b"
            elif llm == 'breeze':
                self.model_label = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
            elif llm == 'taiwan-llm':
                self.model_label = "yentinglin/Taiwan-LLM-7B-v2.1-chat"
            elif llm == 'llama3-70b':
                self.model_label = "meta-llama/Meta-Llama-3-70B-Instruct"
            elif llm == 'llama3-8b':
                self.model_label = "meta-llama/Meta-Llama-3-8B-Instruct"
            if self.llm in ['chatglm2']:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_label, trust_remote_code=True)
                self.engine = AutoModel.from_pretrained(self.model_label, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, temperature=self.temperature).eval()
            else:
                self.engine = pipeline("text-generation", model=self.model_label, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", trust_remote_code=True)

    def generate_response(self, prompt):
        if self.llm in ['ds']:
            full_prompt=[{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content":prompt}]
        elif self.llm in ['baichuan2', 'chatglm2']:
            full_prompt = prompt
        else:
            full_prompt=[{"role":"user","content":prompt}]

        if self.llm in ['gpt4o', 'gpt4', 'gpt3.5', 'ds']:
            return prompt_with_openai(full_prompt=full_prompt, chat_model=self.engine)
        elif self.llm in ['qwen', 'baichuan2', 'chatglm2', 'breeze', 'taiwan-llm', 'llama3-70b', 'llama3-8b']:
            if self.llm in ['chatglm2']:
                response_str, _ = self.engine.chat(self.tokenizer, full_prompt, history=[])
                return response_str
            else:
                response_str = self.engine(full_prompt, temperature=self.temperature, pad_token_id=self.engine.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens)
                if self.llm in ['baichuan2']:
                    return response_str[0]['generated_text'].replace(f'{prompt}\n\n','')
                else:
                    return response_str[0]['generated_text'][1]['content']

    def get_hyperpara(self):
        self.max_new_tokens = 512
        self.temperature = 0.1


def prompt_with_openai(full_prompt, chat_model):
    api_key = "<your-api-key>"
    client = OpenAI(api_key=api_key)
    response=None
    timeout_counter=0

    while response is None and timeout_counter<=30:
        try:
            response = client.chat.completions.create(
                            messages=full_prompt,
                            model=chat_model,
                            temperature=0
                            )    
        except Exception as msg:
            if "timeout=600" in str(msg):
                timeout_counter+=1
            print(msg)
            sleep(5)
            continue

    if response==None:
        response_str=""
    else:
        response_str = response.choices[0].message.content
    return response_str


def construct_path(file_type, task, lang, prompt_id=None, llm=None, sub_file_type=None, no_think=False):
    if task not in ['term', 'name']:
        raise NotImplementedError
    if file_type in ['response']:
        if no_think:
            return f'{file_type}/regional_{task}/{sub_file_type}/{llm}_{lang}_{prompt_id}_no_think.csv'  
        else:
            return f'{file_type}/regional_{task}/{sub_file_type}/{llm}_{lang}_{prompt_id}.csv'    
    elif file_type in ['prompt']:
        return f'{file_type}/regional_{task}/{lang}_{prompt_id}.csv'

def process_df(file_type, task, lang, prompt_id=None, llm=None, action=None, df_to_save=None, sub_file_type=None, no_think=False):
    path = construct_path(file_type=file_type, task=task, lang=lang, prompt_id=prompt_id, llm=llm, sub_file_type=sub_file_type, no_think=no_think)
    
    if file_type in ['response']:
        if action == 'save':
            df_to_save.to_csv(path, index=False)
        elif action == 'load':
            return pd.read_csv(path)
        else:
            raise NotImplementedError
    elif file_type in ['prompt']:
        return pd.read_csv(path)
    else:
        raise NotImplementedError
    
def postprocess_response(df, model, lang, ver):
    if model == 'llama3-70b' and lang == 'simplified' and ver == 2:
        for i in range(len(df)):
            if len(df['correct'][i]) > 1:
                df.loc[i, 'correct'] = '1'
    elif model == 'llama3-8b' and lang == 'traditional' and ver == 2:
        for  i in range(len(df)):
            if df['correct'][i] == '1. 0':
                df.loc[i, 'correct'] = '1'
    elif model == 'llama3-8b' and lang == 'simplified' and ver == 3:
        for i in range(len(df)):
            if "python" in df['correct'][i]:
                df.loc[i, 'correct'] = '1'

def get_oppo(gt, response, mode):
    res = []
    for idx, item in enumerate(gt):
        if mode == 'st':
            oppo = chinese_converter.to_traditional(item)
        elif mode == 'ts':
            oppo = chinese_converter.to_simplified(item)
        if len(oppo) <=0:
            raise "no output"
        if '/' in oppo:
            for tem in oppo.split('/'):
                if tem in response[idx]:
                    oppo_flag=1
                    break
            else:
                oppo_flag=0
        else:
            if type(response[idx]) == float:
                oppo_flag=0
            else:
                if oppo in response[idx]:
                    oppo_flag=1
                else:
                    oppo_flag=0
        
        if '/' in item:
            for tem in item.split('/'):
                if tem in response[idx]:
                    item_flag=1
                    break
            else:
                item_flag=0
        else:
            if type(response[idx]) == float:
                item_flag=0
            else:
                if item in response[idx]:
                    item_flag=1
                else:
                    item_flag=0

        ##### The following contains the edge cases which cannot be detected using the code above. We manually label them.
        if mode == 'st':
            if response[idx] in ['「大陆居民往来台湾通行证」的簡稱為「台胞证」。', '「大陆居民往来台湾通行证」的簡稱是「台胞证」。']:
                item_flag = 0
            
            if response[idx] in ['空乘員。', '空乘', '空客',  
                                 '铁路警察','鐵路警察。', '磁悬浮列車',
                                   '磁悬浮列車。','磁懸浮','公交樞紐']:
                item_flag = 1

            if '空客（Airbus）' in response[idx]:
                item_flag = 1

            if '鐵路警察' in response[idx]:
                item_flag = 1

            if '\n\n\n\nA. 警察局\nB. 鐵路警察\nC. 鐵路局\nD. 鐵路警察局' in response[idx]:
                item_flag = 0
            if '\n\n\nA. 鐵路警察\nB. 鐵路保安\nC. 鐵路警衛\nD. 鐵路保衛\n\n' in response[idx]:
                item_flag = 0
            if '（一）警察局（二）鐵路警察局（三）交通警察局（四）鐵路保安局' in response[idx]:
                item_flag = 0
            if '\nA. 鐵路警察\nB. 鐵路保安局\nC. 鐵路交通警察\nD. 鐵路安全警察' in response[idx]:
                item_flag = 0
            if '）\n* (What is the term for the police responsi' in response[idx]:
                item_flag = 0
            if '\n\n\nA. 警察\nB. 鐵路警察\nC. 鐵路保衛警察\nD. 保衛警察\n\n\n' in response[idx]:
                item_flag = 0
            if '\n\n\nA. 鐵路警察\nB. 鐵路保衛警察\nC. 鐵路安全警察\nD. 鐵路保安警察' in response[idx]:
                item_flag = 0
            if '》\n* Answer: 武警\n\n#### 5.8.1.2.3.2.3\n\n* Qu' in response[idx]:
                item_flag = 0
            if response[idx] in ['「大陸居民來往臺灣通行證」的簡寫是「陸胞證」。', '大陸居民往來臺灣通行證，簡稱為「大通證」。',
                                 '大陸居民往來臺灣通行證（簡稱「陸胞證」）。']:
                oppo_flag = 0
            if '\n\nA. 來台證\nB. 陸胞證\nC. 來台通行證\nD. 通行證\n\n正確答案：' in response[idx]:
                oppo_flag = 0
            if '大陆居民往来台湾地区通行证，简称“台胞证”，是中华人民共和国公民往来台湾地区时所持有的证件。' in response[idx]:
                item_flag = 0
            if '大陆居民往来台湾通行证，简称“大通证”。这是中国政府为' in response[idx]:
                item_flag = 0
            if '大陆居民往来台湾通行证，简称“大通证”，' in response[idx]:
                item_flag = 0
            if '中国公民往来台湾地区的有效证件是大陆居民往来台湾通行证，俗称“台胞证”' in response[idx]:
                item_flag = 0
            if '")\n    print("Answer: 入台證")\n    print("")\n' in response[idx]:
                oppo_flag = 0
            if item == '公共交通':
                if 'print("Answer:巴士")' in response[idx]:
                    item_flag = 0
                    if '公共交通' in response[idx]:
                        oppo_flag = 0
                    
                
                if 'print("Answer: 巴士")' in response[idx]:
                    item_flag = 0
                    if '公共交通' in response[idx]:
                        oppo_flag = 0

                if 'print("Answer:巴士")' in response[idx]:
                    item_flag = 0
                    if '公共交通' in response[idx]:
                        oppo_flag = 0
                
                if '"answer": "巴士"' in response[idx]:
                    item_flag = 0
                    if '公共交通' in response[idx]:
                        oppo_flag = 0
            
        if oppo_flag == 1 or item_flag == 1:
            res.append(1)
        else:
            res.append(0)
    return res


def label_response(df_gpt_correct, oppo):
    #   1: correct
    #   2: misaligned
    #   0: incorrect
    gpt_correct = [int(x) for x in df_gpt_correct['correct'].tolist()]
    res = []
    for idx, item in enumerate(gpt_correct):
        if oppo[idx] == 1:
            group = 2
        else:
            if item == 1:
                group = 1
            else:
                group = 0
        res.append(group)   
    return res


def compute_percentage(df):
    mc = len(df[df['region'] == 1]) / len(df)
    t = len(df[df['region'] == 0]) / len(df)
    na = len(df[df['region'] == -1]) / len(df)
    return mc, t, na


def collect_response_details(df, df_gt):
    counts = df.groupby(['gt', 'label']).size().unstack(fill_value=0)

    for label in [0, 1, 2]:
        if label not in counts.columns:
            counts[label] = 0

    counts['label_counts'] = counts[1].astype(str) + ',' + counts[2].astype(str) + ',' + counts[0].astype(str)
    counts = counts[['label_counts']].reset_index()

    return pd.merge(df_gt.loc[:109][['gt']], counts, on='gt')

class ModelCardDict(object):
    def __init__(self) -> None:
        self.model_card_dict = {
            "qwen": "Qwen/Qwen1.5-7B-Chat",
            "baichuan2": "baichuan-inc/Baichuan2-7B-Chat",
            "chatglm2": "THUDM/chatglm2-6b",
            "breeze": "MediaTek-Research/Breeze-7B-Instruct-v1_0",
            "taiwan-llm": "yentinglin/Taiwan-LLaMa-v1.0",
            "ds": "deepseek-ai/DeepSeek-R1",
            "gpt4o": "gpt-4o-2024-05-13",
            "gpt4": "gpt-4",
            "gpt3.5": "gpt-3.5-turbo",
            "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "qwen3-32b": "Qwen/Qwen3-32B"
        }
        self.model_latex_dict = {
            "qwen": "{\\qwen}",
            "baichuan2": "{\\baichuan}",
            "chatglm2": "{\\chatglm}",
            "breeze": "{\\breeze}",
            "taiwan-llm": "{\\taiwanllm}",
            "ds": "{\\dsf}",
            "gpt4o": "{\\gptivo}",
            "gpt4": "{\\gptiv}",
            "gpt3.5": "{\\gptiii}",
            "llama3-70b": "{\\llamas}",
            "llama3-8b": "{\\llamae}",
            "qwen3-32b": "{\\qwenthree}"
        }

def sig_sign(pval, no_ns=False):
    if pval < 0.05 and pval >=0.01:
        return '*'
    elif pval < 0.01 and pval >=0.001:
        return '**'
    elif pval < 0.001:
        return '***'
    else:
        if no_ns:
            return ''
        else:
            return 'NS'