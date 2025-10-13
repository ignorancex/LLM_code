import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
import pdb
import random
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

os.environ['OPENAI_API_KEY'] = ""
llm = ChatOpenAI(model="gpt-4o-2024-11-20",temperature=0.7) 

def read_memory(memory_list):
    embedding_size = 1536   
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    for mem in memory_list:
        memory.save_context({"input": mem['input']}, {"output":  mem['output']})
    return memory

def read_json_as_dict(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return data
            else:
                return {"data": str(data)}
    except FileNotFoundError:
        print(f"Not find'{file_path}'")
        return {}
    except json.JSONDecodeError:
        print(f"'{file_path}' is not a valid file.")
        return {}


def expand_multilabel_dataset(X, labels):

    expanded_X = []
    expanded_labels = []
    
    for i in range(len(X)):
        sample_labels = np.where(labels[i] == 1)[0]
        
        for label in sample_labels:
            expanded_X.append(X.iloc[i])
            
            new_label = np.zeros_like(labels[i])
            new_label[label] = 1
            expanded_labels.append(new_label)
    
    expanded_X = pd.DataFrame(expanded_X, columns=X.columns)
    expanded_labels = np.array(expanded_labels)
    
    return expanded_X, expanded_labels

def parse_labels(label_str, num_labels):
    # label_list = [int(c) - 1 for c in label_str]
    label_list = [int(c) - 2 for c in label_str]
    multi_hot = [0] * num_labels
    for idx in label_list:
        multi_hot[idx] = 1
    return multi_hot

def append_dict_to_json(file_path, new_dict):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = []

    if not isinstance(data, list):
        data = [data]

    data.append(new_dict)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


num_labels = 4 

train_data_path = 'ml_all.csv'
data = pd.read_csv(train_data_path, delimiter=',', encoding='latin')


data = data.fillna(0)
X = data.iloc[:, :-6]  
all_data = X[:350]
dff = pd.read_csv(train_data_path)


labels = dff['labels'].tolist()
labels = [str(int(lab)) for lab in labels]
labels = [parse_labels(label_str, num_labels) for label_str in labels]
labels = np.array(labels[:350])


train_X, train_labels = expand_multilabel_dataset(all_data, labels)

train_labels_single = np.argmax(train_labels, axis=1)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

rf_classifier.fit(train_X, train_labels_single)

train_data_path = 'Marshall_kincade_test_encoded2_1.csv'
data = pd.read_csv(train_data_path, delimiter=',', encoding='latin')
# demographic
file_path='Marshall_kincade_test2.csv'
df=pd.read_csv(file_path,encoding='latin')

label_data=df['actual_values']

data = data.fillna(0)
X = data.iloc[:, :-1]  
test_X=X
test_predictions = rf_classifier.predict(test_X)

import torch
import torch.nn as nn
from tqdm import tqdm
import json
import pandas as pd
# from classifier import SimpleNet, train_model, load_model, predict
import numpy as np
from openai import OpenAI

import csv

client = OpenAI()

def append_csv(new_column_list,header_name,filename):

    if os.path.exists(filename):
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        if data:
            data[0].append(header_name)
        else:
            data.append([header_name])

        max_rows = max(len(data), len(new_column_list) + 1)

        for i in range(1, max_rows):
            if i < len(data):
                if i - 1 < len(new_column_list):
                    data[i].append(new_column_list[i - 1])
                else:
                    data[i].append('')
            else:
                row = [''] * (len(data[0]) - 1)
                if i - 1 < len(new_column_list):
                    row.append(new_column_list[i - 1])
                else:
                    row.append('')
                data.append(row)

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    else:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([header_name])
            for value in new_column_list:
                writer.writerow([value])



def extract_columns_csv(file_path, column_indices_input):
    df = pd.read_csv(file_path)
    column_indices = [int(idx.strip()) - 1 for idx in column_indices_input.split(',')]
    extracted_columns = []
    for idx in column_indices:
        column_name = df.columns[idx]
        column_data = [column_name] + df.iloc[:, idx].tolist()
        extracted_columns.append(column_data)
    info_eva_list = []
    for ind in range(data_len):
        qs_list = []
        qid=0
        for col in extracted_columns:
            qid+=1
            answer = col[ind + 1]
            question = col[0]
            if isinstance(answer, float):
                response = 'The respondent did not answer to this question.'
            else:
                response = answer
            qs_list.append(f"Question {qid} is: {question}\n The answer to this question is: {response}\n")
        Q1 = ''.join(qs_list)
        info_eva_list.append(Q1)
    return info_eva_list

def extract_quatified_demogra(file_path, column_indices_input):
    df = pd.read_csv(file_path)
    column_indices = [int(idx.strip()) - 1 for idx in column_indices_input.split(',')]
    extracted_columns = []
    for idx in column_indices:
        column_name = df.columns[idx]
        column_data = [column_name] + df.iloc[:, idx].tolist()
        extracted_columns.append(column_data)
    info_eva_list = []
    for ind in range(data_len):
        qs_list = []
        qid=0
        for col in extracted_columns:
            qid+=1
            answer = col[ind + 1]
            question = col[0]
            if isinstance(answer, float):
                response = 'The respondent did not answer to this question.'
            else:
                response = answer
            qs_list.append(response)
        info_eva_list.append(qs_list)
    return info_eva_list

def read_json_as_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)   
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return data
            else:
                return {"data": str(data)}
    except FileNotFoundError:
        print(f"Not find'{file_path}'")
        return {}
    except json.JSONDecodeError:
        print(f"'{file_path}' is not a valid file.")
        return {}


def get_response(model_type, system, question):
    response = client.chat.completions.create(
            model=model_type,
            temperature=0.7,
            messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
            ],
            #   logprobs=True,
            )
    return response.choices[0].message.content

def get_response_with_memory(memory, input, label, epo):
    
    # Add to examples
    history=memory.load_memory_variables({"prompt": input})["history"]
    output_ind=history.find('output')
    history=history[output_ind:]
    history_list=history.split('output:')
    history=''
    history_list=history_list[1:]
    for i in range(len(history_list)):
        history=history+f'\nThis is example {i+1}:'+history_list[i]
    
    prompt_tpl = '''You are an advanced reasoning agent capable of self-improvement through reflection. You have access to a post-wildfire survey completed by local residents who experienced a specific wildfire event. Your task is to generate a logical, step-by-step chain of thought to infer whether the resident evacuated during the wildfire. Ensure each step is clearly connected. You must conclude with a definitive YES or NO answer regarding whether the resident evacuated. You will be provided with previous successful examples that have similar informations. You may reference the rationale from these examples in your analysis.
    **Previous Examples**:\n'''+history+'''\n
    **This is the survey completed by the local resident**:
    {input}

    This is your previous conversation:
    {history}
    '''

    PROMPT = PromptTemplate(
        input_variables=[ 'history',"input"], template=prompt_tpl
    )

    conversation_with_summary = ConversationChain(
        llm=llm, 
        prompt=PROMPT,
        memory=ConversationBufferMemory(),
        verbose=True
    )

    agent_response=conversation_with_summary.predict(input=input)
    if 'YES' in agent_response and label==1:
        correct=True
    elif 'NO' in agent_response and label==0:
        correct=True
    else:
        correct=False
    if not correct and epo<data_len*0.7:
        if label==1:
            label='evacuated'
        else:
            label='not evacuated'
        reflection_prompt=f'''During the fire, this resident {label} from the wildfire. Please reconsider and rethink original questions to provide another clear and logical rationales on why the resident {label}:'''
        
        reflection=conversation_with_summary.predict(input=reflection_prompt)
        return [agent_response, reflection]

    return [agent_response]


def known_f(memory, ind, demogra_type,label,epoch):
    threat_questions=threat_list[demogra_type][ind]
    risk_question=risk_list[demogra_type][ind]

    threat_prompts=threat_prompt1+threat_prompt2.format(survey=risk_question+threat_questions)
    threat=get_response('gpt-4o-2024-11-20',system,threat_prompts)
    
    risk_prompts=risk_prompt1+risk_prompt2.format(perception=threat, survey=risk_question)
    risk=get_response('gpt-4o-2024-11-20',system,risk_prompts)

    final_prmpt=final_decision1+risk+final_decision2+risk_question
    final_output=get_response_with_memory(memory, final_prmpt,label,epoch)
    final_decision=final_output[0]
    output=risk+threat+final_decision
    if 'YES' in final_decision:
        prediction=1
    else:
        prediction=0
    if len(final_output)>1:
        new_entry = {"input": 'Why did the resident evacuate or not evacuate?', "output": final_output[1]}
        append_dict_to_json(reflection_base_path, new_entry)

    return prediction, output

data_len=254


risk_list=[]
threat_list=[]

# risk & threat
q_index='40,32,2,13,21,20,59,12,7,1,23,49'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
q_index='40,10,2,20,59,5,55,7,13,50,32,47,12,33'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)

q_index='40,32,2,13,21,20,59,12,7,1,23,49'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
q_index='40,20,2,10,5,13,55,59,12,7,52,11,47,3,4'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)

q_index='40,20,2,13,32,49,21,50,59,58'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
q_index='40,10,2,20,59,5,55,7,13,50,32,47,12,33'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)

q_index='40,20,2,13,32,49,21,50,59,58'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
q_index='40,20,2,10,5,13,55,59,12,7,52,11,47,3,4'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)

# Prompt hub
threat_prompt1='''Analyze the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their responses to a wildfire survey, provide a brief summary of the resident's threat perception.'''
threat_prompt2='''Response to a wildfire survey: {survey}'''

risk_prompt1='''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Threat Perception and their responses to a wildfire survey, briefly summarize the resident's Risk Perception.'''
risk_prompt2=''' Threat Perception: {perception}. \n Response to a wildfire survey: {survey}.'''

final_decision1='''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Risk Perception Summary and their responses to a wildfire survey, determine whether the resident will decide to evacuate.

Your response should provide reasoning based on the information provided, and end with the evacuation decision in the following format: Answer: YES or Answer: NO.

Risk Perception Summary:
'''
final_decision2='''Response to a wildfire survey:
'''

system='You are an expert at rationale reasoning.'

pred=[]
res_list=[]
file_path='ml_results_cot_reflexion.csv'
reflection_base_path = 'ml_results_no_classi.json'
pdb.set_trace()
test_predictions = [random.randint(0, 3) for _ in range(data_len)]

for qid in tqdm(range(data_len)):
    memory_list = read_json_as_dict(reflection_base_path)
    reflection_memory=read_memory(memory_list)
    num_pred, res=known_f(reflection_memory, qid, test_predictions[qid], label_data[qid], qid)
    pred.append(num_pred)
    res_list.append(res)
append_csv(res_list, f'predict_res',file_path)
append_csv(pred, f'predict_value',file_path)
append_csv(label_data, f'actual_values',file_path)
