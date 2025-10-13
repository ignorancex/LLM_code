import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
import pdb
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

os.environ['OPENAI_API_KEY'] = ''
llm = ChatOpenAI(model="gpt-4o-2024-11-20",temperature=0.7)

def read_memory(memory_list):
    embedding_size = 1536   # OpenAIEmbeddings's dimension
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    for mem in memory_list:
        memory.save_context({"input": mem['input']}, {"output":  mem['output']})
    return memory

def read_json_as_dict(file_path):
    """
    Read JSON file and return contents as dictionary.
    
    :param file_path: Path to JSON file
    :return: File contents (returned as dictionary), if file doesn't exist or is empty, returns empty dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Ensure returned data type is dictionary
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return data
            else:
                return {"data": str(data)}
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"File '{file_path}' is not a valid JSON file or is empty.")
        return {}


def expand_multilabel_dataset(X, labels):
    """
    Expand multi-label dataset
    
    Parameters:
    X: Feature matrix
    labels: Label array for each sample
    
    Returns:
    expanded_X: Expanded feature matrix
    expanded_labels: Expanded label array
    """
    expanded_X = []
    expanded_labels = []
    
    for i in range(len(X)):
        # Get all labels for current sample
        sample_labels = np.where(labels[i] == 1)[0]
        
        # For each label, copy one sample
        for label in sample_labels:
            expanded_X.append(X.iloc[i])
            
            # Create label vector with only current label as 1
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
    # If file exists, read existing content
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = []

    # Ensure it's a list type (for adding new dictionaries)
    if not isinstance(data, list):
        data = [data]

    # Add the new dictionary to the list
    data.append(new_dict)

    # Write the updated list back to JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Initialize parameters
num_labels = 4             # 4 categories

# Read data
train_data_path = 'ml_all.csv'
data = pd.read_csv(train_data_path, delimiter=',', encoding='latin')

# Check and handle missing values
data = data.fillna(0)
X = data.iloc[:, :-6]  # All features
all_data = X[:350]
dff = pd.read_csv(train_data_path)

# Read and process labels
labels = dff['labels'].tolist()
labels = [str(int(lab)) for lab in labels]
labels = [parse_labels(label_str, num_labels) for label_str in labels]
labels = np.array(labels[:350])  # Note: Keep consistent length with feature matrix

# Expand training dataset
train_X, train_labels = expand_multilabel_dataset(all_data, labels)

# Train random forest classifier
train_labels_single = np.argmax(train_labels, axis=1)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

rf_classifier.fit(train_X, train_labels_single)

# Predict on test set
train_data_path = 'Marshall_kincade_test_encoded2_1.csv'
data = pd.read_csv(train_data_path, delimiter=',', encoding='latin')
# demographic
file_path='Marshall_kincade_test2.csv'
df=pd.read_csv(file_path,encoding='latin')

label_data=df['actual_values']

# Check and handle missing values
data = data.fillna(0)
X = data.iloc[:, :-1]  # All features
test_X=X
# Predict
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
        # If file exists, read existing data
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        # Update header row
        if data:
            data[0].append(header_name)
        else:
            data.append([header_name])

        # Calculate maximum number of rows
        max_rows = max(len(data), len(new_column_list) + 1)

        # Ensure data has enough rows
        for i in range(1, max_rows):
            if i < len(data):
                # If row exists, append new column data
                if i - 1 < len(new_column_list):
                    data[i].append(new_column_list[i - 1])
                else:
                    data[i].append('')
            else:
                # If row doesn't exist, create new row and add new column data
                row = [''] * (len(data[0]) - 1)
                if i - 1 < len(new_column_list):
                    row.append(new_column_list[i - 1])
                else:
                    row.append('')
                data.append(row)

        # Write updated data back to CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    else:
        # If file doesn't exist, create new file and write data
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([header_name])
            for value in new_column_list:
                writer.writerow([value])


def extract_demogra(file_path, column_indices_input):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Convert input string to integer list
    column_indices = [int(idx.strip()) - 1 for idx in column_indices_input.split(',')]
    # Extract specified columns and store column headers with data in list
    extracted_columns = []
    for idx in column_indices:
        column_name = df.columns[idx]
        column_data = [column_name] + df.iloc[:, idx].tolist()  # Insert header before data
        extracted_columns.append(column_data)
    
    demogra_list=[]
    for i in range(1, len(extracted_columns[0])):
        demogra_str=f'The resident is {extracted_columns[0][i]} and is {extracted_columns[1][i]}, and is {extracted_columns[2][i]} years old. '
        demogra_str+=f'The highest level of education they have completed is {extracted_columns[3][i]}. The total household income of this resident is {extracted_columns[4][i]}. '
        demogra_str+=f'The employment status of this resident is {extracted_columns[5][i]}. '
        demogra_str+=f'The resident has {extracted_columns[6][i]} children under the age of 13, {extracted_columns[7][i]} children between 13 years old to 17 years old. '
        demogra_str+=f'The resident has {extracted_columns[8][i]} adults between 19 and 64 years old living in the household. '
        demogra_str+=f'The resident has {extracted_columns[9][i]} adults over the age of 65 living in the household. '
        demogra_str+=f'The resident has {extracted_columns[10][i]} pets and {extracted_columns[11][i]} Livestock. '
        if extracted_columns[12][i]=='Yes':
            demogra_str+=f'The resident has a medical condition of {extracted_columns[13][i]}.'
        else:
            demogra_str+='The resident does not have any medical conditions.'
        demogra_list.append(demogra_str)
    return demogra_list

def extract_columns_csv(file_path, column_indices_input):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Convert input string to integer list
    column_indices = [int(idx.strip()) - 1 for idx in column_indices_input.split(',')]
    # Extract specified columns and store column headers with data in list
    extracted_columns = []
    for idx in column_indices:
        column_name = df.columns[idx]
        column_data = [column_name] + df.iloc[:, idx].tolist()  # Insert header before data
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
    # Read CSV file
    df = pd.read_csv(file_path)
    # Convert input string to integer list
    column_indices = [int(idx.strip()) - 1 for idx in column_indices_input.split(',')]
    # Extract specified columns and store column headers with data in list
    extracted_columns = []
    for idx in column_indices:
        column_name = df.columns[idx]
        column_data = [column_name] + df.iloc[:, idx].tolist()  # Insert header before data
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
            # Ensure returned data type is dictionary
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return data
            else:
                return {"data": str(data)}
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"File '{file_path}' is not a valid JSON file or is empty.")
        return {}


def get_response(model_type, system, question):
    response = client.chat.completions.create(
            model="o3-mini",

            messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
            ],
            #   logprobs=True,
            )
    return response.choices[0].message.content

def get_risk_threat_with_memory(memory, threatorisk, flag):
    
    # Add to examples
    history=memory.load_memory_variables({"prompt": threatorisk})["history"]
    output_ind=history.find('output')
    history=history[output_ind:]
    history_list=history.split('output:')
    history=''
    history_list=history_list[1:]
    for i in range(len(history_list)):
        history=history+f'\nThis is example {i+1}:'+history_list[i]
    
    prompt_tpl = f'''You have access to a post-wildfire survey completed by local residents who experienced a specific wildfire event. Your task is to score the resident's {flag} perception on a scale of 1 to 5 based on their {flag} perception. You will be provided with previous scoring examples that have similar informations.
    **Previous Examples**:\n'''+history+f'''\n
    **This is the {flag} perception of the current local resident is**:
    '''+threatorisk
    this_system=f'''Your output should be this format: Based on this assessment, their {flag} perception score on a scale of 1 to 5 is: [1-5]'''
    agent_response=get_response('gpt-4o-2024-11-20', this_system, prompt_tpl)
    agent_output=threatorisk+'\n'+agent_response
    score=agent_response[agent_response.find('is: ')+4:]
    return agent_output,score

def get_response_with_memory(memory, risk, enviroment, label, epo):
    if 'The respondent did not answer to this question.' in enviroment[epo]:
        other_info=''
    else:
        other_info=enviroment[epo][enviroment[epo].find('question is:')+12:]
    # other_info+=' The demographic information of the resident is: ' + demo_list[epo]
    # Add to examples
    history=memory.load_memory_variables({"prompt": risk})["history"]
    output_ind=history.find('output')
    history=history[output_ind:]
    history_list=history.split('output:')
    history=''
    history_list=history_list[1:]
    for i in range(len(history_list)):
        history=history+f'\nThis is example {i+1}:'+history_list[i]
    
    prompt_tpl = '''You have access to a post-wildfire survey completed by local residents who experienced a specific wildfire event. Your task is to generate a logical, step-by-step chain of thought to infer whether the resident evacuated during the wildfire. Ensure each step is clearly connected. You must conclude with a definitive YES or NO answer regarding whether the resident evacuated. You will be provided with previous successful examples that have similar informations. You may reference the rationale from these examples in your analysis.
    **Previous Examples**:\n'''+history+'''\n
    **This is the risk perception and other information of the current local resident is**:
    '''+risk+'\n'+other_info
    this_system='You are an advanced reasoning agent capable of self-improvement through reflection.'
    agent_response=get_response('gpt-4o-2024-11-20', this_system, prompt_tpl)

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
        reflection=f'''During the fire, this resident {label} from the wildfire. Their risk perception and other information is:{risk}\n{other_info}'''
        return [agent_response, reflection]

    return [agent_response]


def known_f(memory, ind, demogra_type,label,enviroment):
    threat_score=0
    risk_score=0
    threat_questions=threat_list[demogra_type][ind]
    risk_question=risk_list[demogra_type][ind]

    threat_prompts=threat_prompt1+threat_prompt2.format(survey=risk_question+threat_questions)
    threat=get_response('gpt-4o-2024-11-20',system,threat_prompts)
    if ind < data_len*0.7:
        if threat_level[test_predictions[demogra_type]][qid][-2]=='d':
            threat_value='5'
        else:
            threat_value=threat_level[demogra_type][ind][-2]
        threat = threat + '\nBased on this assessment, their threat perception score on a scale of 1 to 5 is: '+threat_value
        threat_dict={"input": 'threat_question', "output": threat}
        append_dict_to_json(threat_base, threat_dict)
    else:
        threat,threat_score=get_risk_threat_with_memory(reflection_memory2, threat, 'threat')
    risk_prompts=risk_prompt1+risk_prompt2.format(perception=threat, survey=risk_question)
    risk=get_response('gpt-4o-2024-11-20',system,risk_prompts)
    if ind < data_len*0.7:
        risk = risk + '\nBased on this assessment, their risk perception score on a scale of 1 to 5 is: '+risk_level[demogra_type][ind][-2]
        risk_dict={"input": 'risk_question', "output": risk}
        append_dict_to_json(risk_base, risk_dict)
    else:
        risk,risk_score=get_risk_threat_with_memory(reflection_memory3, risk, 'risk')

    final_output=get_response_with_memory(memory, risk, enviroment,label,ind)
    final_decision=final_output[0]
    output=threat+'\n' +risk+'\n'+final_decision
    if 'YES' in final_decision:
        prediction=1
    else:
        prediction=0
    if len(final_output)>1:
        # Example: Add a new dictionary to json file
        new_entry = {"input": 'Why did the resident evacuate or not evacuate?', "output": final_output[1]}
        append_dict_to_json(reflection_base_path, new_entry)

    return prediction, output, threat_score, risk_score

## Extract Questions
data_len=254


risk_list=[]
threat_list=[]
threat_level=[]
risk_level=[]

# demographic
q_index='47,48,49,50,51,52,53,54,55,56,57,58,59,60,61'
demo_list=extract_demogra(file_path, q_index)


# risk & threat
q_index='32,2,13,21,20,59,12,7,1,23,49'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
threat_value=extract_columns_csv(file_path,'41')
threat_level.append(threat_value)
q_index='10,2,20,59,5,55,7,13,50,32,47,12,33'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)
risk_value=extract_columns_csv(file_path,'43,45')
risk_level.append(risk_value)

q_index='32,2,13,21,20,59,12,7,1,23,49'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
threat_value=extract_columns_csv(file_path,'41')
threat_level.append(threat_value)
q_index='20,2,10,5,13,55,59,12,7,52,11,47,3,4'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)
risk_value=extract_columns_csv(file_path,'44,46')
risk_level.append(risk_value)

q_index='20,2,13,32,49,21,50,59,58'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
threat_value=extract_columns_csv(file_path,'42')
threat_level.append(threat_value)
q_index='10,2,20,59,5,55,7,13,50,32,47,12,33'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)
risk_value=extract_columns_csv(file_path,'43,45')
risk_level.append(risk_value)

q_index='20,2,13,32,49,21,50,59,58'
threat_questions=extract_columns_csv(file_path,q_index)
threat_list.append(threat_questions)
threat_value=extract_columns_csv(file_path,'42')
threat_level.append(threat_value)
q_index='20,2,10,5,13,55,59,12,7,52,11,47,3,4'
risk_questions=extract_columns_csv(file_path,q_index)
risk_list.append(risk_questions)
risk_value=extract_columns_csv(file_path,'44,46')
risk_level.append(risk_value)

# Environment info
q_index='61'
enviroment_info=extract_columns_csv(file_path,q_index)

# Prompt hub
threat_prompt1='''Analyze the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their responses to a wildfire survey, provide a brief summary of the resident's threat perception.'''
threat_prompt2='''Response to a wildfire survey: {survey}'''

risk_prompt1='''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Threat Perception and their responses to a wildfire survey, briefly summarize the resident's Risk Perception.'''
risk_prompt2=''' Threat Perception: {perception}. \n Response to a wildfire survey: {survey}.'''

final_decision1='''Consider the following scenario: A resident is deciding whether to evacuate during a wildfire. Based on their Risk Perception Summary, and other information, determine whether the resident will decide to evacuate.

Your response should provide reasoning based on the information provided, and end with the evacuation decision in the following format: Answer: YES or Answer: NO.

Risk Perception Summary:
'''
final_decision2='''Other information from the resident:
'''

system='You are an expert at rationale reasoning.'

pred=[]
res_list=[]
t_score_list=[]
r_score_list=[]
ori_t_score_list=[]
ori_r_score_list=[]
file_path='alldata_finaprediction_4o.csv'
reflection_base_path = 'alldata_reflexion_base.json'
threat_base='alldata_threat_score.json'
risk_base='alldata_risk_score.json'
memory_list = read_json_as_dict(reflection_base_path)
reflection_memory=read_memory(memory_list)
memory_list = read_json_as_dict(threat_base)
reflection_memory2=read_memory(memory_list)
memory_list = read_json_as_dict(risk_base)
reflection_memory3=read_memory(memory_list)

for qid in tqdm(range(round(data_len*0.7),data_len)):
    if threat_level[test_predictions[qid]][qid][-2]=='d':
        ori_t_score_list.append(threat_level[test_predictions[qid]][qid][-18])
    else:
        ori_t_score_list.append(risk_level[test_predictions[qid]][qid][-2])
    risk_value1=risk_level[test_predictions[qid]][qid][risk_level[test_predictions[qid]][qid].find('n is: ')+6:risk_level[test_predictions[qid]][qid].find('n is: ')+7]
    risk_value=int(risk_value1)+int(risk_level[test_predictions[qid]][qid][-2])/2
    ori_r_score_list.append(risk_value)
    num_pred, res, t_score, r_score=known_f(reflection_memory, qid, test_predictions[qid], label_data[qid], enviroment_info)
    t_score_list.append(t_score)
    r_score_list.append(r_score)
    pred.append(num_pred)
    res_list.append(res)
append_csv(res_list, f'predict_res',file_path)
append_csv(pred, f'predict_value',file_path)
append_csv(label_data, f'actual_values',file_path)
append_csv(t_score_list, f'threat_prediction',file_path)
append_csv(r_score_list, f'risk_prediction',file_path)
append_csv(ori_t_score_list, f'threat',file_path)
append_csv(ori_r_score_list, f'risk',file_path)

