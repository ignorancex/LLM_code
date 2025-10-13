import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

import numpy as np
import pandas as pd
import argparse
import sys
from tqdm import tqdm

# Get the directory containing projection.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pprint import pprint

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from attacker_models import SequenceCrossEntropyLoss
from sentence_transformers import SentenceTransformer
from simcse_persona import get_persona_dict
# from attacker_evaluation_gpt import eval_on_batch
from datasets import load_dataset
import baseline_models

from sklearn import metrics
import logging
import logging.handlers
import json
from data_process import get_sent_list

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler('logs/new_datasets.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s -  %(message)s"))
logger.addHandler(f_handler)



def save_blmodel(model, path):
    torch.save(model.state_dict(), path)


def load_blmodel(model, path):
    model.load_state_dict(torch.load(path))


class sent_list_dataset(Dataset):
    def __init__(self, sent_list, label):
        assert len(sent_list) == len(label)
        self.sent_list = sent_list
        self.label = label

    def __len__(self):
        return len(self.sent_list)

    def __getitem__(self, index):
        sentence = self.sent_list[index]
        label = self.label[index]

        return sentence, label

    def collate(self, unpacked_data):
        return unpacked_data

class collated_dataset(Dataset):
    def __init__(self, sent_list, config):
        
        self.sent_list = sent_list
        self.config = config


    def __len__(self):
        return len(self.sent_list)

    def __getitem__(self, index):
        sentence = self.sent_list[index]

        return sentence

    def collate(self, unpacked_data):
        #pprint(unpacked_data)
        dial_tokens_np, input_labels = process_sent_list(unpacked_data, self.config)
        #print(dial_tokens_np.shape)
        #print(input_labels.shape)
        return unpacked_data, input_labels

def process_sent_list(sent_list, config):
    tokenizer = config['tokenizer']
    turn_ending = config['eos_token']
    # print(tokenizer)
    # dial_tokens = [tokenizer.encode(item) + turn_ending for item in conv]
    dial_tokens = [tokenizer.encode(item,max_length=150,padding='max_length', truncation=True) for item in sent_list]
    # for input loss training
    token_num = len(tokenizer)
    # print('lenth of tokenizer ', end='')
    # print(token_num)
    dial_tokens_np = np.array(dial_tokens)
    input_labels = []
    for i in dial_tokens_np:
        temp_i = np.zeros(token_num)
        temp_i[i] = 1
        input_labels.append(temp_i)
    input_labels = np.array(input_labels)

    return dial_tokens_np, input_labels


'''
    choosing baseline attacker models here
'''


def init_baseline_model(config, embed_model_dim, type='NN'):
    tokenizer = config['tokenizer']
    token_num = len(tokenizer)
    device = config['device']
    if type == 'NN':
        # print(f'line 112 embed_model_dim:{embed_model_dim}')
        baseline_model = baseline_models.baseline_NN(out_num=token_num, in_num=embed_model_dim)
    elif type == 'RNN':
        baseline_model = baseline_models.baseline_RNN(out_num=token_num, in_num=embed_model_dim)


    optimizer = AdamW(baseline_model.parameters(),
                      lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()  # usage criterion(external_out,input_label)
    baseline_model.to(device)
    return baseline_model, optimizer, criterion


'''
Sentence bert based:
T5: sentence-t5-large                   dim 768
mpnet: all-mpnet-base-v1                dim 768
Roberta: all-roberta-large-v1           dim 1024


SIMCSE based:
princeton-nlp/unsup-simcse-roberta-large        dim 1024
princeton-nlp/sup-simcse-bert-large-uncased     dim 1024
'''


def get_embedding(dataloader, config, eval=False):
    model_cards = {}
    model_cards['sent_t5'] = 'sentence-t5-large'
    model_cards['mpnet'] = 'all-mpnet-base-v1'
    model_cards['sent_roberta'] = 'all-roberta-large-v1'
    model_cards['simcse_bert'] = 'princeton-nlp/sup-simcse-bert-large-uncased'
    model_cards['simcse_roberta'] = 'princeton-nlp/unsup-simcse-roberta-large'
    embed_model = config['embed_model']
    embed_model_dim = 1024
    assert embed_model in ['mpnet', 'sent_roberta', 'simcse_bert', 'simcse_roberta', 'sent_t5']
    model_name = model_cards[embed_model]
    if embed_model in ['mpnet', 'sent_roberta', 'sent_t5']:
        if embed_model in ['mpnet', 'sent_t5']:
            embed_model_dim = 768
            # print(f'line 146 embed_model_dim:{embed_model_dim}')
        if eval:
            eval_sent(dataloader, model_name, embed_model_dim, config)
        else:
            train_sent(dataloader, model_name, embed_model_dim, config)
    elif embed_model in ['simcse_bert', 'simcse_roberta']:
        if (eval):
            eval_simcse(dataloader, model_name, embed_model_dim, config)
        else:
            train_simcse(dataloader, model_name, embed_model_dim, config)


def train_sent(dataloader, model_name, embed_model_dim, config):
    device = config['device']
    num_epochs = config['num_epochs']
    model = SentenceTransformer(model_name, device=device)
    type = config['model_type']
    baseline_model, optimizer, criterion = init_baseline_model(config, embed_model_dim, type=type)
    save_path = 'blmodels/' + type + '_' + config['dataset'] + '_' + config['embed_model']

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f'save_path:{save_path}')

    hx = torch.zeros(64, 512).cuda()
    for i in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):

        for idx, batch in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):
            # print(batch)
            batch_text, batch_label = batch
            # print(batch_label)
            batch_label = torch.tensor(batch_label)
            batch_label = batch_label.to(device)
            batch_text = list(batch_text)
            # print(f'{i}: Batch id: {idx}.   batch_text: {batch_text[0]}.     batch_label: {batch_label.size()}.')
            print(f'{i}: Batch id: {idx}.', end=' ')
            ### no grad for embedding model
            with torch.no_grad():
                embeddings = model.encode(batch_text, convert_to_tensor=True)
            # print(f'embeddings:{embeddings.size()}')
            # embeddings:torch.Size([64, 768, 1])
            # print(f'batch_label:{batch_label.size()}')
            # batch_label:torch.Size([64, 50257, 1])
            # print(save_path)
            # exit()

            # here is the embedding sentences, we need to turn it into tensor for future use
            if type == 'NN':
                train_on_batch(baseline_model, optimizer, criterion, embeddings, batch_label)
            elif type == 'RNN':
                total_loss = baseline_model(embeddings, hx, batch_label)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print(f'========total_loss:{total_loss}')
    save_blmodel(baseline_model, save_path)


def train_simcse(dataloader, model_name, embed_model_dim, config):
    device = config['device']
    num_epochs = config['num_epochs']
    type = config['model_type']
    baseline_model, optimizer, criterion = init_baseline_model(config, embed_model_dim, type=type)
    save_path = 'blmodels/' + type + '_' + config['dataset'] + '_' + config['embed_model']
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f'save_path:{save_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    hx = torch.zeros(64, 512).cuda()
    for i in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
        for idx, batch in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):
            batch_text, batch_label = batch
            batch_label = torch.tensor(batch_label)
            batch_label = batch_label.to(device)
            batch_text = list(batch_text)
            print(f'{i}: Batch id: {idx}.   batch_text: {batch_text[0]}.     batch_label: {batch_label.size()}.')
            with torch.no_grad():
                inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            print(f'embeddings:{embeddings.size()}')
            print(save_path)
            if type == 'NN':
                train_on_batch(baseline_model, optimizer, criterion, embeddings, batch_label)
            elif type == 'RNN':
                total_loss = baseline_model(embeddings, hx, batch_label)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print(f'========total_loss:{total_loss}')
    save_blmodel(baseline_model, save_path)


def train_on_batch(baseline_model, optimizer, criterion, embedding_batch, label_batch):
    logits = baseline_model(embedding_batch)
    loss = criterion(logits, label_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'========loss:{loss}')


def report_score(y_true, y_pred):
    # micro result should be reported
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    logger.info(f"micro precision_score on token level: {precision}")
    print(f"micro precision_score on token level: {precision}")
    print("macro precision_score: {:.2f} ".format( metrics.precision_score(y_true, y_pred, average='macro')))
    print('==' * 20)
    logger.info(f"micro recall_score on token level: {recall}")
    print(f"micro recall_score on token level: {recall}")
    print("macro recall_score: {:.2f} ".format( metrics.recall_score(y_true, y_pred, average='macro')))
    print('==' * 20)
    logger.info(f"micro f1_score on token level: {f1}")
    print(f"micro f1_score on token level: {f1}")
    print("macro f1_score: {:.2f} ".format( metrics.f1_score(y_true, y_pred, average='macro')))


def eval_sent(dataloader, model_name, embed_model_dim, config):
    device = config['device']
    num_epochs = config['num_epochs']
    model = SentenceTransformer(model_name, device=device)
    type = config['model_type']
    hx = torch.zeros(64, 512).cuda()

    baseline_model, optimizer, criterion = init_baseline_model(config, embed_model_dim, type=type)
    model_path = 'blmodels/' + type + '_' + config['dataset'] + '_' + config['embed_model']
    baseline_model.load_state_dict(torch.load(model_path))
    baseline_model.eval()

    print(f'load baseline {type} from {model_path}')
    # load_blmodel(baseline_model,model_path)
    log_text = model_path
    logger.info('=======New baseline model evaluation=========')
    logger.info(log_text)
    predict = []
    ground_truth = []
    input_text = []
    for idx, batch in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):
        # print(batch)
        batch_text, batch_label = batch
        batch_label = torch.tensor(batch_label)
        batch_label = batch_label.to(device)
        batch_text = list(batch_text)
        # print(f'Batch id: {idx}.   batch_text: {batch_text[0]}.     batch_label: {batch_label.size()}.')
        print(f'Batch id: {idx}.',end=" ")
        ### no grad for embedding model
        with torch.no_grad():
            embeddings = model.encode(batch_text, convert_to_tensor=True)
            # print(f'embeddings:{embeddings.size()}')
            # embeddings:torch.Size([64, 1024])    [batch size, feature(dim)]
            if type == 'NN':
                predict_result = eval_on_batch(baseline_model, criterion, embeddings, batch_label, config['threshold'])
            elif type == 'RNN':
                zero_tensor = torch.zeros(64, 50257).cuda()
                # one_tensor = torch.ones(64, 50257).cuda()
                predicted_token_index_batch_tensor = baseline_model(embeddings, hx, batch_label, eval=True)
                predict_result = zero_tensor.scatter_(1, predicted_token_index_batch_tensor, value = 1)
            predict.extend(predict_result.cpu().detach().numpy())
            ground_truth.extend(batch_label.cpu().detach().numpy())
            input_text.extend(batch_text)
    eval_label(predict, ground_truth, input_text, config, type=type)
    report_score(ground_truth, predict)


def eval_simcse(dataloader, model_name, embed_model_dim, config):
    hx = torch.zeros(64, 512).cuda()
    device = config['device']
    num_epochs = config['num_epochs']
    type = config['model_type']
    baseline_model, optimizer, criterion = init_baseline_model(config, embed_model_dim, type=type)
    model_path = 'blmodels/' + type + '_' + config['dataset'] + '_' + config['embed_model']
    baseline_model.load_state_dict(torch.load(model_path))
    baseline_model.eval()
    log_text = model_path
    logger.info('=======New baseline model evaluation=========')
    logger.info(log_text)
    print(f'load baseline {type} from {model_path}')
    # load_blmodel(baseline_model,model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    predict = []
    ground_truth = []
    input_text = []
    for idx, batch in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):
        batch_text, batch_label = batch
        batch_label = torch.tensor(batch_label)
        batch_label = batch_label.to(device)
        batch_text = list(batch_text)
        print(f'Batch id: {idx}.   batch_text: {batch_text[0]}.     batch_label: {batch_label.size()}.')
        with torch.no_grad():
            inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
            # print(f'inputs:{inputs.size()}')
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            print(f'embeddings:{embeddings.size()}')

            # batch_pred_labels = eval_on_batch(baseline_model,criterion,embeddings,batch_label)
            if type == 'NN':
                predict_result = eval_on_batch(baseline_model, criterion, embeddings, batch_label, config['threshold'])
            elif type == 'RNN':
                zero_tensor = torch.zeros(64, 50257).cuda()
                # one_tensor = torch.ones(64, 50257).cuda()
                predicted_token_index_batch_tensor = baseline_model(embeddings, hx, batch_label, eval=True)
                predict_result = zero_tensor.scatter_(1, predicted_token_index_batch_tensor, value=1)
            predict.extend(predict_result.cpu().detach().numpy())
            ground_truth.extend(batch_label.cpu().detach().numpy())
            input_text.extend(batch_text)
    eval_label(predict, ground_truth, input_text, config, type=type)
    report_score(ground_truth, predict)


def eval_on_batch(baseline_model, criterion, embedding_batch, label_batch, threshold):
    logits = baseline_model(embedding_batch, eval=True)
    loss = criterion(logits, label_batch)
    # print(logits)
    # print(type(logits))
    logits[logits >= threshold] = 1
    logits[logits < threshold] = 0
    print(f'========eval loss:{loss}')
    return logits


'''
Used for case study to see what are outputted by baselines                                                   
pred labels are (N,dim) 0,1 vectors, where 1 indicate token i show up 
'''


def eval_label(pred_labels, ground_truth, input_text, config, type='NN'):
    tokenizer = config['tokenizer']
    
    # Convert inputs to tensors more efficiently using numpy first
    if not torch.is_tensor(pred_labels):
        pred_labels = torch.from_numpy(np.array(pred_labels))
    if not torch.is_tensor(ground_truth):
        ground_truth = torch.from_numpy(np.array(ground_truth))
    
    # Get indices where values are 1
    pred_indices = torch.nonzero(pred_labels, as_tuple=True)
    gt_indices = torch.nonzero(ground_truth, as_tuple=True)
    
    # Create save path based on type
    if type == 'NN':
        threshold = config['threshold']
        str_threshold = f'{threshold:.2f}'
        save_path = f'logs/test_{type}_{config["dataset"]}_{config["embed_model"]}_threshold_{str_threshold}.label'
    else:
        save_path = f'logs/test_{type}_{config["dataset"]}_{config["embed_model"]}.label'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    

    BATCH_SIZE = config['batch_size'] 
    
    pred_tokens = []
    for i in range(0, len(pred_indices[1]), BATCH_SIZE):
        # Ensure end index doesn't exceed array length
        end_idx = min(i + BATCH_SIZE, len(pred_indices[1]))
        batch = pred_indices[1][i:end_idx].reshape(-1, 1)
        pred_tokens.extend(tokenizer.batch_decode(batch))
    
    gt_tokens = []
    for i in range(0, len(gt_indices[1]), BATCH_SIZE):
        # Ensure end index doesn't exceed array length
        end_idx = min(i + BATCH_SIZE, len(gt_indices[1]))
        batch = gt_indices[1][i:end_idx].reshape(-1, 1)
        gt_tokens.extend(tokenizer.batch_decode(batch))
    
    # Group tokens by sample index
    pred_grouped = [[] for _ in range(len(input_text))]
    gt_grouped = [[] for _ in range(len(input_text))]
    
    # Use tensor operations to group tokens
    for idx, token in zip(pred_indices[0].tolist(), pred_tokens):
        pred_grouped[idx].append(token)
    for idx, token in zip(gt_indices[0].tolist(), gt_tokens):
        gt_grouped[idx].append(token)
    
    # Create save data structure
    save_data = [
        {
            'pred': pred_grouped[i],
            'gt': gt_grouped[i],
            'input': input_text[i]
        }
        for i in range(len(input_text))
    ]
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=4)


if __name__ == '__main__':
    '''
    Sentence bert based:
    T5: sentence-t5-large                   dim 768
    mpnet: all-mpnet-base-v1                dim 768
    Roberta: all-roberta-large-v1           dim 1024
    SIMCSE based:
    princeton-nlp/unsup-simcse-roberta-large        dim 1024
    princeton-nlp/sup-simcse-bert-large-uncased     dim 1024

    list of supported datasets:
    ['personachat'.'qnli','mnli','sst2','wmt16','multi_woz','abcd']
    '''
    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--model_dir', type=str, default='baseline_weights/DialoGPT-medium', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoches.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch_size #.')
    parser.add_argument('--dataset', type=str, default='abcd', help="List of supported datasets: ['personachat'.'qnli','mnli','sst2','wmt16','multi_woz','abcd']")
    # parser.add_argument('--dataset', type=str, default='qnli', help='Name of dataset: personachat or qnli')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    #parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--embed_model', type=str, default='simcse_bert',
                        help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
    # parser.add_argument('--embed_model', type=str, default='simcse_roberta', help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
    parser.add_argument('--model_type', type=str, default='NN', help='Type of baseline model: RNN or NN')
    # parser.add_argument('--eval', type=str, default=False, help='True or False')
    
    args = parser.parse_args()
    config = {}
    if args.dataset == 'personachat':
        model_thres = {'mpnet': 0.2, 'sent_roberta': 0.2, 'simcse_bert': 0.5, 'simcse_roberta': 0.5, 'sent_t5': 0.1}
    elif args.dataset == 'qnli':
        model_thres = {'mpnet': 0.45, 'sent_roberta': 0.2, 'simcse_bert': 0.6, 'simcse_roberta': 0.75, 'sent_t5': 0.2}
    if args.data_type == 'train':
        config['model_dir'] = args.model_dir
        config['num_epochs'] = args.num_epochs
        config['batch_size'] = args.batch_size
        config['dataset'] = args.dataset
        config['data_type'] = args.data_type
        config['embed_model'] = args.embed_model
        config['model_type'] = args.model_type
        
        config['threshold'] = model_thres[args.embed_model]

        config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config['tokenizer'] = AutoTokenizer.from_pretrained(args.model_dir)
        config['eos_token'] = config['tokenizer'].eos_token
        config['tokenizer'].pad_token = config['eos_token']
        sent_list = get_sent_list(config)

        #dataset = sent_list_dataset(sent_list, onehot_labels)
        dataset = collated_dataset(sent_list,config)
        dataloader = DataLoader(dataset=dataset,
                                shuffle=False,
                                batch_size=config['batch_size'],
                                collate_fn=dataset.collate,
                                drop_last=True)
        get_embedding(dataloader, config, eval=False)
    
    # for evaluation
    config['model_dir'] = args.model_dir
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['dataset'] = args.dataset
    config['data_type'] = 'test'
    config['embed_model'] = args.embed_model
    config['model_type'] = args.model_type
    config['threshold'] = model_thres[args.embed_model]

    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['tokenizer'] = AutoTokenizer.from_pretrained(args.model_dir)
    config['eos_token'] = config['tokenizer'].eos_token
    config['tokenizer'].pad_token = config['eos_token']
    sent_list = get_sent_list(config)

    #dataset = sent_list_dataset(sent_list, onehot_labels)
    dataset = collated_dataset(sent_list,config)
    dataloader = DataLoader(dataset=dataset,
                            shuffle=False,
                            batch_size=config['batch_size'],
                            collate_fn=dataset.collate,
                            drop_last=True)
    get_embedding(dataloader, config, eval=True)
