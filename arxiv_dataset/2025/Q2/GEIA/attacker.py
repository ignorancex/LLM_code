import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

import json
import numpy as np
import pandas as pd
import argparse
import sys

from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,T5Config,
    AdamW, get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
from attacker_models import SequenceCrossEntropyLoss
from sentence_transformers import SentenceTransformer
from simcse_persona import get_persona_dict
from attacker_evaluation_gpt import eval_on_batch
from datasets import load_dataset
from data_process import get_sent_list



class linear_projection(nn.Module):
    
    def __init__(self, in_num, out_num=1024):
        super(linear_projection, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,in_num) according to gpt2 output
        out_shape = x.size()[-1]
        assert(x.size()[1] == out_shape)
        out = self.fc1(x)
        return out


def init_gpt2():
    config = GPT2Config.from_pretrained('microsoft/DialoGPT-medium')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def init_opt():
    config = T5Config.from_pretrained('google/t5-large-lm-adapt')
    model = T5ForConditionalGeneration(config)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-large-lm-adapt")
    return model, tokenizer


class personachat(Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text = self.data[index]

        return  text
        
    def collate(self, unpacked_data):
        return unpacked_data


def process_data(
        data, 
        batch_size: int, 
        device: torch.device, 
        config: dict, 
        attacker_emb_size: int
    ):

    # Victim model (model f)
    model = SentenceTransformer(config['embed_model_path'], device=device)   # dim 768
    dataset = personachat(data)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('Load data done')

    embedding_dimension: int = 768
    if config['embed_model_path'] == 'all-roberta-large-v1':
        embedding_dimension = 1024
    elif config['embed_model_path'] == 'sent_t5_large':
        embedding_dimension = 1024
    elif config['embed_model_path'].find('simcse') != -1:
        embedding_dimension = 1024

    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    # Extra Projection; aligning model f (victim) with the attacker
    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size).to(device)
    
    # Attacker Model
    if config['model_dir'] == 'random_gpt2_medium':
        model_attacker, tokenizer_attacker = init_gpt2()
    else:
        model_attacker = AutoModelForCausalLM.from_pretrained(config['model_dir'])
        tokenizer_attacker = AutoTokenizer.from_pretrained(config['model_dir'])
    
    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(device)
    param_optimizer = list(model_attacker.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_gradients_accumulation = 1
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_train_optimization_steps  = len(dataloader) * num_epochs // num_gradients_accumulation
    optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)
    if need_proj:
        optimizer.add_param_group({'params': projection.parameters()})
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
    
    # 1. Get the embeddings from victim on text
    # 2. Pass through the projection (if embedding spaces don't match)
    # 3. Train the attacker, by:
    #     3.a Taking the sentence embedding and generating text
    #     3.b Calculate the loss compared of the generated text, and the GT text
    for i in range(num_epochs):
        model.eval()
        for idx, batch_text in enumerate(dataloader):
            
            with torch.no_grad():       
                embeddings = model.encode(batch_text, convert_to_tensor=True).to(device)
                # print(f'Embedding dim: {embeddings.size()}')

            # attacker part, needs training
            if need_proj:
               embeddings = projection(embeddings)

            record_loss, perplexity = train_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=model_attacker,
                tokenizer=tokenizer_attacker,
                criterion=criterion,
                device=device,
                train=True
            )
            optimizer.step()
            scheduler.step()
            # make sure no grad for GPT optimizer
            optimizer.zero_grad()
            print(f'Training: epoch {i} batch {idx} with loss: {record_loss} and PPL {perplexity} with size {embeddings.size()}')
        
        if need_proj:
            torch.save(projection.state_dict(), config['projection_save_path'])
        model_attacker.save_pretrained(config['attacker_save_path'])


def process_data_simcse(
        data,
        batch_size: int,
        device: torch.device,
        config: dict,
        attacker_emb_size: int
    ):
    embed_model_name = config['embed_model']
    tokenizer = AutoTokenizer.from_pretrained(config['embed_model_path'])
    model = AutoModel.from_pretrained(config['embed_model_path']).to(device)
    dataset = personachat(data)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')

    # TODO: Create a function for this to be re-usable
    embedding_dimension: int = 768
    if config['embed_model_path'] == 'all-roberta-large-v1':
        embedding_dimension = 1024
    elif config['embed_model_path'] == 'sent_t5_large':
        embedding_dimension = 1024
    elif config['embed_model_path'].find('simcse') != -1:
        embedding_dimension = 1024

    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    ### extra projection
    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size).to(device)

    ### for attackers
    ### TODO: make a function to create the model/tokenizer parametrized on model_dir
    if config['model_dir'] == 'random_gpt2_medium':
        model_attacker, tokenizer_attacker = init_gpt2()
    else:
        model_attacker = AutoModelForCausalLM.from_pretrained(config['model_dir'])
        tokenizer_attacker = AutoTokenizer.from_pretrained(config['model_dir'])
    
    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(device)
    param_optimizer = list(model_attacker.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_gradients_accumulation = 1
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_train_optimization_steps  = len(dataloader) * num_epochs // num_gradients_accumulation
    optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)
    if need_proj:
        optimizer.add_param_group({'params': projection.parameters()})
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
    
    ### process to obtain the embeddings
    for i in range(num_epochs):
        for idx,batch_text in enumerate(dataloader):
            with torch.no_grad():           
                inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output     
                print(embeddings.size())

            ### attacker part, needs training
            if need_proj:
               embeddings = projection(embeddings)

            record_loss, perplexity = train_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=model_attacker,
                tokenizer=tokenizer_attacker,
                criterion=criterion,
                device=device,
                train=True
            )
            optimizer.step()
            scheduler.step()
            # make sure no grad for GPT optimizer
            optimizer.zero_grad()
            print(f'{embed_model_name}: Training: epoch {i} batch {idx} with loss: {record_loss} and PPL {perplexity} with size {embeddings.size()}')
        
        if need_proj:
            torch.save(projection.state_dict(), config['projection_save_path'])
        
        model_attacker.save_pretrained(config['attacker_save_path'])



### used for testing only
def process_data_test(
        data,
        batch_size: int,
        device: torch.device,
        config: dict,
        attacker_emb_size: int
    ):
    model = SentenceTransformer(config['embed_model_path'], device=device)
    
    if(config['decode'] == 'beam'):
        save_path = "logs/" + config['attacker_save_name'] + '_beam.log'
    else:
        save_path = "logs/" + config['attacker_save_name'] + '.log'
    
    dataset = personachat(data)
    # no shuffle for testing data
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')

    embedding_dimension: int = 768
    if config['embed_model_path'] == 'all-roberta-large-v1':
        embedding_dimension = 1024
    elif config['embed_model_path'] == 'sent_t5_large':
        embedding_dimension = 1024
    elif config['embed_model_path'].find('simcse') != -1:
        embedding_dimension = 1024

    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size)
        projection.load_state_dict(torch.load(config['projection_save_path']))
        projection.to(device)
        print('load projection done')
    else:
        print('no projection loaded')
    
    # setup on config for sentence generation   AutoModelForCausalLM
    config['model'] = AutoModelForCausalLM.from_pretrained(config['attacker_save_path']).to(device)
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():  
        for idx,batch_text in enumerate(dataloader):

            embeddings = model.encode(batch_text, convert_to_tensor=True).to(device)
  
            if need_proj:
                embeddings = projection(embeddings)

            sent_list, gt_list = eval_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=config['model'],
                tokenizer=config['tokenizer'],
                device=device,
                config=config
            )    
            print(f'Testing {idx} batch done with {idx*batch_size} samples')
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)
        
        with open(save_path, 'w') as f:
            json.dump(sent_dict, f, indent=4)

    return 0
        

### used for testing only
def process_data_test_simcse(
        data,
        batch_size: int,
        device: torch.device,
        config: dict,
        attacker_emb_size: int
    ):
    tokenizer = AutoTokenizer.from_pretrained(config['embed_model_path'])
    model = AutoModel.from_pretrained(config['embed_model_path']).to(device)

    if(config['decode'] == 'beam'):
        print('Using beam search decoding')
        save_path = 'logs/' + config['attacker_save_name'] + '_beam.log'
    else:
        save_path = 'logs/' + config['attacker_save_name'] +'.log'
    
    dataset = personachat(data)
    # no shuffle for testing data
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    print('load data done')
    
    embedding_dimension: int = 768
    if config['embed_model_path'] == 'all-roberta-large-v1':
        embedding_dimension = 1024
    elif config['embed_model_path'] == 'sent_t5_large':
        embedding_dimension = 1024
    elif config['embed_model_path'].find('simcse') != -1:
        embedding_dimension = 1024

    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size)
        projection.load_state_dict(torch.load(config['projection_save_path']))
        projection.to(device)
        print('load projection done')
    else:
        print('no projection loaded')
    
    # setup on config for sentence generation AutoModelForCausalLM
    config['model'] = AutoModelForCausalLM.from_pretrained(config['attacker_save_path']).to(device)
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():  
        for idx,batch_text in enumerate(dataloader):
            inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output  
            
            if need_proj:
                embeddings = projection(embeddings)

            sent_list, gt_list = eval_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=config['model'],
                tokenizer=config['tokenizer'],
                device=device,
                config=config
            ) 
            print(f'Testing {idx} batch done with {idx*batch_size} samples')
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)
        
        with open(save_path, 'w') as f:
            json.dump(sent_dict, f,indent=4)

    return 0


def train_on_batch(
        batch_X: torch.TensorType,
        batch_D: torch.TensorType,
        model,
        tokenizer,
        criterion,
        device: torch.DeviceObjType,
        train: bool = True
    ):
    """
    batch_X: output from model f (victim model) with the sentence embeddings
    batch_D: the text that the attacker needs to generate, in order to have a perfect inversion
    """
    # padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize the text (gt) needed for a perfect inversion
    inputs = tokenizer(batch_D, return_tensors='pt', padding='max_length', truncation=True, max_length=40)
    #dial_tokens = [tokenizer.encode(item) + turn_ending for item in batch_D]
    #print(inputs)
    input_ids = inputs['input_ids'].to(device) # tensors of input ids
    labels = input_ids.clone()
    #print(input_ids.size())

    # embed the input ids of the gt text using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)
    
    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
    
    # The first token becomes the sentence embedding
    # Rest follows the gt text [Only for training]
    # So the labels are leading always 1 to the ground truth as seen in Figure 2 of the paper
    inputs_embeds = torch.cat((batch_X_unsqueeze, input_emb),dim=1) # [batch, 1 + max_length, emb_dim]
    # print(inputs_embeds.size())
    past = None
    # need to move to device later
    inputs_embeds = inputs_embeds

    #logits, past = model(inputs_embeds=inputs_embeds,past = past)
    logits, past = model(inputs_embeds=inputs_embeds, past_key_values=past, return_dict=False)
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()

    return record_loss, perplexity



if __name__ == '__main__':
    model_cards = {}
    model_cards['sent_t5_large'] = 'sentence-t5-large'
    model_cards['sent_t5_base'] = 'sentence-t5-base'
    model_cards['sent_t5_xl'] = 'sentence-t5-xl'
    model_cards['sent_t5_xxl'] = 'sentence-t5-xxl'
    model_cards['mpnet'] = 'all-mpnet-base-v1'
    model_cards['sent_roberta'] = 'all-roberta-large-v1'
    model_cards['simcse_bert'] = 'princeton-nlp/sup-simcse-bert-large-uncased'
    model_cards['simcse_roberta'] = 'princeton-nlp/sup-simcse-roberta-large'
    
    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--model_dir', type=str, default='microsoft/DialoGPT-large', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoches.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch_size #.')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: personachat or qnli')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--embed_model', type=str, default='sent_t5_base', help='Name of embedding model: mpnet/sent_roberta/simcse_bert/simcse_roberta/sent_t5')
    parser.add_argument('--decode', type=str, default='beam', help='Name of decoding methods: beam/sampling')
    args = parser.parse_args()
    
    config = {}
    config['model_dir'] = args.model_dir
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['dataset'] = args.dataset
    config['data_type'] = args.data_type
    config['embed_model'] = args.embed_model
    config['decode'] = args.decode
    config['embed_model_path'] = model_cards[config['embed_model']]

    # custom save paths
    attacker_model: str = "rand_gpt2_m" if args.model_dir == "random_gpt2_medium" else "dialogpt2"
    attacker_save_name: str = f"attacker_{attacker_model}_{config['dataset']}_{config['embed_model']}"
    projection_save_name: str = f"projection_{attacker_model}_{config['dataset']}_{config['embed_model']}"
    config['attacker_save_path'] = f"models/{attacker_save_name}"
    config['projection_save_path'] = f"models/{projection_save_name}"
    config['attacker_save_name'] = attacker_save_name
    config['projection_save_name'] = projection_save_name

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda")
    else:
        raise ImportError("Torch couldn't find CUDA devices. Can't run the attacker on CPU.")
    
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    config['eos_token'] = config['tokenizer'].eos_token
    config['use_opt'] = False

    device = config['device']
    batch_size = config['batch_size']

    sent_list = get_sent_list(config)
    
    # TODO: figure out when projection is necessary
    attacker_emb_size = 768
    if config['model_dir'].find("medium") != -1:
        attacker_emb_size = 1024
    elif config['model_dir'].find("large") != -1:
        attacker_emb_size = 1280


    if(config['data_type'] == 'train'):
        # -- Training --
        if('simcse' in config['embed_model']):
            process_data_simcse(sent_list,batch_size,device,config,attacker_emb_size)
        else:
            process_data(sent_list,batch_size,device,config,attacker_emb_size)

    elif(config['data_type'] == 'test'):
        # -- Inference --
        if('simcse' in config['embed_model']):
            process_data_test_simcse(sent_list,batch_size,device,config,attacker_emb_size=attacker_emb_size)
        else:
            process_data_test(sent_list,batch_size,device,config,attacker_emb_size=attacker_emb_size)

