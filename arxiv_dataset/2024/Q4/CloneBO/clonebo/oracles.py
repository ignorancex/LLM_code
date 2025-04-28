import transformers
import pandas as pd
import os
import json
import torch
import re
import random
import warnings

from transformers import ( 
    AutoConfig,
)

from .bytenet import(
    ByteNetForSequenceClassification, ByteNet
)
from .linear import(
    LinearForSequenceClassification,
)
from .modeling_llama import LlamaForCausalLM

from abnumber import Chain
from .data import AHO_HEAVY_LENGTH, AHO_LIGHT_LENGTH

def extract_cdr3(sequence: str) -> str:
    chain = Chain(sequence.replace("-",""), scheme='aho', cdr_definition='chothia')
    return chain.cdr3_seq

def align(seq):
    chain = Chain(seq.replace("-",""), scheme='aho', cdr_definition='chothia')
    chain_dict = {int(re.sub("[^0-9]", "", str(pos)[1:])): aa for pos, aa in chain}
    expect_length = AHO_HEAVY_LENGTH if chain.is_heavy_chain() else AHO_LIGHT_LENGTH
    aligned = [chain_dict.get(i, "-") for i in range(1, expect_length+1)]
    return "".join(aligned)


class RegressionOracle():
    
    def __init__(self, model, tokenizer, label_mean=0., label_std=1.) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_mean = label_mean
        self.label_std = label_std
        
    def predict_batch(self, sequences: list) -> list:
        sequences = [" ".join(sequence) for sequence in sequences]
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        with torch.no_grad():
            outputs = self.model(inputs)
        logits = outputs.logits * self.label_std + self.label_mean
        return logits
    
    def predict(self, sequence: str) -> float:
        return self.predict_batch([sequence])[0]
    

class BinaryClassificationOracle():

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        
    def predict_batch(self, sequences: list) -> list:
        sequences = [" ".join(sequence) for sequence in sequences]
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        with torch.no_grad():
            outputs = self.model(inputs)
        logits = outputs.logits
        logodds = logits[:, 1] - logits[:, 0]
        return logodds
        # prob_one = torch.nn.functional.softmax(logits, dim=1)[:,1]
        # return prob_one
    
    def predict(self, sequence: str) -> float:
        return self.predict_batch([sequence])[0]
    

class LLMOracle(RegressionOracle):
    def __init__(self, model=None, tokenizer=None, max_length=2048, ignore_index=-100) -> None:
        if tokenizer is None:
            tokenizer = transformers.BertTokenizerFast(
                vocab_file="oracle_data/vocab.txt",
                do_lower_case=False,
            )

            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.bos_token_id = tokenizer.cls_token_id
            tokenizer.eos_token = tokenizer.sep_token
            tokenizer.eos_token_id = tokenizer.sep_token_id

            tokenizer.hv_token = "[AbHC]"
            tokenizer.hv_token_id = tokenizer.convert_tokens_to_ids(tokenizer.hv_token)
            tokenizer.lv_token = "[AbLC]"
            tokenizer.lv_token_id = tokenizer.convert_tokens_to_ids(tokenizer.lv_token)
            tokenizer.seq_sep_token = "[ClSep]"
            tokenizer.seq_sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.seq_sep_token)

        if model is None:
            config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            config.vocab_size = len(tokenizer)
        
            gpt2_config = AutoConfig.from_pretrained("openai-community/gpt2-large")

            config.hidden_size = 512
            config.intermediate_size = 512 * 4
            config.num_hidden_layers = 12
            config.num_attention_heads = 4
            config.num_key_value_heads = 4
            config.max_position_embeddings = 2048
            config.update({"use_clone_attention":False})

            # LLaMA Oracle
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained("CloneBO/OracleLM")

        super().__init__(model, tokenizer, None, None)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # setup models
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.ignore_index = ignore_index


    def predict_batch(self, sequences: list) -> list:
        assert isinstance(sequences, list), "the input should be a list of string sequences"
        assert all(isinstance(sequence, str) for sequence in sequences), "the input should be a list of string sequences"
        
        tokens = [
            self.tokenizer(" ".join(seq), return_tensors="pt").input_ids[0][1:-1] for seq in sequences
        ]

        one_t = torch.ones(1, dtype=torch.long)

        tokens = list(map(lambda token: torch.cat([self.tokenizer.bos_token_id * one_t, token, self.tokenizer.seq_sep_token_id * one_t]), tokens))
        tokens = list(map(lambda token: token[:self.max_length], tokens))

        input_ids = labels = tokens

        # define loss
        loss_fn = torch.nn.CrossEntropyLoss()

        list_of_loss = []
        for i in range (len(input_ids)):
            curr_input_id = input_ids[i].clone().detach()
            curr_label = labels[i].clone().detach()

            with torch.no_grad():
                outputs = self.model(curr_input_id.to(self.device).unsqueeze(0), labels=curr_label.to(self.device))

            shift_logits = outputs.logits[...,:-1,:].contiguous()
            shift_labels = curr_label[...,1:].contiguous()
        
            loss = loss_fn(shift_logits.squeeze(), shift_labels.to(self.device))
            list_of_loss.append(loss.item())
        
        return list_of_loss
    
    def predict(self, sequence: str) -> float:
        return self.predict_batch([sequence])





class RandInitNeuralNetOracle(RegressionOracle):
    def __init__(self, model=None, tokenizer=None, max_length=512, ignore_index=-100, alpha=0.5) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha

        # LLM Oracle
        self.llm_oracle = LLMOracle()

        # Random Oracle
        self.rand_oracle_tokenizer = self.llm_oracle.tokenizer

        rand_bytenet_ckpt_path = "oracle_data/oracle_weights/rand_bytenet_model_ckpt.pth"
        if os.path.exists(rand_bytenet_ckpt_path):
            self.rand_oracle = ByteNet(n_tokens=self.rand_oracle_tokenizer.vocab_size, d_embedding=32, d_model=64, n_layers=3, kernel_size=3, r=1)
            self.rand_oracle.load_state_dict(torch.load(rand_bytenet_ckpt_path, map_location=torch.device('cpu')))
        else:
            self.rand_oracle = ByteNet(n_tokens=self.rand_oracle_tokenizer.vocab_size, d_embedding=32, d_model=64, n_layers=3, kernel_size=3, r=1)
            torch.save(self.rand_oracle.state_dict(), rand_bytenet_ckpt_path)

        # mean and std statistics
        llm_oracle_obj_loss_path = 'oracle_data/oracle_weights/llm_oracle_loss_stats.pth'
        rand_oracle_obj_loss_path = 'oracle_data/oracle_weights/rand_oracle_loss_stats.pth'
        
        self.llm_oracle_obj_loss = torch.load(llm_oracle_obj_loss_path)
        self.rand_oracle_obj_loss = torch.load(rand_oracle_obj_loss_path)

        self.llm_oracle_obj_loss_mean = torch.mean(self.llm_oracle_obj_loss)
        self.rand_oracle_obj_loss_mean = torch.mean(self.rand_oracle_obj_loss)

        self.llm_oracle_obj_loss_std = torch.std(self.llm_oracle_obj_loss)
        self.rand_oracle_obj_loss_std = torch.std(self.rand_oracle_obj_loss)

        super().__init__(model, tokenizer, None, None)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # setup models
        self.rand_oracle.to(self.device)
        self.rand_oracle.eval()

        self.max_length = max_length
        self.ignore_index = ignore_index

    def pad_tokens(self, token):
        num_25_to_add = self.max_length - len(token)
        token = torch.cat([token, torch.tensor([25] * num_25_to_add)])
        return token

    def predict_batch(self, unaligned_sequences: list) -> list:
        # 1. get LLM score
        if self.alpha == 1:
            pass
        else:
            llm_score = self.llm_oracle.predict_batch(unaligned_sequences)
        
        aligned_sequences = [align(sequence) for sequence in unaligned_sequences] 

        tokens = [
            self.rand_oracle_tokenizer(" ".join(seq), return_tensors="pt").input_ids[0][1:-1] for seq in aligned_sequences
        ]

        tokens = [self.pad_tokens(token) for token in tokens] 

        one_t = torch.ones(1, dtype=torch.long)

        tokens = list(map(lambda token: torch.cat([self.rand_oracle_tokenizer.bos_token_id * one_t, token, self.rand_oracle_tokenizer.seq_sep_token_id * one_t]), tokens))
        tokens = list(map(lambda token: token[:self.max_length], tokens))

        input_ids = labels = tokens

        # define loss
        loss_fn = torch.nn.CrossEntropyLoss()

        rand_score = []
        for i in range(len(input_ids)):
            curr_input_id = input_ids[i].clone().detach()
            curr_label = labels[i].clone().detach()

            with torch.no_grad():
                outputs = self.rand_oracle(curr_input_id.to(self.device).unsqueeze(0)) #, labels=curr_label.to(self.device))
            
            shift_logits = outputs[...,:-1,:].contiguous()
            shift_labels = curr_label[...,1:].contiguous()
        
            loss = loss_fn(shift_logits.squeeze(), shift_labels.to(self.device))
            rand_score.append(loss.item())
        
        # total scores
        if self.alpha == 1:
            total_score = torch.tensor(rand_score)
        else:
            total_score = self.alpha * (torch.tensor(rand_score) - self.rand_oracle_obj_loss_mean) / self.rand_oracle_obj_loss_std  + (1-self.alpha) * (torch.tensor(llm_score) - self.llm_oracle_obj_loss_mean) / self.llm_oracle_obj_loss_std
            

        return total_score

    def predict(self, sequence: str) -> float:
        return self.predict_batch([sequence]) 
