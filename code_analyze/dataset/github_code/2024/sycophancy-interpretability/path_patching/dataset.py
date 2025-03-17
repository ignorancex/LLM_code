#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 14:50
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 17:31
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import json

from transformers import AutoTokenizer


class PathPatchingDataset:
    """
    Dataset for path patching
    """

    def __init__(self, filename: str, tokenizer: AutoTokenizer) -> None:
        """Initialize dataset

        Parameters
        ----------
        filename : str
            Name of the .jsonl file.
        tokenizer : AutoTokenizer
            Tokenizer for tokenization.
        """

        assert filename.endswith(".jsonl"), "Dataset file must be a .jsonl file"
        
        self.filename           = filename
        self.all_data           = None
    
        self.xr                 = []
        self.xc                 = []
        self.xr_toks            = []
        self.xc_toks            = []
    
        self.record_tokens      = []
        self.record_token_ids   = []
        self.predict_token      = []
        self.predict_token_id   = []
        
        self.length             = []
        self.data_id            = []
        
        # initialize
        self.setup(tokenizer=tokenizer)

    def load(self):
        with open(self.filename, "r") as f:
            self.all_data = [json.loads(line) for line in f.readlines()]
        self.length = len(self.all_data)
    
    def setup(self, tokenizer: AutoTokenizer):
        """Set up dataset

        Parameters
        ----------
        tokenizer : AutoTokenizer
            Tokenizer for tokenization.
        """

        self.load()
        
        assert all(["reference_data"      in item for item in self.all_data]), "Each data of dataset must contain reference_data"
        assert all(["counterfactual_data" in item for item in self.all_data]), "Each data of dataset must contain counterfactual_data"
        
        for item in self.all_data:
            if isinstance(item["reference_data"], str) and isinstance(item["counterfactual_data"], str):
                reference_data = item["reference_data"]
                counterfactual_data = item["counterfactual_data"]
                add_special_tokens = True
            else:
                add_generation_prompt = item['reference_data'][-1]['role'] == 'user'
                reference_data = tokenizer.apply_chat_template(item["reference_data"], tokenize=False, add_generation_prompt=add_generation_prompt)
                counterfactual_data = tokenizer.apply_chat_template(item["counterfactual_data"], tokenize=False, add_generation_prompt=add_generation_prompt)
                add_special_tokens = False
            
            self.xr.append(reference_data)
            self.xc.append(counterfactual_data)
            
            self.xr_toks.append(tokenizer.encode(reference_data, add_special_tokens=add_special_tokens))
            self.xc_toks.append(tokenizer.encode(counterfactual_data, add_special_tokens=add_special_tokens))
            
            self.predict_token.append(item["predict_token"])
            self.predict_token_id.append(tokenizer.encode(item["predict_token"], add_special_tokens=False)[0])
            
            self.record_tokens.append(item["record_tokens"])
            self.record_token_ids.append([tokenizer.encode(token, add_special_tokens=False)[0] for token in item["record_tokens"]])
            
            self.data_id.append(item["id"])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'xr': self.xr[idx],
            'xc': self.xc[idx],
            'xr_toks': self.xr_toks[idx],
            'xc_toks': self.xc_toks[idx],
            'predict_token': self.predict_token[idx],
            'predict_token_id': self.predict_token_id[idx],
            'record_tokens': self.record_tokens[idx],
            'record_token_ids': self.record_token_ids[idx],
        }
