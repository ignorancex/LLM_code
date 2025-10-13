from utils.constants import TEMPLATE_EVAL, TEMPLATE_EVAL_RES, INST_CODELLAMA_EVAL, INST_CODELLAMA_EVAL_RES, INST_CODELLAMA_EVAL_DB

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,  GenerationConfig

import numpy as np
import torch
import sqlite3
import random
from collections import Counter

from utils.functions import find_last_occurrence


class OracleEvaluator():

    def __init__(self, db_path, oracle_prob):
        self.db_path = db_path
        self.oracle_prob = oracle_prob


    def score(self, db_id, question, candidates, gold_sql):
        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        try:
            cursor.execute(gold_sql)
        except:
            return [1 for cand_sql in candidates]

        gold_res = []
        for i in range(5):
            row = cursor.fetchone()
            if row is None:
                break
            gold_res.append(Counter(row))

        scores = []
        for cand_sql in candidates:
            try:
                cursor.execute(cand_sql)
            except:
                if random.random() < self.oracle_prob:
                    scores.append(1)
                else:
                    scores.append(-1)
                continue

            cand_res = []
            for i in range(5):
                row = cursor.fetchone()
                if row is None:
                    break
                cand_res.append(Counter(row))

            if len(gold_res) == 0:
                if len(cand_res) == 0:
                    if random.random() < self.oracle_prob:
                        scores.append(-1)
                    else:
                        scores.append(0)
                else:
                    if random.random() < self.oracle_prob:
                        scores.append(0)
                    else:
                        scores.append(-1)
            else:
                overlap = 0
                for i, row in enumerate(cand_res):
                    if i >= len(gold_res):
                        break
                    
                    gold_row = gold_res[i]
                    for col in row:
                        if col in gold_row:
                            overlap += row[col]
                
                gold_sum = sum([sum([v for v in gold_row.values()]) for gold_row in gold_res])
                cand_sum = sum([sum([v for v in cand_row.values()]) for cand_row in cand_res])

                prec = overlap / gold_sum
                rec = (overlap / cand_sum) if cand_sum > 0 else 0

                if random.random() < self.oracle_prob:
                    scores.append(
                        - (2 * prec * rec / (prec + rec + 1e-8))
                    )
                else:
                    scores.append(
                        (2 * prec * rec / (prec + rec + 1e-8)) - 1
                    )

        return scores

class LLMEvaluator():
    # yes_token_indx: 
    # The index of the token in the vocabulary that corresponds to the "Yes" text.
    # CodeLlama-Instruct: "No" 1939 "Yes" 3869
    # TinyLlama: "Yes" 3869

    def __init__(self, model_name_or_dir, db_path, device="cuda",yes_token_indx=None):
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_dir,
                use_fast=False
            )
        except:
            # load tokenizer without use_fast=False
            # for stable-code-3b model
            self.tokenizer = AutoTokenizer.from_pretrained( model_name_or_dir )
            Warning("Tokenizer is not loaded with use_fast=False")

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)
        self.model.eval()

        # set other params
        self.db_path = db_path
        self.device = device
        if yes_token_indx:
            self.yes_token_indx = yes_token_indx
            self.validate_yes_token()
        else:
            self.yes_token_indx = self.get_yes_token()

    def get_yes_token(self):
        return int( self.tokenizer.encode("Yes")[-1])
    
    def validate_yes_token(self):
        assert self.get_yes_token() == self.yes_token_indx

    def score(self, db_id, question, candidates, evaluation_config):
        # load db
        db_path=f'{self.db_path}/{db_id}/{db_id}.sqlite'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
                # create prompt
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                # create prompt
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        TEMPLATE_EVAL.format(question, cand_sql)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            # move to device    
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                # negative of softmax as score. so, lower is better
                scores.append(
                    - torch.nn.functional.softmax(
                        outputs["logits"][:, -1, :], dim=-1
                    ).flatten()[self.yes_token_indx].item()
                )

        return scores


    def score_fewshot(self, db_id, question, candidates, retriever, evaluation_config):
        demos = retriever.retrieve(question)

        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
                
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["sql"], d["exec_res"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["neg_sql"], d["neg_exec_res"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                )
                
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        "\n\n".join(prompt_strs)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["sql"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["neg_sql"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL.format(question, cand_sql)
                )

                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        "\n\n".join(prompt_strs)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            
                scores.append(
                    - torch.nn.functional.softmax(
                        outputs["logits"][:, -1, :], dim=-1
                    ).flatten()[self.yes_token_indx].item()
                )

        return scores


class LLMLoraEvaluator(LLMEvaluator):

    def __init__(self, model_name_or_dir, peft_model_dir, db_path, device="cuda",yes_token_indx=3869):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False
        )
        # load model
        # first, set up the quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  
            llm_int8_threshold=6.0, 
            llm_int8_enable_fp32_cpu_offload=True  
        )
        # then, load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)
        
        # load the PEFT model
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_dir
        )
        self.model.eval()

        # set other params
        self.db_path = db_path
        self.device = device
        if yes_token_indx:
            self.yes_token_indx = yes_token_indx
            self.validate_yes_token()
        else:
            self.yes_token_indx = self.get_yes_token()


# reasoning evaluators
from modules.llm_generator import LLMGenerator
import re
import json
from utils.prompts import prepare_eval_prompt, prepare_eval_res_prompt, prepare_eval_promptV2
from utils.functions import swap_memory

class LLMReasoningEvaluator(LLMGenerator):
    
    def __init__(self, base_model_name, db_path, device, search_format = r"\*\*(.*?)\*\*", pos = "Yes", neg = "No", failvalue = 0.5, max_new_tokens = 300, json_key = "correct", use_logits = True):
        super().__init__(base_model_name, device)  # Calls the Parent class __init__

        # set other params
        self.db_path = db_path
        self.search_format = search_format
        self.pos = pos
        self.neg = neg
        self.failval = failvalue
        self.max_new_tokens = max_new_tokens
        self.json_key = json_key
        self.use_logits = use_logits

    def generate(self, prompt, generation_config):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,  # Same as "longest"
            truncation=True,   
            add_special_tokens=False
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Making sure model is in the save device
        if self.model.device != self.device:
            self.model.to(self.device)
            Warning('Moved model to the same device.')

        with torch.no_grad():
            completion = self.model.generate(
                inputs=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=GenerationConfig(**generation_config)
            )

        return [self.tokenizer.decode(c, skip_special_tokens=True) for c in completion]

    def generate_with_logits(self, prompt, generation_config):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,  # Same as "longest"
            truncation=True,   
            add_special_tokens=False
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Making sure model is in the save device
        if self.model.device != self.device:
            self.model.to(self.device)
            Warning('Moved model to the same device.')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=GenerationConfig(**generation_config),
                return_dict_in_generate=True,  # Ensures logits are returned
                output_scores=True,  # This will give you logits at each generation step
                output_hidden_states=False  # You can set this to True if you want hidden states too
            )

        # Extract the generated token IDs (completion['sequences'])
        generated_ids = outputs.sequences # this has input ids as well
        input_id_length = batch['input_ids'].shape[-1] # find out input id length
        generated_new_ids = generated_ids[:,input_id_length:] # only keep the new generated ids

        # print(f" gen ids: {generated_ids.shape}")
        # print(f"input ids: {batch['input_ids'].shape}")
        # print(f" new gen ids: {generated_new_ids.shape}")
        
        # Decode the generated token IDs into text
        decoded_text = [self.tokenizer.decode(g, skip_special_tokens=False) for g in generated_ids]

        # Extract the logits from the output_scores tuple (for each generation step)
        logits = outputs.scores # tuple length is total generated tokens. each element: (batch x vocab)
        # print(f"logits: {len(logits)}")
        
        # Returning both the decoded text and the tensor of logits for the last step
        return decoded_text, logits, generated_new_ids

    def extract_correct_valueV2(self, s):
        """
        Extracts the boolean value from the 'correct' key in the last valid JSON object found in the input string.
        Also returns the index [start, stop] of this JSON object in the input string.

        Parameters:
        s (str): Input string containing text with embedded JSON objects.

        Returns:
        tuple: (bool or None, list) - The last extracted boolean value of 'correct', or None if not found,
            along with the [start, stop] index of the JSON object.
        """
        matches = list(re.finditer(r'\{.*?\}', s, re.DOTALL))  # Capture all JSON blocks
        final_ans = None  # Store the last valid 'correct' value
        final_range = [0, 0]  # Store the index range of the last valid JSON object
        for match in matches:
            json_str = match.group()
            start, stop = match.start(), match.end() # index of the JSON payload

            # Fix trailing commas before parsing
            json_str = re.sub(r',\s*\}', '}', json_str)

            # Convert Python-style booleans to JSON-style
            json_str = json_str.replace("False", "false").replace("True", "true")

            try:
                data = json.loads(json_str)  # Convert JSON string to dictionary
                if self.json_key in data:
                    value = data[self.json_key]
                    if isinstance(value, bool):
                        final_ans = value  # Use directly if it's a boolean
                    elif isinstance(value, str):
                        lower_val = value.lower()
                        if lower_val == "true":
                            final_ans = True
                        elif lower_val == "false":
                            final_ans = False
                        else:
                            final_ans = None  # Invalid value, set to None
                    final_range = [start, stop]  # Update final range
            except json.JSONDecodeError:
                continue  # Skip invalid JSON and proceed to next match
        
        # final json block
        final_json_block = s[final_range[0]:final_range[1]]
        final_json_block_lower = final_json_block.lower()

        # get the index of the value of key in the json block
        target_words = ['true', 'false']
        word_indx = [0,0]
        for word in target_words:
            match = re.search(rf'\b{re.escape(word)}\b', final_json_block_lower)
            if match:
                word_indx = [final_range[0] + match.start(), final_range[0] + match.end()]
                break
        
        final_word = s[word_indx[0]:word_indx[1]] # this is the word
        # print(final_word) 
        return final_ans, word_indx, final_word  # Return the last found 'correct' value and its range

    def find_word_token_index(self,generated_new_ids):
        """
        Find the index of the token where the word resides
        """
        s = [self.tokenizer.decode(g, skip_special_tokens=False) for g in generated_new_ids]

        # print(s)
        # Tokenize the string (without recomputing logits)
        inputs = self.tokenizer(s, return_tensors="pt", 
                                    padding=True,  # Same as "longest",
                                    add_special_tokens=True,
                                    return_offsets_mapping=True)

        # print(inputs["offset_mapping"].shape)
        offset_mappings = inputs["offset_mapping"]  # Token position mapping. batch x number of tokens x 2 (start, stop index)
        
        # MANUAL OFFSET: somehow first token is just empty
        offset_mappings = offset_mappings[:,1:,:]
        # codes below shows why we are offsetting by 1 
        # tt = inputs['input_ids']
        # tt = tt[:,1:] # compare this to `generated_new_ids`
        
        # print(f"offset: {offset_mappings[0].shape}")
        word_token_indx = []
        final_ans_list = []
        final_word_list = []

        for i in range(len(generated_new_ids)):
            offset_mapping = offset_mappings[i]
            # get the value's index in the answer string s
            final_ans, word_indx, final_word = self.extract_correct_valueV2(s[i])
            final_ans_list.append(final_ans)
            # final_word_list.append(final_word)

            # print(word_indx)
            # print(final_word)

            # find the index of the value's token

            #1. Assuming offset_mapping is a list or tensor of shape (num_tokens, 2)
            offset_mapping = offset_mapping.float()  # Ensure it's a float tensor

            #2. Convert target to float as well
            target = torch.tensor(word_indx, dtype=torch.float32)
            # print(target.shape)

            #3. Compute squared Euclidean distances row-wise
            distances = torch.norm(offset_mapping - target, dim=1)  # No need to wrap in another tensor

            #4. Find the index of the minimum distance
            min_index = torch.argmin(distances)
            min_index = min_index.detach().tolist()
            word_token_indx.append(min_index)
            # print(min_index)
            
            #5. get the tokenized portion of the final word
            final_word_index = offset_mapping[min_index,:].tolist()
            final_word = s[i][int(final_word_index[0]): int(final_word_index[1])]
            final_word_list.append(final_word)
        
        return final_ans_list, word_token_indx, final_word_list

    def isCorrectV3(self, generated_new_ids, logits):
        """
        Use the token's softmax as score.
        for true: prob
        for false: 1-prob
        else: self.failVal
        """
        # get value's token index
        final_ans_list, word_token_indx, final_word_list = self.find_word_token_index(generated_new_ids)
        # print(word_token_indx)

        # collect logits
        final_word_logits = [] # batch x (1 x vocab)
        for i in range(len(word_token_indx)):
            token_index = word_token_indx[i]
            word_logit = logits[token_index][i]
            final_word_logits.append(word_logit)

        scores = []
        for i in range(len(final_word_logits)):
            final_word = final_word_list[i]
            word_logit = final_word_logits[i]
            # print(final_word)
            if len(final_word)>0:
                # vocab id for the word
                vocab_indx = int( self.tokenizer.encode(final_word, add_special_tokens=False)[0])
                # print(vocab_indx)
                
                # get the probability
                word_probs = torch.nn.functional.softmax(word_logit, dim=-1)
                prob = word_probs[vocab_indx].item()
                
                # modify probability based on conditions
                if 'true' in final_word.lower():
                    prob = prob
                elif 'false' in final_word.lower():
                    prob = 1 - prob
                else:
                    prob = self.failval
            else:
                prob = self.failval
            # add as score
            scores.append(prob)
        # print(scores)
        return scores

    # scores each response 0 or 1
    def isCorrect(self, response):
        divider = "</think>"
        # divider does not exist
        if not re.search(divider,response):
            divider = "<think>"

        # find answer after divider tag
        ans = response.split(divider)[-1]

        # find Yes/ No from the end
        final_ans, _ = find_last_occurrence(ans, [self.pos, self.neg] )
        
        # scoring system
        if not final_ans: # Yes/ No not found. The desciminator might have failed
            return 0.0
        elif final_ans == self.pos:
            return 1.0
        elif final_ans == self.neg:
            return 0.0
        return 0.0
    
    """
    Search for JSON with key 'correct'.
    1. no JSON = self.failval
    2. correct: True = 1
    3. correct: False = 0
    4. else = self.failval
    """
    def isCorrectV2(self, response):
        final_ans, _ , _ = self.extract_correct_valueV2(response)       
        # scoring system
        if final_ans is None: # Yes/ No not found. The desciminator might have failed
            return self.failval
        elif final_ans is True: # correct
            return 1.0
        elif final_ans is False: # incorrect
            return 0.0
        else:
            return self.failval


    def extract_correct_value(self, s):
        """
        Extracts the boolean value from the 'correct' key in the last valid JSON object found in the input string.

        The function:
        - Finds all JSON-like objects in the input string using regex.
        - Fixes trailing commas to ensure valid JSON syntax.
        - Replaces Python-style booleans ('True', 'False') with JSON-compliant ('true', 'false').
        - Parses each JSON object and updates the value of 'correct' if found.
        - Correctly handles both boolean and string representations of 'true' and 'false'.
        - Returns None if 'correct' contains an invalid value.
        
        Parameters:
        s (str): Input string containing text with embedded JSON objects.

        Returns:
        bool or None: The last extracted boolean value of 'correct', or None if no valid JSON object is found.
        """

        matches = re.findall(r'\{.*?\}', s, re.DOTALL)  # Capture all JSON blocks
        final_ans = None  # Store the last valid 'correct' value

        for match in matches:
            json_str = match

            # Fix trailing commas before parsing
            json_str = re.sub(r',\s*\}', '}', json_str)  

            # Convert Python-style booleans to JSON-style
            json_str = json_str.replace("False", "false").replace("True", "true")  

            try:
                data = json.loads(json_str)  # Convert JSON string to dictionary
                if self.json_key in data:
                    value = data[self.json_key]
                    if isinstance(value, bool):
                        final_ans = value  # Use directly if it's a boolean
                    elif isinstance(value, str):
                        lower_val = value.lower()
                        if lower_val == "true":
                            final_ans = True
                        elif lower_val == "false":
                            final_ans = False
                        else:
                            final_ans = None  # Invalid value, set to None
            except json.JSONDecodeError:
                continue  # Skip invalid JSON and proceed to next match

        return final_ans  # Return the last found 'correct' value

    # if the case is sql runnable. example has two keys: 'db_id' and 'sql'
    def isRunnable(self, example):
        # load db
        db=f"{self.db_path}/{example['db_id']}/{example['db_id']}.sqlite"
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        try:
            cursor.execute(example['sql'])
            score = 1.0
        except:
            score = 0.0
        return score
    
    # returns query results. example has two keys: 'db_id' and 'sql'
    def findQueryResult(self, example):
        # load db
        db=f"{self.db_path}/{example['db_id']}/{example['db_id']}.sqlite"
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(example['sql'])
            # Fetch column names and initialize result storage
            columns = [col[0].lower() for col in cursor.description]
            rows = [cursor.fetchone() for _ in range(5)] # up to 5 rows
            rows = [row for row in rows if row]  # Remove None values

            if not rows:
                return "Error"
            
            # Format output
            formatted_rows = [
                f"{col}: {', '.join(str(v)[:128] + '...' if isinstance(v, str) and len(v) > 128 else str(v) for v in col_values)}"
                for col, col_values in zip(columns, zip(*rows))
            ]
            return "-- " + "\n-- ".join(formatted_rows)
        except:
            return "Error"

    # score for the candidates
    """
    You can check validitity of sql via running and then check correctness.
    Currently, evaluator can't handle results but the system can get the query results.
    1    : invalid sql
    0    : valid sql but incorrect
    -0.5 : valid sql but evaluator failed
    -1   : valid sql and correct
    """
    def score(self, example, sql_completions, evaluation_config, retriever_eval=None, device_swap=False,evaluator_config=None, useSchema=False):
        """
        example is a hashmap with keys: ['db_id', 'schema', 'question', 'sql']
        evaluator is a LLM model
        candidates is list of sql queries
        evaluation_config is a hashmap for running evaluator
        example config:
            config ={
            "_from_model_config": True,
            "do_sample": False,
            "num_return_sequences": 1
        }
        """
        # set config for evaluation generation
        if not evaluator_config:
            evaluator_config ={
            "_from_model_config": True,
            "do_sample": False,
            "num_return_sequences": 1
            }
        
        candidates = []
        # candidates: make copies of example with sql completions in 'sql' key
        for s in sql_completions:
            temp = example.copy()
            temp['sql'] = s
            candidates.append(temp)
        
        final_scores = np.array([1.0 for s in candidates]) # by default all candidates are invalid
        
        # mask candidates based on runnablility    
        if evaluation_config["check_exec"]:    
            mask = [ self.isRunnable(c) > 0 for c in candidates ]
        else:
            mask = [ True for c in candidates ]
    
        if not any(mask):
            evaluation_log = "all solutions are invalid"
            #print(evaluation_log)
        else:
            if evaluation_config["use_exec_res"]:
                # get query results and add new key 'results' in new_candidates
                for j in range(len(candidates)):
                    candidates[j]['result'] = self.findQueryResult(candidates[j])
            
            # only use the valid candidates
            candidates = [c for c, m in zip(candidates, mask) if m ]

            
            # c_lengths = np.array([len(s['sql']) for s in candidates])
            # c_lengths_invnorm = (np.max(c_lengths) - c_lengths) / np.max(c_lengths) # normalize. longest has value 0 shorest has 1

            # create list of prompts
            if evaluation_config["use_exec_res"]:
                prompts = [prepare_eval_res_prompt(ex) for ex in candidates]
            else:
                # prompts = [prepare_eval_prompt(ex) for ex in candidates]
                prompts = [prepare_eval_promptV2(ex, useSchema) for ex in candidates]

            if device_swap: # move model to GPU
                swap_memory(self.model, device="cuda", verbose=False)
            
            # generate evaluator response
            # responses = self.generate(prompts, evaluator_config)
            
            # generate evaluator response with logits
            responses, logits, generated_new_ids = self.generate_with_logits( prompts, evaluator_config)
            
            if device_swap: # move model to GPU
                swap_memory(self.model, device="cpu", verbose=False)
            
            # calculate scores. ranges from 0 to 1. 1 means good.
            # corr_scores = np.array([self.isCorrect(r) for r in responses])
            if self.use_logits:
                corr_scores = np.array( self.isCorrectV3(generated_new_ids, logits) )
            else:
                corr_scores = np.array([self.isCorrectV2(r) for r in responses])

            # invert the correctness and get the score
            scores = -1 * corr_scores 
            #scores = -1 * corr_scores * c_lengths_invnorm

            # find what % of cases evaluator failed to determine
            fail = np.sum(scores == -self.failval)
            fail = fail / len(candidates) * 100

            # for debug
            # print(f"Mask: {mask}\nAccepted: {np.mean(np.array(mask)) * 100} % \nLengths: {c_lengths}\nCorrct Scores: {corr_scores}\nEval fail rate: {fail} % \nScores: {scores}")
            
            # log
            response_log = "\n=======\n".join(responses)
            evaluation_log = f"Mask: {mask}\nAccepted: {np.mean(np.array(mask)) * 100} % \nCorrct Scores: {corr_scores}\nEval fail rate: {fail} % \nScores: {scores}\nResponses:\n{response_log}"
        
            # apply mask to make the final scores
            final_scores[mask] = scores # add scores for valid candidates

        return final_scores.tolist(), evaluation_log