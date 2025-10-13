import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, Qwen2Config
import collections
import argparse
import os
import numpy as np
from datasets import load_dataset
import ast
import random
import transformers
from datasets import set_caching_enabled
from src.data_wrapper import main_wrapper
import os
import statistics
import psutil
import copy
from src.ordering_strategy import GlobalOrderingRunner

set_caching_enabled(False)

class EvalPipeline():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        if not self.args.onlyeval and not self.args.routing:
            self.route_method()
        self.DataRouter = main_wrapper.MainDataWrapper(self.args)

    # assigin self.model, self.tokenizer here
    def route_method(self):
        if self.args.method == 'pcw':
            from PCW.model_loaders import load_pcw_wrapper
            self.model = load_pcw_wrapper(self.args.model_name, n_windows=self.args.pcw_window_k)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        else:
            if self.args.method == 'orig':
                self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, torch_dtype=torch.bfloat16, device_map='auto')
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
                self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            elif self.args.method in ['pine', 'ours']:
                if 'llama' in self.args.model_name.lower():
                    from PINE.pine import LlamaForCausalLMWithPS, LlamaTokenizerFastWithPS, LlamaForCausalLMWithPSRoToR
                    self.tokenizer = LlamaTokenizerFastWithPS.from_pretrained(self.args.model_name)
                    config = LlamaConfig.from_pretrained(self.args.model_name)
                    config._attn_implementation = 'eager'
                    if self.args.method == 'pine':
                        self.model = LlamaForCausalLMWithPS.from_pretrained(self.args.model_name, torch_dtype=torch.bfloat16, config=config, device_map='auto')
                        self.sorter = None
                    elif self.args.method == 'ours':
                        config.sorting_method = self.args.sorting_method
                        self.sorter = GlobalOrderingRunner(self.args)
                        self.model = LlamaForCausalLMWithPSRoToR.from_pretrained(self.args.model_name, torch_dtype=torch.bfloat16, config=config, device_map='auto')
                elif 'qwen' in self.args.model_name.lower():
                    from PINE.pine import Qwen2TokenizerWithPS, Qwen2ForCausalLMWithPS, Qwen2ForCausalLMWithPSRoToR
                    self.tokenizer = Qwen2TokenizerWithPS.from_pretrained(self.args.model_name)
                    config = Qwen2Config.from_pretrained(self.args.model_name)
                    config._attn_implementation = 'eager'
                    if self.args.method == 'pine':
                        self.model = Qwen2ForCausalLMWithPS.from_pretrained(self.args.model_name, torch_dtype=torch.bfloat16, config=config, device_map='auto')
                        self.sorter = None
                    elif self.args.method == 'ours':
                        config.sorting_method = self.args.sorting_method
                        self.sorter = GlobalOrderingRunner(self.args)
                        self.model = Qwen2ForCausalLMWithPSRoToR.from_pretrained(self.args.model_name, torch_dtype=torch.bfloat16, config=config, device_map='auto')
                self.model.generation_config.do_sample = False
                self.model.generation_config.max_new_tokens=500
                self.model.generation_config.pad_token_id=self.tokenizer.eos_token_id
            else:
                print(f"Routing method for {self.args.method} is not implemented.")
                raise NotImplementedError
        self.terminators = [self.tokenizer.eos_token_id]
        if self.args.measure_flops:
            if self.args.sorting_method == 'monot5':
                self.prof = FlopsProfiler(self.sorter.model)
            else:
                self.prof = FlopsProfiler(self.model)
            self.prof.start_profile()

    def build_inputs(self, prefix, parallel_text, suffix):
        if self.args.method in ['pine', 'ours']:
            input_string = [prefix, parallel_text, suffix]
            inputs = self.tokenizer(input_string)
            inputs = {k: torch.tensor(v).to('cuda') for k,v in inputs.items()}
            if self.sorter is not None:
                inputs['global_ordering'] = self.sorter.run_global_sort(prefix, parallel_text, suffix, inputs)
            else:
                inputs['global_ordering'] = []
        else:
            inputs = self.tokenizer(prefix + ''.join(parallel_text) + suffix, return_tensors='pt').to(self.device)
        return inputs

    def run_generation(self, input_data):
        if self.args.method == 'pcw':
            self.model.model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            prefix = input_data['prefix']
            parallel_text = input_data['parallel']
            suffix = input_data['suffix']
            if self.args.method == 'pcw':
                if self.args.inference_type == 'log_likelihood': # only for MMLU
                    candidate_token_ids = [self.tokenizer.encode(x, add_special_tokens=False) for x in candidates] # 362 425 356 422 for Qwen, 362 426 356 423 for Llama
                    labels_input_ids = np.array(candidate_token_ids)
                    from PCW.logits_processor import RestrictiveTokensLogitsProcessor
                    logit_processor = RestrictiveTokensLogitsProcessor(labels_input_ids,
                            eos_token_id=self.tokenizer.eos_token_id)
                else:
                    logit_processor = None
                if self.args.do_sample:
                    outputs = self.model.pcw_generate(contexts = [prefix + x for x in parallel_text],
                            restrictive_logit_preprocessor=logit_processor,
                            task_text=suffix, max_new_tokens=100, do_sample=True, temperature=self.args.temperature, top_p=0.9)
                else:
                    outputs = self.model.pcw_generate(contexts = [prefix + x for x in parallel_text],
                            restrictive_logit_preprocessor=logit_processor,
                            task_text=suffix, max_new_tokens=100, do_sample=False)
                return outputs.strip(), 0
            else:
                inputs = self.build_inputs(prefix, parallel_text, suffix)
            if self.args.debug:
                outputs = self.model.generate(**inputs, max_new_tokens=100, eos_token_id=self.terminators, do_sample=False) # 0.6
                response = outputs[0][inputs['input_ids'].shape[-1]:]
                pred = self.tokenizer.decode(response, skip_special_tokens=not self.args.keep_special_tokens)
            else:
                if 'ed' in self.args.method:
                    inputs['normalize'] = self.args.normalize
                if self.args.inference_type == 'generation':
                    if self.args.do_sample:
                        outputs = self.model.generate(**inputs, max_new_tokens=100, eos_token_id=self.terminators, do_sample=True, temperature=self.args.temperature, top_p=0.9, output_scores=True, return_dict_in_generate=True) # 0.6
                    else:
                        outputs = self.model.generate(**inputs, max_new_tokens=100, eos_token_id=self.terminators, do_sample=False, output_scores=True, return_dict_in_generate=True) # 0.6
                    response = outputs['sequences'][0][inputs['input_ids'].shape[-1]:]
                    pred = self.tokenizer.decode(response, skip_special_tokens=not self.args.keep_special_tokens)
                    scores = outputs['scores']  # [2:-1] to remove the prefix and suffix
                    top1_confidences = torch.stack([torch.softmax(t, dim=-1).max(dim=-1).values for t in scores]).squeeze(-1).tolist()
                elif self.args.inference_type == 'log_likelihood':
                    prompt_outputs = self.model(**inputs)
                    prompt_logits = prompt_outputs.logits
                    last_token_logits = prompt_logits[0,-1,:]
                    candidates = input_data['candidates'] # [' A', ' B', ' C', ' D']
                    candidate_token_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in candidates] # 362 425 356 422 for Qwen, 362 426 356 423 for Llama
                    candidate_scores = []
                    prefix_len = inputs['input_ids'].shape[1]
                    candidate_input = self.build_inputs(prefix, parallel_text, suffix) # currently it's same as original inputs
                    outputs = self.model(**candidate_input)
                    logits = outputs.logits
                    abcd_probs = logits[0, prefix_len-1, candidate_token_ids] # corresponds to abcd
                    abcd_probs = torch.nn.functional.softmax(abcd_probs, dim=-1)
                    abcd_max_idx = torch.argmax(abcd_probs)
                    pred = ['A','B','C','D'][abcd_max_idx.item()]
                    top1_confidences = abcd_probs[abcd_max_idx.item()].item()
        return pred, top1_confidences

    def run_selective_routing(self):
        orig_output_dir = self.args.output_dir.replace(f'/{self.args.method}/', '/orig/')
        ours_output_dir = self.args.output_dir
        orig_output_dir = orig_output_dir.replace(f'/sorting-method-{self.args.sorting_method}','')
        with open(f"{orig_output_dir}/result.json", 'r') as f:
            orig_data = json.load(f)
        with open(f"{ours_output_dir}/result.json", 'r') as f:
            ours_data = json.load(f)
        if len(orig_data) != len(ours_data):
            print(f"Length is different!")
            min_len = min(len(orig_data), len(ours_data))
            orig_data = orig_data[:min_len]
            ours_data = ours_data[:min_len]
            print(f"Cutting to : {min_len}")
        routing_output_dir = ours_output_dir.replace(f'/{self.args.method}/', '/routing/')
        os.makedirs(routing_output_dir, exist_ok=True)
        print(f"Made output routing dir: {routing_output_dir}")
        accs = []
        total_len = len(ours_data)
        print(f'Data len: {total_len}')
        orig_cnt = 0
        new_data = []
        for orig, ours in zip(orig_data, ours_data):
            orig_c = orig['confidence']
            ours_c = ours['confidence']
            new = copy.deepcopy(orig)
            new['conf_orig'] = orig_c
            new['conf_ours'] = ours_c
            if type(orig_c) == list:
                orig_c = sum(orig_c)/len(orig_c)
                ours_c = sum(ours_c)/len(ours_c)
            if (orig_c + self.args.routing_alpha) <= ours_c:
                pred = ours['prediction']
            else:
                orig_cnt += 1
                pred = orig['prediction']
            new['prediction'] = pred
            if self.args.data == 'structlm':
                acc = self.DataRouter.get_accuracy_instance(pred, new)
            elif self.args.data == 'mmlu':
                acc = self.DataRouter.get_accuracy_instance(pred, new['answer'])
            else:
                acc = self.DataRouter.get_accuracy_instance(pred, new['answers'])
            new['accuracy'] = acc
            new_data.append(new)
        all_acc = self.DataRouter.get_accuracy_all(new_data)
        print(f"{'@'*20}\n{self.args.method} | sorting method: {self.args.sorting_method} | alpha: {self.args.routing_alpha}")
        print(all_acc['acc'])
        print(f"{'@'*20}\n")
        json.dump(new_data, open(routing_output_dir + '/result.json', "w"), indent=4)
        print(f"select portion: orig {orig_cnt}/{total_len} ({100*orig_cnt/total_len:.2f}%), ours {total_len-orig_cnt}/{total_len} ({100*(total_len-orig_cnt)/total_len:.2f}%)")
        print('finish!')
        return

    def run_eval(self):
        total_len = self.DataRouter.total_length
        with open(f"{self.args.output_dir}/result.json", 'r') as f:
            data = json.load(f)
        data = data[:total_len]
        accs = []
        print(f'Data len: {len(data)}')
        all_acc = self.DataRouter.get_accuracy_all(data)
        print(f"Metric output: {all_acc}")

        excel_cols = ', '.join([str(x).split('/')[-1] for x in all_acc.keys()])
        excel_acc = ', '.join([str(x) for x in all_acc.values()])
        print(f"\n\nExcelized:\n\nfor {self.args.output_dir}\n{excel_cols}\n{excel_acc}\n\n")
        return

    def run_main(self):
        #preds, refs = [], []
        output_data = []
        error_table_ids = set()
        acc_by_index = []
        self.oom_idx = []
        self.none_count = 0
        # set output path
        self.oom_idx = []
        total_len = self.DataRouter.total_length
        confidence_list = []
        output_path = f"{self.args.output_dir}/result.json"
        if self.args.resume:
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            bkup_path = f"{self.args.output_dir}/result_backup.json"
            json.dump(output_data, open(bkup_path, "w"), indent=4)

        for i in tqdm(range(total_len)):
            if self.args.resume and i < len(output_data):
                continue
            input_data = self.DataRouter.get_data_by_index(i)
            output, confidence = self.run_generation(input_data)
            pred = self.DataRouter.parse_output(output)

            if pred == 'oom':
                self.oom_idx.append(i)
                continue
            output_instance = copy.deepcopy(input_data['full'])
            output_instance["prediction"] = pred
            output_instance['confidence'] = confidence
            if pred == 'None':
                self.none_count += 1
            acc = self.DataRouter.get_accuracy_instance(pred, input_data['answers'])

            output_instance["accuracy"] = acc
            output_data.append(output_instance)
            if i % self.args.print_freq == 0:
                 pred_results = self.DataRouter.get_accuracy_all(output_data)
                 print(f"\nPred results for {i}: {pred_results}, none_count: {self.none_count}\n, oom_cnt: {len(self.oom_idx)}")
                 json.dump(output_data, open(output_path, 'w'), indent=4)
            if i % self.args.print_freq == 0:
                print(f"Input prefix: {input_data['prefix']}\n\nParallel texts: \n{''.join(input_data['parallel'])}\n\nSuffix: \n{input_data['suffix']}")
                print(f"[Answer]: {input_data['answers']} \n[Output:] {output} \n[parsed:] {pred}")
                if acc == 1:
                    print(f"Correct ✅\n")
                else:
                    print(f"Incorrect ❌ acc {acc}\n")
                acc = self.DataRouter.get_accuracy_all(output_data)
                print(f"\nAcc up to now: {acc}\n")
        json.dump(output_data, open(output_path, "w"), indent=4)
        prediction_results = self.DataRouter.get_accuracy_all(output_data)
        print(prediction_results)
        output_prediction_file = f"{output_dir}/logs.json"
        logstr = f"\nPred results for final: {prediction_results}, none_count: {self.none_count}\n, oom_cnt: {len(self.oom_idx)}"
        print(logstr)
        if self.args.measure_flops:
            self.prof.stop_profile()
            flops = self.prof.get_total_flops()
            prediction_results['flops'] = flops
            print(f"\n\nFLOPS: {flops}\n\n")
        prediction_results['logstr'] = logstr
        prediction_results['oom_idx'] = self.oom_idx
        json.dump(prediction_results, open(output_prediction_file, "w"), indent=4)
        print('run main done!')

def validate_args(args):
    if args.method == 'pcw':
        # You should have pcw window k if option is pcw.
        if args.pcw_window_k == -1:
            raise Exception("you should set pcw_window_k to specific value for method pcw.")
    if args.mode != '':
        print(f'Running on mode: {args.mode}')
    else:
        print(f"No args.mode specified!")
    if args.inference_type == 'log_likelihood':
        print(f"You should have the 'candidates' key for data for this option (log likelihood)!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--method", type=str, default='orig', choices=['orig', 'ours', 'pcw', 'pine'])
    # dataset loading args
    parser.add_argument("--data", type=str, default='robut')
    parser.add_argument("--split", type=str, default='wtq')
    parser.add_argument('--subsplit', type=str, default='') #, choices=['', '0','4','9','14','19','24','29'])
    parser.add_argument("--name", type=str, default="empty_name", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--base_dir', type=str, default='/workspace/TableInvariance')

    # generation args
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--resume", action='store_true')

    # options
    parser.add_argument('--sorting_method', type=str, default='lexical', choices=['lexical', 'monot5', 'freq'])
    parser.add_argument('--mode', type=str, default='' )#choices=['no_indexing', 'random_shuffle', 'template_swap'])
    parser.add_argument('--add_row_number', action='store_true')
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--pcw_window_k', type=int, default=-1)
    parser.add_argument('--keep_special_tokens', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--confidence', action='store_true')
    parser.add_argument('--inference_type', type=str, default='generation') # generation, loglikelihood

    # print options
    parser.add_argument('--print_freq', type=int, default=300)
    parser.add_argument('--data_split_total', type=int, default=-1)
    parser.add_argument('--data_split_cur', type=int, default=-1)
    # debug
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--onlyeval', action='store_true')
    parser.add_argument('--routing', action='store_true')
    parser.add_argument('--routing_alpha', type=float, default=0.2)

    args = parser.parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.measure_flops:
        from deepspeed.profiling.flops_profiler import FlopsProfiler

    output_dir = f"./outputs/{args.data}/{args.split}/{args.method}/{args.split}/{args.name}/{args.model_name}"
    if args.subsplit != '':
        output_dir += f"/subsplit-{args.subsplit}"
    if args.mode != '':
        output_dir += f"/mode-{args.mode}"
    if args.sorting_method != 'lexical':
        output_dir += f"/sorting-method-{args.sorting_method}"
    if args.seed != 0:
        output_dir += f"/seed-{args.seed}"
    if args.data_split_total != -1:
        output_dir += f"/split-{args.data_split_cur}of{args.data_split_total}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    evaluator = EvalPipeline(args)
    if args.onlyeval:
        evaluator.run_eval()
    elif args.routing:
        evaluator.run_selective_routing()
    else:
        evaluator.run_main()
        print(f"Args: {args}, output_dir: {output_dir}")
        print('-'*50)
