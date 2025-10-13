import pickle
import numpy as np
import torch
from datasets import Dataset
import pandas as pd
import json
from typing import List, Dict, Union, Optional, Any
import collections
from pathlib import Path
import re
from datasets import load_dataset, concatenate_datasets
import gzip
import tqdm
import random


# format prompt for model input
def format_prompt_for_chat(dataset, model_name, tokenizer, prompt_template, few_shot_examples, **kwargs):
    def apply_template(prompt, prompt_template, **kwargs):
        if any(f"{{{key}}}" in prompt_template for key in kwargs):
            prompt = prompt_template.format(user_prompt=prompt, **kwargs)
        else:
            prompt = prompt_template.format(user_prompt=prompt)
        return prompt

    prompts = []

    # for base models: add few-shot-examples; for chat models, set to ""
    is_chat_model = any(suffix in model_name.lower() for suffix in ['chat', 'instruct'])
    if is_chat_model or not few_shot_examples:
        few_shot_examples = ""
        prompt_template = prompt_template['chat']
    else:
        prompt_template = prompt_template['non_chat']
    
    # dataset may be a DataFrame or a Dataset object
    records = dataset.to_dict('records') if isinstance(dataset, pd.DataFrame) else dataset
    for data in records:
        # get keywards of row
        prompt = data['prompt']
        responses = data.get('truncated_responses', None)
        info = data.get('info', None)
        
        prompt = apply_template(prompt, prompt_template, info=info, few_shot_examples=few_shot_examples)
        if is_chat_model:
            if kwargs.get('to_ids', False):
                # return: input_ids, attention_mask, labels
                full_qa = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": data['response']}
                    ],
                    add_generation_prompt=False,
                    skip_special_tokens=False,
                    tokenize=False,
                )
                full_qa = tokenizer.encode(full_qa, padding=True, truncation=True, max_length=512)
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    add_generation_prompt=True,
                    skip_special_tokens=False,
                    tokenize=False,
                )
                prompt = tokenizer.encode(prompt, padding=True, truncation=True, max_length=512)
                labels = full_qa.copy()
                labels[:len(prompt)] = [tokenizer.pad_token_id] * len(prompt)
                prompts.append({
                    "input_ids": full_qa,
                    "attention_mask": [1] * len(full_qa),
                    "labels": labels
                })
                continue
                
            else:
                prompt = tokenizer.apply_chat_template(
                    [
                        # {"role": "system", "content": ""},
                        {"role": "user", "content": prompt},
                    ],
                    add_generation_prompt=True,
                    skip_special_tokens=False,
                    tokenize=False,
                )
        else:
            pass
            
        if responses:
            prompts.extend(prompt + response for response in responses)
        else:
            prompts.append(prompt)

    return prompts







class DatasetManager:
    """Manages datasets for different pipeline stages with improved organization and type hints."""
    
    VALID_STAGES = {'collecting_response', 'labeling_and_augmentation', 'collecting_features', 'testing_probes'}
    
    def __init__(self, args, running_stage: str):
        if running_stage not in self.VALID_STAGES:
            raise ValueError(f"running_stage must be one of {self.VALID_STAGES}")
            
        self.seed = getattr(args, 'seed', 0)
        np.random.seed(self.seed)
        self.running_stage = running_stage
        self.prompt_template_type = args.prompt_template_type
        self.model_name = args.model_name
        self.output_dir = None

        # Set attributes from args
        self._set_attributes_from_args(args)
        
        # Load appropriate dataset based on running stage
        self._load_stage_specific_data(args)

    def _set_attributes_from_args(self, args):
        """Set class attributes from command line arguments"""
        for attr in ['dataset_name', 'train_dataset_name', 'test_dataset_name']:
            if hasattr(args, attr) and getattr(args, attr):
                setattr(self, f"_{attr}", getattr(args, attr).split('/')[-1])

        for attr in ['test_ratio', 'train_val_ratio', 'in_data_prefix', 'out_data_prefix', 'in_train_data_prefix', 'in_test_data_prefix']:
            if hasattr(args, attr):
                setattr(self, f"_{attr}", getattr(args, attr))
                
        for attr in ['activation_model_name']:
            if hasattr(args, attr):
                setattr(self, f"_{attr}", getattr(args, attr))
                
        for attr in ['shuffle_activations', 'randomize_labels', 'non_trained_probes', 'process_step']:
            if hasattr(args, attr):
                setattr(self, f"_{attr}", getattr(args, attr))

    def _get_path(self, data_type: str, dataset_name: str, prefix:str, suffix: str, sub_dir='') -> Path:
        """Generate path for data files"""
        base_path = Path("./data")
        type_paths = {
            'response': base_path / "responses",
            'labeled': base_path / "labeled_responses",
            'features': base_path / "features",
            'results': base_path / "results",
            'probe_states': base_path / "probe_states",
        }
        model_name = self.model_name
        if hasattr(self, '_activation_model_name') and data_type in ['features', 'results', 'probe_states'] and self._activation_model_name != "":
            model_name = f'{model_name}_{self._activation_model_name}'
            sub_dir = sub_dir + f"cross/"
        if getattr(self, '_process_step', -1) >= 0 and 'results' in suffix:
            sub_dir = sub_dir + f"process/"
        if (getattr(self, '_non_trained_probes', False) or getattr(self, '_randomize_labels', False) or getattr(self, '_shuffle_activations', False)) and 'results' in suffix:
            sub_dir = sub_dir + f"ablation/"
            
        return type_paths[data_type] / dataset_name / f"{sub_dir}{prefix}{self.prompt_template_type}_{model_name}_{suffix}"

    # for process study naming
    def _is_the_same_dataset(self):
        return self._train_dataset_name == self._test_dataset_name #or 'unfolding' in self._in_test_data_prefix

    def _load_stage_specific_data(self, args):
        """Load data specific to the running stage"""
        if self.running_stage == 'collecting_response':
            self.dataset = self._load_raw_datasets_and_transform(self._dataset_name)
            
        elif self.running_stage == 'labeling_and_augmentation':
            path = self._get_path('response', self._dataset_name, self._in_data_prefix, 'responses.pkl')
            self.dataset = pd.read_pickle(path)
            
        elif self.running_stage == 'collecting_features':
            path = self._get_path('labeled', self._dataset_name, self._in_data_prefix,'labeled_responses.pkl')
            self.dataset = pd.read_pickle(path)
            
        elif self.running_stage == 'testing_probes':
            self._load_test_data()

    def _load_test_data(self):
        """Load and process data for testing probes stage"""
        # Load datasets
        self.train_dataset = pd.read_pickle(self._get_path('labeled', self._train_dataset_name, self._in_train_data_prefix,'labeled_responses.pkl'))
        self.test_dataset = pd.read_pickle(self._get_path('labeled', self._test_dataset_name, self._in_test_data_prefix,'labeled_responses.pkl'))
        # if index is not continue: reset index
        if not all(self.train_dataset.index == np.arange(len(self.train_dataset))):
            self.train_dataset.reset_index(drop=True, inplace=True)
        if not all(self.test_dataset.index == np.arange(len(self.test_dataset))):
            self.test_dataset.reset_index(drop=True, inplace=True)


        # Load activations
        self.train_activations = torch.load(self._get_path('features', self._train_dataset_name, self._in_train_data_prefix,'layer_wise.pt'))
        self.test_activations = torch.load(self._get_path('features', self._test_dataset_name, self._in_test_data_prefix,'layer_wise.pt'))
        # record all labels of the train_set
        self.all_train_labels = np.concatenate(self.train_dataset['augmented_labels'].values)
        # record all labels of the test_set
        self.all_test_labels = np.concatenate(self.test_dataset['augmented_labels'].values)

        # Split indices and format data
        self._beforeward_process() # for process study: fetch all data at the same step
        self._get_split_idxs()
        self._split_and_format_data_for_testing()
        self._afterward_process() # for ablation study: change activations and labels

    def _beforeward_process(self):
        """Process data at a specific step index before train/val/test splitting."""
        if not hasattr(self, '_process_step') or self._process_step < 0 or self.running_stage != 'testing_probes':
            return

        def get_step_value(sequence):
            """Get value at process_step or last value if out of range."""
            idx = min(self._process_step, len(sequence) - 1)
            return [sequence[idx]]
            # return [sequence[idx]]

        def process_dataset(dataset, activations):
            """Process a single dataset and its activations at the specified step."""
            # Process activations
            # if activations is not None:
            offsets = np.cumsum([0] + [len(labels) for labels in dataset['augmented_labels']])
            indices = [offset + min(self._process_step, next_offset - offset - 1) 
                    for offset, next_offset in zip(offsets[:-1], offsets[1:])]
            # return dataset
        
            # Process list-type columns (augmented_labels and truncated_responses)
            for col in ['augmented_labels', 'truncated_responses']:
                if col in dataset.columns:
                    dataset[col] = dataset[col].map(get_step_value)

            return dataset, activations[indices]

        # Process both train and test datasets
        # self.train_dataset, self.train_activations = process_dataset(self.train_dataset, self.train_activations)
        self.test_dataset, self.test_activations = process_dataset(self.test_dataset, self.test_activations)


    def _get_split_idxs(self):
        """Split datasets into train/val/test sets"""
        # if self._train_dataset_name == self._test_dataset_name:
        if self._is_the_same_dataset():
            n_samples = len(self.train_dataset)
            self.train_idxs = np.random.choice(range(n_samples), 
                                             size=int(n_samples * (1 - self._test_ratio)), 
                                             replace=False)
            self.test_idxs = np.array([x for x in range(n_samples) if x not in self.train_idxs])
        else:
            self.train_idxs = np.arange(len(self.train_dataset))
            self.test_idxs = np.arange(len(self.test_dataset))
        
        train_idxs, val_idxs = np.split(
            np.random.permutation(self.train_idxs),
            [int(len(self.train_idxs) * self._train_val_ratio)]
        )
        self.train_idxs, self.val_idxs = train_idxs, val_idxs

    def _split_and_format_data_for_testing(self):
        """Format activations and labels for testing"""
        self.x_train, self.y_train = self._format_activations_and_labels(self.train_dataset, self.train_activations, self.train_idxs, split='train')
        self.x_val, self.y_val = self._format_activations_and_labels(self.train_dataset, self.train_activations, self.val_idxs, split='val')
        self.x_test, self.y_test = self._format_activations_and_labels(self.test_dataset, self.test_activations, self.test_idxs, split='test')
        
        self.val_dataset = self.train_dataset.iloc[self.val_idxs]
        self.train_dataset = self.train_dataset.iloc[self.train_idxs]
        self.test_dataset = self.test_dataset.iloc[self.test_idxs]

    # @staticmethod
    def _format_activations_and_labels(self, dataset: pd.DataFrame, activations: torch.Tensor, indices: np.ndarray, split=None):
        """Format activations and labels for model training"""
        augmented_labels = dataset['augmented_labels']
        indices_list = indices.tolist()
        
        selected_labels = np.array([label for i in indices_list for label in augmented_labels[i]])

        label_lengths = np.cumsum([0] + [len(labels) for labels in augmented_labels])
        selected_activations = torch.cat([
            activations[label_lengths[i]:label_lengths[i+1]]
            for i in indices_list
        ])

        # assert selected_activations.shape[0] == selected_labels.shape[0]
        
        return selected_activations, selected_labels
    
    def _afterward_process(self):
        """Perform ablation studies by shuffling activations or randomizing labels if specified."""
        if hasattr(self, '_shuffle_activations') and self._shuffle_activations:
            # Shuffle activations independently for train, val, and test sets
            self.x_train = self.x_train[torch.randperm(self.x_train.size(0))]
            self.x_val = self.x_val[torch.randperm(self.x_val.size(0))]
            self.x_test = self.x_test[torch.randperm(self.x_test.size(0))]
            
        if hasattr(self, '_randomize_labels') and self._randomize_labels:
            # Generate completely new random labels
            def randomize_labels(y):
                if isinstance(y, torch.Tensor):
                    min_val = y.min()
                    max_val = y.max()
                    return torch.rand(y.shape) * (max_val - min_val) + min_val
                elif isinstance(y, np.ndarray):
                    min_val = y.min()
                    max_val = y.max()
                    return np.random.uniform(min_val, max_val, y.shape).astype(y.dtype)
                else:
                    raise TypeError("Labels must be either torch.Tensor or np.ndarray")
            
            self.y_train = randomize_labels(self.y_train)
            self.y_val = randomize_labels(self.y_val)
            self.y_test = randomize_labels(self.y_test)
            
            # Update the datasets to maintain consistency
            def update_dataset_labels(dataset, new_labels):
                label_counts = [len(labels) for labels in dataset['augmented_labels']]
                start_idx = 0
                new_augmented_labels = []
                for count in label_counts:
                    new_augmented_labels.append(new_labels[start_idx:start_idx + count])
                    start_idx += count
                dataset['augmented_labels'] = new_augmented_labels
                return dataset
            
            self.train_dataset = update_dataset_labels(self.train_dataset, self.y_train)
            self.val_dataset = update_dataset_labels(self.val_dataset, self.y_val)
            self.test_dataset = update_dataset_labels(self.test_dataset, self.y_test)

    def _add_ablation_suffix(self, base_suffix, result_type) -> str:
            """Generate consistent suffix for ablation studies and process steps.
            Args:
                base_suffix: Base suffix to append ablation flags to
            Returns:
                Modified suffix with ablation flags
            """
            ablation_flags = {
                '_shuffle_activations': lambda x: x,
                '_randomize_labels': lambda x: x,
                '_non_trained_probes': lambda x: x,
                '_process_step': lambda x: x >= 0
            }
            
            suffix = base_suffix
            for key, condition in ablation_flags.items():
                if hasattr(self, key) and condition(getattr(self, key)):
                    if key == '_process_step':
                        if result_type == list:
                            suffix = f"{key[1:]}_{getattr(self, key)}_{base_suffix}"
                        # suffix = f"{key[1:]}_{getattr(self, key)}_{base_suffix}"
                    else:
                        suffix = f"{key[1:]}_{base_suffix}"
            
            return suffix

    def save_results(self, results: Union[pd.DataFrame, torch.Tensor, List[Dict]], **kwargs):
        """Save results based on their type, handling different formats appropriately."""

        # Create save path based on result type and stage
        if isinstance(results, pd.DataFrame):
            assert self.running_stage in ['collecting_response', 'labeling_and_augmentation']
            suffix = 'responses.pkl' if self.running_stage == 'collecting_response' else 'labeled_responses.pkl'
            data_type = 'response' if self.running_stage == 'collecting_response' else 'labeled'
            save_path = self._get_path(data_type, self._dataset_name, self._out_data_prefix, suffix)
        elif isinstance(results, collections.OrderedDict):
            # raise ValueError("Saving activations is not supported in this method. Use save_activation_chunks instead.")
            # save_path = self._get_path('features', self._dataset_name, self._out_data_prefix, 'layer_wise.pt')
            assert self.running_stage in ['testing_probes'] and 'layer_name' in kwargs
            suffix = f"layer_{kwargs['layer_name']}_hiddenSize_{kwargs['hidden_size']}.pt"
            # if ablation: add additional suffix
            suffix = self._add_ablation_suffix(suffix, type(results))
            save_path = self._get_path('probe_states', self._train_dataset_name, self._out_data_prefix, suffix, sub_dir=f'{self.seed}/') # saving models
        elif isinstance(results, list):
            assert self.running_stage in ['testing_probes']
            if self._is_the_same_dataset():
                dataset_name = self._train_dataset_name
            else:
                dataset_name = f'{self._train_dataset_name}_{self._test_dataset_name}'
            suffix = f'results.json'
            # suffix = f'hiddenSize_{kwargs["hidden_size"]}_results.json'
            # if ablation: add additional suffix
            suffix = self._add_ablation_suffix(suffix, type(results))
            save_path = self._get_path('results', dataset_name, self._out_data_prefix, suffix, sub_dir=f'{self.seed}/')
        else:
            raise ValueError(f"Unsupported result type: {type(results)}")

        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on type
        if isinstance(results, pd.DataFrame):
            results.to_pickle(save_path)
            # save as json
            pd_to_json(results, save_path.with_suffix('.json'), compress=False)
            tqdm.tqdm.write(f"Results saved to: {save_path} and {save_path.with_suffix('.json')}")
        elif isinstance(results, collections.OrderedDict):
            torch.save(results, save_path)
        else:  # List[Dict]
            formatted_data = [{k: v.replace('\n', '[n]') if isinstance(v, str) else v 
                             for k, v in item.items()} for item in results]
            save_json(formatted_data, save_path)
            tqdm.tqdm.write(f"Results saved to: {save_path}")

    def save_activation_chunks(self, activations: torch.Tensor, chunk_id: int) -> Path:
        """Save activation chunk to temporary file."""
        save_path = self._get_path('features', self._dataset_name, self._out_data_prefix, f'tmp_layer_wise_{chunk_id}.pt')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activations, save_path)
        return save_path

    def concat_and_save_activations(self, final_activations: torch.Tensor, 
                                  num_chunks: int, cleanup: bool = True) -> None:
        """Concatenate all activation chunks and save final result."""
        if num_chunks == 0:
            # If no chunks were saved, just save the current activations
            save_path = self._get_path('features', self._dataset_name, self._out_data_prefix, 'layer_wise.pt')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_activations, save_path)
            return

        # Load all chunks
        activation_chunks = []
        for i in range(num_chunks):
            tmp_path = self._get_path('features', self._dataset_name, self._out_data_prefix, f'tmp_layer_wise_{i}.pt')
            activation_chunks.append(torch.load(tmp_path))
            
        # Add final chunk if it's not empty
        if final_activations.shape[0] > 0:
            activation_chunks.append(final_activations)

        # Concatenate and save
        combined_activations = torch.cat(activation_chunks, dim=0)
        save_path = self._get_path('features', self._dataset_name, self._out_data_prefix, 'layer_wise.pt')
        torch.save(combined_activations, save_path)

        tqdm.tqdm.write(f"Final activations saved to: {save_path}")

        # Cleanup temporary files
        if cleanup:
            for i in range(num_chunks):
                tmp_path = self._get_path('features', self._dataset_name, self._out_data_prefix, f'tmp_layer_wise_{i}.pt')
                tmp_path.unlink(missing_ok=True)
        

    def load_probes(self, hidden_sizes, model_layer_num):
        """Load probe models for all layers. Returns {layer_name: probe_model} dict."""
        import torch
        
        probes = {}
        # init
        for hidden_size in hidden_sizes:
            probes[str(hidden_size)] = {}

        # First try loading all numbered layers until a gap is found
        for hidden_size in hidden_sizes:
            for layer_idx in range(model_layer_num+1): 
                layer_name = str(layer_idx) if layer_idx < model_layer_num else 'all_layers'
                probe_path = self._get_path('probe_states', self._train_dataset_name,
                                        self._out_data_prefix,
                                        f"{self._add_ablation_suffix('', collections.OrderedDict)}layer_{layer_name}_hiddenSize_{hidden_size}.pt",
                                        sub_dir=f'{self.seed}/')
                if not probe_path.exists():
                    probes[str(hidden_size)][layer_name] = None
                else: 
                    probes[str(hidden_size)][layer_name] = torch.load(probe_path)
            
        return probes

    def check_if_results_exists(self, HIDDEN_SIZES=None):
        """Check if results already exist for the current stage"""

        if self.running_stage == 'collecting_response':
            file_path = self._get_path('response', self._dataset_name, self._out_data_prefix, 'responses.pkl')
            # load the file and output the existing row num of data
            already_collected_data_num = 0
            if file_path.exists():
                already_collected_data_num = len(pd.read_pickle(file_path))
            return (file_path.exists(), already_collected_data_num, file_path)

        elif self.running_stage == 'labeling_and_augmentation':
            return self._get_path('labeled', self._dataset_name, self._out_data_prefix, 'labeled_responses.pkl').exists()
        elif self.running_stage == 'collecting_features':
            final_file_exists = self._get_path('features', self._dataset_name, self._out_data_prefix, 'layer_wise.pt').exists()
            if final_file_exists:
                return True, 0
            else:
                # Check for temporary files
                chunk_start_idx = 0
                for i in range(1000):
                    tmp_path = self._get_path('features', self._dataset_name, self._out_data_prefix, f'tmp_layer_wise_{i}.pt')
                    if not tmp_path.exists():
                        break
                    chunk_start_idx += 1
                return False, chunk_start_idx
            
        elif self.running_stage == 'testing_probes':
            # Handle dataset name combination consistent with save_results
            if self._test_dataset_name == self._train_dataset_name:
                dataset_name = self._train_dataset_name
            else:
                dataset_name = f'{self._train_dataset_name}_{self._test_dataset_name}'
            
            # Check results.json with proper suffixes
            results_suffix = 'results.json'
            results_suffix = self._add_ablation_suffix(results_suffix, list)
            results_path = self._get_path('results', dataset_name, self._out_data_prefix, results_suffix, 
                                        sub_dir=f'{self.seed}/')
            results_path_exists = results_path.exists()
        
            
            # Check probe state for first layer with proper suffixes
            probe_suffix = "layer_all_layers.pt"
            probe_suffix = self._add_ablation_suffix(probe_suffix, collections.OrderedDict)
            probe_path = self._get_path('probe_states', self._train_dataset_name, self._out_data_prefix,
                                    probe_suffix, sub_dir=f'{self.seed}/')
            probe_exists = all(probe_path.with_name(probe_path.name.replace(f'.pt', f'_hiddenSize_{hidden_size}.pt')).exists() for hidden_size in HIDDEN_SIZES)
            # return results_path.exists() and probe_path.exists()
            return (probe_exists, results_path_exists)
        
            # half-way check: part of the probes exists, but not all -> then load the existing probes
        else:
            raise ValueError(f"Unsupported running stage: {self.running_stage}")
        return False

    @staticmethod
    def _load_raw_datasets_and_transform(dataset_name):
        # load dataset and transform to specific form: prompt, options, catogory,  label
        # alpaca_eval
        if dataset_name == "tatsu-lab/alpaca_eval" or dataset_name == "alpaca_eval":
            dataset = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)["eval"]
            def transform_fn(batch):
                transformed_batch = {
                    'prompt': batch['instruction'],
                    'options': [[]] * len(batch['instruction']),
                    'info': [o.replace('\n', '[n]') for o in batch['output']],
                    'label': [1] * len(batch['instruction']),
                }
                return transformed_batch
            dataset = dataset.map(transform_fn, batched=True)
        # ultraChat
        elif dataset_name == "stingning/ultrachat" or dataset_name == "ultrachat":
            dataset = load_dataset("stingning/ultrachat")['train']
            dataset = dataset.select(range(10, 3010))
            def transform_fn(batch):
                return {
                    'prompt': [data[0] for data in batch['data']],
                    'options': [[]] * len(batch['data']),
                    'info': [''] * len(batch['data']),
                    'label': [1] * len(batch['data']),
                }
            dataset = dataset.map(transform_fn, batched=True)
        # openai/gsm8k
        elif dataset_name == "openai/gsm8k" or dataset_name == "gsm8k":
            dataset = load_dataset("openai/gsm8k", 'main')['train']
            dataset = dataset.select(range(3000))
            def transform_fn(batch):
                return {
                    'prompt': batch['question'],
                    'options': [[]] * len(batch['question']),
                    'info': batch['answer'],
                    'label': [1] * len(batch['question']),
                }
            dataset = dataset.map(transform_fn, batched=True)
            
        # hendrycks/math
        elif dataset_name in ["hendrycks/math", "math", "lighteval/MATH", "MATH"]:
            full_dataset = load_dataset("lighteval/MATH", "algebra")
            dataset = concatenate_datasets([full_dataset[split] for split in ['train', 'test']])
            # select out first 10 as few-shot examples
            dataset = dataset.select(range(10, len(dataset)))
            def transform_fn(batch):
                return {
                    'prompt': batch['problem'],
                    'options': [[]] * len(batch['problem']),
                    'info': batch['level'],
                    'label': [1] * len(batch['problem']),
                }
            dataset = dataset.map(transform_fn, batched=True)
            
 
        ## multiple_choice (label:0~3)
        # openlifescienceai/medmcqa
        elif dataset_name == "openlifescienceai/medmcqa" or dataset_name == "medmcqa":
            dataset = load_dataset("openlifescienceai/medmcqa")['train']
            dataset = dataset.select(range(10, 8010))
            def transform_fn(batch):
                prompts = [f"{q}\n\nA) {opa}\nB) {opb}\nC) {opc}\nD) {opd}" for q, opa, opb, opc, opd in 
                        zip(batch['question'], batch['opa'], batch['opb'], batch['opc'], batch['opd'])]
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': batch['exp'],
                    'label': batch['cop']
                }
            dataset = dataset.map(transform_fn, batched=True)
            
        # "allenai/ai2_arc_Challenge" or "allenai/ai2_arc_Easy"
        elif dataset_name in ["allenai/ai2_arc-Challenge", "ai2_arc-Challenge", "allenai/ai2_arc-Easy", "ai2_arc-Easy"]:
            if dataset_name == "allenai/ai2_arc-Challenge" or dataset_name == "ai2_arc-Challenge":
                full_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
            else:
                full_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
            dataset = concatenate_datasets([full_dataset[split] for split in ['train', 'test', 'validation']])
            dataset = dataset.select(range(2500))
            def transform_fn(batch):
                prompts = []
                for q, choices in zip(batch['question'], batch['choices']):
                    options = choices['text']
                    # Format with consistent label style but variable number of options
                    prompt = f"{q}\n\n"
                    prompt += '\n'.join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
                    prompts.append(prompt)
                    
                def convert_label(label):
                    if label in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:  # Handle letter labels
                        return ord(label) - ord('A')
                    elif str(label).isdigit():  # Handle numeric labels
                        return int(label) - 1
                    else:  # Handle any other unexpected cases
                        raise ValueError(f"Unexpected label format: {label}")
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': [[]] * len(prompts),
                    'label': [convert_label(l) for l in batch['answerKey']]
                }
            dataset = dataset.map(transform_fn, batched=True)
        # timchen0618/Kialo
        elif dataset_name.lower() in ["timchen0618/kialo", "kialo"]:
            dataset = load_dataset("timchen0618/Kialo")['test']
            
            # Filter for yes-no questions (questions that have exactly two opposing perspectives)
            def is_yes_no_question(question, perspectives):
                if len(perspectives) != 2:
                    return False
                
                first_word = question.strip().split()[0].lower()
                yes_no_starters = {
                    'will', 'is', 'are', 'should', 'do', 'does', 'can', 'would', 'has', 'have'
                }
                return first_word in yes_no_starters
            
            # First, filter yes-no questions
            filtered_data = {
                'question': [],
                'perspectives': []
            }
            
            for question, perspectives in zip(dataset['question'], dataset['perspectives']):
                if is_yes_no_question(question, perspectives):
                    filtered_data['question'].append(question)
                    filtered_data['perspectives'].append(perspectives)
            
            # Then transform based on the filtered index
            def transform_fn(batch):
                prompts = []
                for idx, perspectives in enumerate(batch['perspectives']):
                    # Choose first perspective for even indices, second for odd indices
                    chosen_perspective = perspectives[idx % 2]
                    prompts.append(chosen_perspective)
                
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': [''] * len(prompts),
                    'label': [1] * len(prompts),
                }
            
            # Create new dataset from filtered data
            filtered_dataset = Dataset.from_dict(filtered_data)
            dataset = filtered_dataset.map(transform_fn, batched=True, remove_columns=filtered_dataset.column_names)
            dataset = dataset.select(range(min(len(dataset), 3000)))
        
        # roneneldan/TinyStories
        elif dataset_name in ['roneneldan/TinyStories', 'TinyStories']:
            dataset = load_dataset("roneneldan/TinyStories")['train']
            dataset = dataset.select(range(10, 20000))
        
            def transform_fn(batch):
                prompts = []
                for story in batch['text']:
                    match = re.match(r'[^.!?]*[.!?]', story)
                    first_sentence = match.group(0) if match else story
                    
                    prompts.append(first_sentence)
                
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': [''] * len(prompts),
                    'label': [1] * len(prompts)
                }
            
            processed_dataset = dataset.map(transform_fn, batched=True, remove_columns=dataset.column_names)
            return processed_dataset

        # Ximing/ROCStories
        elif dataset_name in ['Ximing/ROCStories', 'ROCStories']:
            full_dataset = load_dataset("Ximing/ROCStories")
            dataset = concatenate_datasets([full_dataset[split] for split in ['train', 'test', 'validation']])
            def transform_fn(batch):
                return {
                    'prompt': batch['prompt'],
                    'options': [[]] * len(batch['prompt']),
                    'info': [''] * len(batch['prompt']),
                    'label': [1] * len(batch['prompt'])
                }
            
            processed_dataset = dataset.map(transform_fn, batched=True, remove_columns=dataset.column_names)
            return processed_dataset
        
        # commonsense-qa
        elif dataset_name in ['commonsense_qa', 'tau/commonsense_qa']:
            dataset = load_dataset("tau/commonsense_qa")['train']  # You might want to add validation/test splits
            dataset = dataset.select(range(5000))
            
            def transform_fn(batch):
                prompts = []
                options = []
                labels = []
                
                for question, choices, ans_key in zip(batch['question'], batch['choices'], batch['answerKey']):
                    # Format the multiple choice options
                    choice_texts = choices['text']
                    choice_labels = choices['label']
                    
                    # Create formatted options list
                    formatted_options = []
                    for label, text in zip(choice_labels, choice_texts):
                        formatted_options.append(f"{label}) {text}")
                    
                    # Create the full prompt with question and options
                    full_prompt = f"{question}\n\n" + "\n".join(formatted_options)
                    
                    prompts.append(full_prompt)
                    # options.append(formatted_options)
                    # Convert answer key (A/B/C/D/E) to index (0,1,2,3,4)
                    label_idx = ord(ans_key) - ord('A')
                    labels.append(label_idx)
                
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': [''] * len(prompts),  # Keeping consistent with other datasets
                    'label': labels
                }
            
            processed_dataset = dataset.map(transform_fn, batched=True, remove_columns=dataset.column_names)
            return processed_dataset
    
        # allenai/social_i_qa
        elif dataset_name in ['allenai/social_i_qa', 'social_i_qa']:
            dataset = load_dataset("allenai/social_i_qa", trust_remote_code=True)['train']  # You might want to add validation/test splits
            dataset = dataset.select(range(5000))
            
            def transform_fn(batch):
                prompts = []
                options = []
                labels = []
                
                for context, question, ans1, ans2, ans3, label in zip(
                    batch['context'], 
                    batch['question'],
                    batch['answerA'],
                    batch['answerB'],
                    batch['answerC'],
                    batch['label']
                ):
                    # Create formatted options list
                    formatted_options = [
                        f"A) {ans1}",
                        f"B) {ans2}",
                        f"C) {ans3}"
                    ]
                    
                    # Create the full prompt with context, question and options
                    full_prompt = f"Context: {context}\nQuestion: {question}\n\n" + "\n".join(formatted_options)
                    
                    prompts.append(full_prompt)
                    # options.append(formatted_options)
                    # Convert label (1/2/3) to index (0/1/2)
                    label_idx = int(label) - 1
                    labels.append(label_idx)
                
                return {
                    'prompt': prompts,
                    'options': [[]] * len(prompts),
                    'info': [''] * len(prompts),  # Keeping consistent with other datasets
                    'label': labels
                }
            
            processed_dataset = dataset.map(transform_fn, batched=True, remove_columns=dataset.column_names)
            return processed_dataset
        
        # load_dataset("amydeng2000/CREAK")
        elif dataset_name in ['amydeng2000/CREAK', 'CREAK']:
            full_dataset = load_dataset("amydeng2000/CREAK")
            dataset = concatenate_datasets([full_dataset[split] for split in ['train', 'validation']])
            dataset = dataset.select(range(3,len(dataset)))
            def transform_fn(batch):
                return {
                    'prompt': batch['sentence'],
                    'options': [[]] * len(batch['sentence']),
                    'info': [''] * len(batch['sentence']),
                    'label': batch['label']
                }
            dataset = dataset.map(transform_fn, batched=True)
            # processed_dataset = dataset.filter(lambda x: not x['label'] or x['label'] == 'false')
            # return processed_dataset

        # fever/fever
        elif dataset_name in ['fever', 'fever/fever']:
            full_dataset = load_dataset("fever/fever", 'v1.0', trust_remote_code=True)
            dataset = concatenate_datasets([full_dataset[split] for split in ['train']])
            dataset = dataset.select(range(10, 20010))
            
            def transform_fn(batch):
                # Convert FEVER labels to binary
                # SUPPORTS -> True (1)
                # REFUTES or NOT ENOUGH INFO -> False (0)
                binary_labels = [1 if label == "SUPPORTS" else 0 for label in batch['label']]
                
                return {
                    'prompt': batch['claim'],  # FEVER uses 'claim' instead of 'sentence'
                    'options': [[]] * len(batch['claim']),
                    'info': [''] * len(batch['claim']),
                    'label': binary_labels
                }
            dataset = dataset.map(transform_fn, batched=True)

        else:
            raise NotImplementedError
        
        # remove other labels except ['prompt', 'options', 'info', 'label']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['prompt', 'options', 'info', 'label']])

        return dataset
    

def pd_to_json(df: pd.DataFrame, filepath: Union[str, Path], compress: bool = False) -> None:
    """
    Save a pandas DataFrame to JSON with proper type conversion.
    
    Args:
        df: pandas DataFrame to save
        filepath: Path to save the JSON file
        compress: Whether to compress the output by removing indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert DataFrame to dict and handle special types
    data = df.to_dict(orient='records')
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=None if compress else 2)


def json_to_pd(filepath: Path) -> pd.DataFrame:
    """
    Load JSON file back into DataFrame, handling special characters and nested structures.
    
    Args:
        filepath: Path to the JSON file (can be .json or .json.gz)
    
    Returns:
        DataFrame with properly restored data
    """
    def process_record(record):
        """Process a single record when loading from JSON."""
        processed = {}
        for key, value in record.items():
            if isinstance(value, str):
                # Restore special characters
                value = value.replace('[n]', '\n').replace('[r]', '\r').replace('[t]', '\t')
                # Try to parse JSON strings for nested structures
                try:
                    if value.startswith('{') or value.startswith('['):
                        value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            processed[key] = value
        return processed

    # Handle both compressed and uncompressed files
    if str(filepath).endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    processed_data = [process_record(record) for record in data]
    return pd.DataFrame(processed_data)


def truncate_to_last_sentence(text, max_len=150):
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_punctuation = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_punctuation != -1:
        return truncated[:last_punctuation + 1]
    return truncated


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert an object and its nested contents to JSON-serializable types.
    Handles numpy/pandas integer types, numpy arrays, and other special cases.
    
    Args:
        obj: Any Python object to be converted
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.int_, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif pd.isna(obj):  # Handles np.nan, pd.NA, etc.
        return None
    return obj


def save_json(data: Union[List, Dict], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file with proper type conversion.
    
    Args:
        data: Data to save (list or dict)
        filepath: Path to save the JSON file
        indent: Number of spaces for indentation in the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=indent)