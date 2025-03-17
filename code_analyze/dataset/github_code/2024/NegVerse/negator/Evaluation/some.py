import argparse
import os
import torch
import numpy as np

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import InputExample, InputFeatures
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


class Scorer(object):
    def __init__(self, models_path = 'gfm-models', batch_size = 16, save_to_file=False,file_path='scores.txt', **kwargs):

        self.g_dir = f"{models_path}/grammar"
        self.f_dir = f"{models_path}/fluency"

        self.model_g = BertForSequenceClassification.from_pretrained(self.g_dir)
        self.model_f = BertForSequenceClassification.from_pretrained(self.f_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.batch_size = batch_size
        self.save_to_file = save_to_file
        self.file_path = file_path

        # Default values
        self.default_args = {
            'weight_g': 1.0,
            'weight_f': 1.0,
            'model_type': 'bert'
        }
        
        # Override defaults with provided kwargs
        self.args = {**self.default_args, **kwargs}

    def add(self, pred):
        # make dataset for sreg and ssreg
        self.data_sreg = self.create_dataset(pred)

    def create_example(self, pred):
        examples = []
        for i, p in enumerate(pred):
            examples.append(
                InputExample(guid=i, text_a=p, text_b=None, label=None)
            )
        return examples

    def convert_examples_to_features(
        self,
        examples,
        tokenizer,
        max_length=None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample):
            if example.label is None:
                return None
            elif output_mode == 'classification':
                return label_map[example.label]
            elif output_mode == 'regression':
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features

    def create_dataset(self,pred):
        # load examples and convert to features
        examples = self.create_example(pred)
        tokenizer = self.tokenizer
        features = self.convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[None],
            max_length=128,
            output_mode='regression',
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset

    def min_max_normalize(self, x, x_min=1, x_max=4):
        return (x - x_min) / (x_max - x_min)
    
    def save_scores_to_file(self, score_g, score_f):
        # Write to a .txt file
        with open(self.file_path, 'w') as file:
            file.write('grammaticality\tfluency\n')  # Column headers
            for g, f in zip(score_g, score_f):
                file.write(f"{g:.3f}\t{f:.3f}\n")  # Write scores

        print(f"Scores saved to {self.file_path}")

    def score(self):
        # normalize
        score_g = [self.min_max_normalize(x) for x in self.predict(task='grammer')]
        score_f = [self.min_max_normalize(x) for x in self.predict(task='fluency')]

        # Convert lists to NumPy arrays
        score_g = np.array(score_g)
        score_f = np.array(score_f)

        # Get average
        average_score_g = np.mean(score_g)
        average_score_f = np.mean(score_f)

        scores = self.args['weight_g'] * score_g + self.args['weight_f'] * score_f
        total_score = np.mean(scores)

        # Optionally save to file
        if self.save_to_file:
            self.save_scores_to_file(score_g, score_f)

        # Return scores as a dictionary
        return {
            'grammaticality': average_score_g,
            'fluency': average_score_f
            #'total': total_score
        }

    def predict(self, task):
        # Setup CUDA, GPU & distributed training
        device = torch.device('cuda')

        if task == 'grammer':
            model = self.model_g
            pred_dataset = self.data_sreg
        elif task == 'fluency':
            model = self.model_f
            pred_dataset = self.data_sreg

        model.to(device)

        pred_bach_size = self.batch_size
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=pred_bach_size)

        preds = None

        for batch in pred_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}
                if self.args['model_type'] != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None  
                outputs = model(**inputs)
                logits = outputs[:2][0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds = np.squeeze(preds)
        return preds
