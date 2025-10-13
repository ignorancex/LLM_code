from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import random
from collections import Counter

class GlobalOrderingRunner():
    def __init__(self, args):
        self.args = args
        self.duplicate_cnt = 0
        self.init_setup()

    def init_setup(self):
        if self.args.sorting_method == 'monot5':
            self.model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-msmarco-10k').to('cuda')
            self.tok = T5Tokenizer.from_pretrained('castorini/monot5-base-msmarco-10k')
        elif self.args.sorting_method == 'lexical':
            pass

    def run_global_sort(self, prefix, parallel_rows, suffix, tokenized_input):
        if self.args.sorting_method == 'monot5':
            return self.run_global_sort_monot5(prefix, parallel_rows, suffix, tokenized_input)
        elif self.args.sorting_method == 'lexical':
            return self.run_global_sort_lexical(prefix, parallel_rows, suffix, tokenized_input)
        elif self.args.sorting_method == 'freq':
            return self.run_global_sort_freqwithpos(prefix, parallel_rows, suffix, tokenized_input)

    def hierarchical_sort(self, tensor):
        rows, cols = tensor.shape
        indices = torch.arange(rows, device=tensor.device)
        for col in reversed(range(cols)):
            _, sorted_indices = torch.sort(tensor[indices, col], stable=True)
            indices = indices[sorted_indices]
        return indices.tolist()

    def calculate_weighted_frequency_score(self, sequence, freq_dict):
        # Assign higher weights to tokens in earlier positions
        return sum((1 / freq_dict[token]) * (1 / (i + 1)) for i, token in enumerate(sequence))

    def run_global_sort_freqwithpos(self, prefix, parallel_rows, suffix, tokenized_input):
        input_ids = tokenized_input['input_ids']
        zero_index_tokid = input_ids[0][0].item() # 128000 for Llama, 151644 for Qwen
        input_ids_doc = torch.stack([input_ids[0].index_select(0, tokenized_input['doc_range'][i]) for i in range(tokenized_input['doc_range'].shape[0])])
        input_ids_doc = input_ids_doc.tolist()
        all_tokens = [zero_index_tokid] + [token for seq in input_ids_doc for token in seq]
        token_frequencies = Counter(all_tokens)
        zero_value = 1/token_frequencies[zero_index_tokid]
        freq_converted_ids = torch.tensor([[1/token_frequencies[x] for x in y] for y in input_ids_doc])
        freq_converted_ids[freq_converted_ids==zero_value] = 0
        return self.hierarchical_sort(freq_converted_ids)

    def run_global_sort_lexical(self, prefix, parallel_rows, suffix, tokenized_input):
        kwargs = tokenized_input
        input_ids = kwargs['input_ids']
        zero_index_tokid = input_ids[0][0].item() # 128000 for Llama, 151644 for Qwen
        input_ids_doc = [input_ids[0][y] for y in kwargs['doc_range']]
        rotor_input_uids = torch.stack([input_ids[0].index_select(0, kwargs['doc_range'][i]) for i in range(kwargs['doc_range'].shape[0])])
        # 0 means padding, so replace it
        zero_index_tokid = input_ids[0][0].item() # 128000 for Llama, 151644 for Qwen
        rotor_input_uids[rotor_input_uids==zero_index_tokid] = 0
        return self.hierarchical_sort(rotor_input_uids)

    def run_global_sort_monot5(self, prefix, parallel_rows, suffix, tokenized_input):
        # split suffix (or prefix) into question
        if self.args.data == 'mmlu':
            question = prefix.strip().split('\n')[-1]
        else:
            question = suffix.split('<|eot_id|>')[0].replace('\n','').replace('Question: ','').replace('question: ','')
            question = question.replace('dialog: ','').replace('statement: ', '').strip()

        if self.args.data == 'lostinthemiddle':
            question = question.replace('Answer:', '').strip()
        text = [f'Query: {question} Document: {x} Relevant:' for x in parallel_rows]
        input_tensors = self.tok(text, return_tensors='pt',
                padding='max_length', max_length=1024,
                truncation=True).to('cuda')
        output = self.model.generate(**input_tensors, max_length=2,
                return_dict_in_generate=True, output_scores=True)
        scores = torch.stack(output.scores)
        yesno_softmax_scores = torch.nn.functional.log_softmax(scores[0][:, [1176, 6136]], dim=1)[:, 0] # true, false
        rank = torch.argsort(yesno_softmax_scores).tolist()
        rank.reverse()
        return rank
