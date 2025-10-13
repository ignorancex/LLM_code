import os
import copy
import random

from functools import partial
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

DATA_DIR = os.getenv("DATASET_DIR")


class BaseDocument:
    """ """

    def __init__(self):
        """ """
        raise NotImplementedError()

    def process_data(self):
        """
        Args:
        Returns:
        Raises:
        """
        pass

    def load_data(self):
        """
        Args:
        Returns:
        Raises:
        """
        pass

    def make_instructions(self):
        """
        Args:
        Returns:
        Raises:
        """
        pass

    def shuffle_items(self):
        """
        Args:
        Returns:
        Raises:
        """
        pass


class SynthBioDoc(BaseDocument):
    def __init__(self, **configs):
        self.sentence_perplex = False
        self.set_variables(configs)
        self.default_prompt = "Please describe about a person, {title}."
        self.set_prompts()
        self.key_title = "name"
        self.key_content = "data_list"
        # self.num_shuffle = 100

    def set_variables(self, config):
        self.no_answer = False
        for key, value in config.items():
            setattr(self, key, value)
        self.filename = os.path.join(DATA_DIR, self.filename)

    def set_prompts(self):
        self.set_start_prompt()
        main_prompt = [
            " a person, {title}.",
            " the early life of {title}.",
            " the background of {title}.",
            " {title}'s entry into the professional world.",
        ]
        self.title_prompts = []
        for prompt in main_prompt:
            self.title_prompts += [sp + prompt for sp in self.start_prompt]
        self.title_prompts = [self.default_prompt] + self.title_prompts

    def set_start_prompt(self):
        prepend = [
            "Please",
            "Could you",
            "Could you please",
            "Would you",
            "Would you please",
        ]
        start_prompt = [
            "Describe",
            "Explore",
            "Discuss",
            "Summarize",
            "Delve into",
            "Talk about",
            "Tell me",
            "Show me",
        ]
        new_prompts = []
        for prep in prepend:
            new_prompts += [prep + " " + sp.lower() for sp in start_prompt]
        start_prompt += ["I want to know about"]
        start_prompt += ["I need to know about"]
        self.start_prompt = start_prompt + new_prompts

    def init_dataset(self):
        data = self.load_dataset()
        if self.shuffle:
            for i in range(self.shuffle):
                dataset_shl = data.map(
                    self.process_dataset, batched=True, load_from_cache_file=False
                )
                if i == 0:
                    self.dataset = data.map(
                        self.process_dataset, batched=True, load_from_cache_file=False
                    )
                else:
                    self.dataset["train"] = concatenate_datasets(
                        [self.dataset["train"], dataset_shl["train"]]
                    )
        else:            
            self.dataset = data.map(
                self.process_dataset, batched=True, load_from_cache_file=False
            )
        print(len(self.dataset))

    def load_dataset(self):
        return load_dataset("json", data_files=self.filename, cache_dir=DATA_DIR)  
    
    def process_dataset(self, examples):
        responses = examples[self.key_content]
        queries = self.make_instructions(examples)
        inputs_queries = self.tokenizer(
            queries,
        )["input_ids"]
        responses = self.shuffle_items(responses)
        inputs_answers = self.tokenizer(
            responses,
        )["input_ids"]
        inputs = {}
        inputs["answer"] = responses
        inputs["query"] = queries
        new_inputs = []
        new_mask = []
        length_query = []
        length_seq = []

        for inputs_query, inputs_answer in zip(inputs_queries, inputs_answers):
            input_new = inputs_query + inputs_answer[1:]
            length_query.append(len(inputs_query))
            length_seq.append(len(input_new))
            attention_mask = [1] * len(input_new) + [0] * (
                self.max_seq_len - len(input_new)
            )
            input_new += [self.tokenizer.pad_token_id] * (
                self.max_seq_len - len(input_new)
            )
            new_inputs.append(input_new)
            new_mask.append(attention_mask)

        inputs["input_ids"] = new_inputs
        inputs["input_queries"] = inputs_queries
        inputs["attention_mask"] = new_mask
        labels = copy.deepcopy(new_inputs)

        new_labels = []
        for label, q_length, s_length in zip(labels, length_query, length_seq):
            if self.sentence_perplex:
                # when evaluating perplexity per sentence,
                # we should avoid computing perplexity
                # on the last token of each sentence.
                s_length -= 2
                #q_length += 4
            label[:q_length] = [-100] * q_length
            label[s_length + 1 :] = [-100] * len(label[s_length + 1 :])
            new_labels.append(label)
        inputs["end_question"] = length_query
        inputs["length_seq"] = length_seq
        if self.no_answer:
            inputs["input_ids"] = inputs["input_queries"]
            inputs["answer"] = inputs_answers
        inputs["labels"] = [inst[: self.max_seq_len] for inst in new_labels]
        inputs["input_ids"] = [inst[: self.max_seq_len] for inst in inputs["input_ids"]]
        inputs["attention_mask"] = [
            inst[: self.max_seq_len] for inst in inputs["attention_mask"]
        ]
        return inputs

    def make_instructions(self, examples):
        titles = examples[self.key_title]
        new_queries = []
        prompt = self.title_prompts[0]
        for title in titles:
            if self.inst_mode:
                instruction = prompt.format(title=title)
                ## This is used in llama.
                query = "[INST] " + instruction + " [/INST]"
                new_queries.append(query)
            elif self.sentence_perplex:
                new_queries.append("")
            else:
                ## default prompting.
                new_queries.append(title + " :")

        return new_queries

    def shuffle_items(self, sentences):
        ## sentences are the list of sentences.
        new_sents = []
        for sentence in sentences:
            sentence[-1] = sentence[-1] + "\n"
            if self.shuffle:
                random.shuffle(sentence)
            new_sents.append(" ".join(sentence))
        return new_sents

class SynthBioPerplex(SynthBioDoc):
    def __init__(self, **configs):
        super().__init__(**configs)

    def make_instructions(self, examples):
        titles = examples["question"]
        new_queries = []
        for title in titles:
            ## default prompting.
            new_queries.append(title)
        return new_queries

    
class Wiki2023(SynthBioDoc):
    def __init__(self, **configs):
        self.set_variables(configs)
        self.title_prompts = ["Please describe about {title}."]
        self.key_title = "title"
        self.key_content = "text"
        # self.num_shuffle = 100


class Wiki2023_EVAL(SynthBioDoc):
    def __init__(self, **configs):
        print(configs)
        self.set_variables(configs)
        self.title_prompts = ["Please describe about {title}."]
        self.key_title = "title"
        self.key_content = "text"
        # self.num_shuffle = 100

    def process_dataset(self, examples):
        responses = examples[self.key_content]
        queries = self.make_instructions(examples)
        inputs_queries = self.tokenizer(
            queries,
        )["input_ids"]
        responses = self.shuffle_items(responses)
        index_all = []
        new_response = []
        for response in responses:
            indexes = []
            for i, sentence in enumerate(response):
                ids = self.tokenizer(sentence)["input_ids"][1:-1]
                index_list = [i] * len(ids)
                indexes += index_list
            indexes = [0] + indexes + [-1] * (self.max_seq_len - len(indexes) - 1)
            index_all.append(indexes)
            new_response.append(" ".join(response))
        inputs_answers = self.tokenizer(
            new_response,
        )["input_ids"]
        inputs = {}
        inputs["index_list"] = index_all
        inputs["answer"] = new_response
        inputs["query"] = queries
        new_inputs = []
        new_mask = []
        length_query = []
        length_seq = []

        for inputs_query, inputs_answer in zip(inputs_queries, inputs_answers):
            input_new = inputs_query + inputs_answer[1:]
            length_query.append(len(inputs_query))
            length_seq.append(len(input_new))
            attention_mask = [1] * len(input_new) + [0] * (
                self.max_seq_len - len(input_new)
            )
            input_new += [self.tokenizer.pad_token_id] * (
                self.max_seq_len - len(input_new)
            )
            new_inputs.append(input_new)
            new_mask.append(attention_mask)

        inputs["input_ids"] = new_inputs
        inputs["input_queries"] = inputs_queries
        inputs["attention_mask"] = new_mask
        labels = copy.deepcopy(new_inputs)

        new_labels = []
        for label, q_length, s_length in zip(labels, length_query, length_seq):
            label[:q_length] = [-100] * q_length
            label[s_length + 1 :] = [-100] * len(label[s_length + 1 :])
            new_labels.append(label)
        inputs["end_question"] = length_query
        inputs["length_seq"] = length_seq
        if self.no_answer:
            inputs["input_ids"] = inputs["input_queries"]
            inputs["answer"] = inputs_answers
        inputs["labels"] = [inst[: self.max_seq_len] for inst in new_labels]
        inputs["input_ids"] = [inst[: self.max_seq_len] for inst in inputs["input_ids"]]
        inputs["attention_mask"] = [
            inst[: self.max_seq_len] for inst in inputs["attention_mask"]
        ]
        return inputs

    def shuffle_items(self, sentences):
        ## sentences are the list of sentences.
        new_sents = []
        for sentence in sentences:
            sentence[-1] = sentence[-1] + "\n"
            new_sents.append(sentence)
        return new_sents
