import json
import sys
import datasets
import random
import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import csv
import gzip
import requests
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class KnowledgeDataset():

    def __init__(self, model_name: str, path_to_knowledge_dataset: str = "datasets/", dataset_name: str = "triviaqa"):
        set_seed(42)
        torch.manual_seed(42)
        MODEL_NAME = model_name
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        self.model.eval()
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.eos_token
        self.dataset_name = dataset_name
        if self.dataset_name == "triviaqa":
            initial_dataset = self.create_initial_dataset_for_trivia_qa()
        else:  # natural questions
            initial_dataset = self.create_initial_dataset_for_natural_questions()
        self.create_knowledge_dataset(initial_dataset, path_to_knowledge_dataset)

    def create_initial_dataset_for_trivia_qa(self):
        """
        create the initial dataset for the hallucination detection task for triviaqa
        :param path:
        :return:
        """
        print("creating initial dataset for trivia qa")
        # dataset
        dataset = datasets.load_dataset("trivia_qa", 'rc', ignore_verifications=True)
        train, validation, test = dataset["train"], dataset["validation"], dataset["test"]
        dataset = train
        print(f"the length of the dataset is {len(dataset)}")
        random.seed(42)

        data = []
        for i, row in enumerate(dataset):
            prompt = "question: " + row["question"] + "\nanswer:"
            old_target = row["answer"]["value"]
            old_target = old_target
            if old_target.isupper() and len(
                    old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [char.isdigit()
                                                                                                         for char in
                                                                                                         old_target]:
                old_target = old_target[0] + old_target[1:].lower()
            old_token = self.tok(old_target)["input_ids"][
                        1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                               or "mistral" in self.model_name else \
                self.tok(old_target)["input_ids"]
            if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
                old_token = self.tok(" " + old_target)["input_ids"][1:]
            if len(
                    old_token) > 5 or prompt in [e[0] for e in data]:
                continue
            if i < 10:
                print([prompt, old_target, old_token])
            data.append([prompt, old_target, old_token])

        # randomly select 100k examples
        data = random.sample(data, min(70000, len(data)))
        print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def create_initial_dataset_for_natural_questions(self):
        """
        create the initial dataset for the hallucination detection task for natural questions
        :param path:
        :return:
        """
        print("creating initial dataset for natural questions")
        NQ_URL = "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"
        response = requests.get(NQ_URL)
        response.raise_for_status()
        dataset = response.content
        data = []
        number_of_examples = 0
        with gzip.GzipFile(fileobj=BytesIO(dataset)) as read_file:
            for line in read_file:
                json_line = json.loads(line.decode('utf-8'))
                question = json_line["question_text"]
                prompt = "question: " + question + "?\nanswer:"
                short_answers = []

                # Extract short answers (if any exist)
                if "annotations" in json_line and len(json_line["annotations"]) > 0:
                    short_answers_pre = json_line["annotations"][0]["short_answers"]
                    if len(short_answers_pre) == 1 and short_answers_pre[0]["start_token"] != -1:
                        ss = short_answers_pre[0]["start_token"]
                        se = short_answers_pre[0]["end_token"]
                        short_answer_text = " ".join(json_line["document_text"].split()[ss:se])
                        short_answers.append(short_answer_text)
                if len(short_answers) > 1 or len(short_answers) == 0:
                    continue
                number_of_examples += 1
                old_target = short_answers[0]
                if old_target.isupper() and len(
                        old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [
                    char.isdigit()
                    for char in
                    old_target]:
                    old_target = old_target[0] + old_target[1:].lower()
                old_token = self.tok(old_target)["input_ids"][
                            1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                                   or "mistral" in self.model_name else \
                    self.tok(old_target)["input_ids"]
                if len(
                        old_token) > 5 or prompt in [e[0] for e in data]:
                    # print("too long")
                    continue
                if number_of_examples < 10:
                    print([prompt, old_target, old_token])
                data.append([prompt, old_target, old_token])

            # randomly select 100k examples
            data = random.sample(data, min(70000, len(data)))
            print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def create_knowledge_dataset(self, initial_dataset, path_to_save):
        knowledge_dataset = []
        non_knowledge_dataset = []
        else_dataset = []
        self.list_good_shot = [
            "question: What is the capital of France?\nanswer: Paris\n",
            "question: How many continents are there?\nanswer: 7\n",
            "question: Who wrote 'Romeo and Juliet'?\nanswer: William Shakespeare\n",
            "question: What is the square root of 64?\nanswer: 8\n",
            "question: Which element has the chemical symbol 'H'?\nanswer: Hydrogen\n",
            "question: Who was the first President of the United States?\nanswer: George Washington\n",
            "question: What is the powerhouse of the cell?\nanswer: Mitochondria\n",
            "question: In what year did World War II end?\nanswer: 1945\n",
            "question: What is the currency of Japan?\nanswer: Japanese Yen\n",
            "question: Who painted the Mona Lisa?\nanswer: Leonardo da Vinci\n",
            "question: What is the speed of light?\nanswer: 299,792 kilometers per second\n",
            "question: How many sides does a hexagon have?\nanswer: 6\n",
            "question: What is the boiling point of water in Celsius?\nanswer: 100 degrees\n",
            "question: Who wrote 'To Kill a Mockingbird'?\nanswer: Harper Lee\n",
            "question: What is the capital of Australia?\nanswer: Canberra\n",
            "question: What is the largest ocean on Earth?\nanswer: Pacific Ocean\n",
            "question: Who discovered penicillin?\nanswer: Alexander Fleming\n",
            "question: What is the chemical symbol for gold?\nanswer: Au\n",
            "question: What is the smallest prime number?\nanswer: 2\n",
            "question: How many planets are there in our solar system?\nanswer: 8\n"]
        index = 0
        for point in initial_dataset:
            index += 1
            if index % 1000 == 0:
                print(f"finished {index} examples")
                print(f"knowledge dataset has {len(knowledge_dataset)} examples")
                print(f"non knowledge dataset has {len(non_knowledge_dataset)} examples")
                self.save_data(knowledge_dataset,
                               path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}_knowledge_dataset.json")
                self.save_data(non_knowledge_dataset,
                               path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}__non_knowledge_dataset.json")
                self.save_data(else_dataset,
                               path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}__else_dataset.json")
            prompt, old_target, old_token = point
            index_of_shots = random.sample(range(len(self.list_good_shot)), 3)
            good_shots = self.list_good_shot[index_of_shots[0]] + self.list_good_shot[index_of_shots[1]] + \
                         self.list_good_shot[index_of_shots[2]]
            temp_generation = self.batch_generation_with_temperature(self.model, good_shots + prompt, temperature=0.5)
            # temp_generation = [temp_generation[i][len(good_shots+prompt):] for i in range(len(temp_generation))]
            greedy_generation = self.greedy_generation(self.model, good_shots + prompt, length=10)
            temp_generation.append(greedy_generation)
            assert len(temp_generation) == 6
            know_answer = False
            count_know = 0
            for temp in temp_generation:
                if old_target.strip().lower() in temp.lower().strip() or old_target.lower().strip() in temp.lower().strip():
                    count_know += 1
            if count_know == 6:
                know_answer = True
            if know_answer:
                knowledge_dataset.append([prompt, old_target, old_token, count_know])
            if count_know == 0:
                non_knowledge_dataset.append([prompt, old_target, old_token, count_know])
            else:
                else_dataset.append([prompt, old_target, old_token, count_know])
            print(f"{count_know=} {greedy_generation=} {temp_generation=} {old_target=}")
        print(f"knowledge dataset has {len(knowledge_dataset)} examples")
        print(f"non knowledge dataset has {len(non_knowledge_dataset)} examples")
        self.save_data(knowledge_dataset,
                       path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}_knowledge_dataset.json")
        self.save_data(non_knowledge_dataset,
                       path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}__non_knowledge_dataset.json")
        self.save_data(else_dataset,
                       path_to_save + f"{self.model_name.replace('/', '_')}_{self.dataset_name}__else_dataset.json")

    def save_data(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f)

    def batch_generation_with_temperature(self, model, prompt, temperature=0.5):
        """
        generate 5 examples with the same prompt and return the generated texts
        :param model:
        :param prompt:
        :param temperature:
        :return:
        """
        if "Instruct" in self.model_name or "-it" in self.model_name:
            split_prompt = [x.strip() for x in prompt.split("\n") if x.strip() != ""]
            split_prompt = split_prompt[:-1]
            messages = [
                {"role": "assistant", "content": x.replace('answer: ', 'The answer is ') + "\n"} if i % 2 == 1 else {
                    "role": "user", "content": x.replace('question: ', '') + "\n"} for i, x in enumerate(split_prompt)]

            messages += [{"role": "assistant", "content": " The answer is "}]
            unwanted_tokens_at_the_end = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n",
                                          "<end_of_turn>", "<start_of_turn>", "model", " ", "\n\n", "</s>"]
            unwanted_tokens_embedded = self.tok(unwanted_tokens_at_the_end)["input_ids"]
            unwanted_tokens_embedded = [x for y in unwanted_tokens_embedded for x in y]
            unwanted_tokens_embedded = list(set(unwanted_tokens_embedded))
            print(f"{messages=}")
            input_ids = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            while input_ids[0][-1] in unwanted_tokens_embedded:
                input_ids = input_ids[:, :-1]
            terminators = [
                self.tok.eos_token_id,
                self.tok.convert_tokens_to_ids("<|eot_id|>")
            ]
            generated = []
            for i in range(5):
                with torch.no_grad():
                    response = model.generate(input_ids, max_length=(len(input_ids[0]) + 10), do_sample=True,
                                              pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature,
                                              eos_token_id=terminators)
                generated.append(self.tok.decode(response[0], skip_special_tokens=True))
        else:
            input_ids = \
                self.tok([prompt for i in range(5)], padding=True, return_token_type_ids=False, return_tensors="pt")[
                    "input_ids"].to(device)
            # run the same prompt 5 times in a batch
            with torch.no_grad():
                model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + 10), do_sample=True,
                                           pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature)
            generated = self.tok.batch_decode(model_out, skip_special_tokens=True)
        print(f"generated with temp={generated}")
        return generated

    def greedy_generation(self, model, prompt, length=10):
        if "Instruct" in self.model_name or "-it" in self.model_name:
            split_prompt = [x.strip() for x in prompt.split("\n") if x.strip() != ""]
            split_prompt = split_prompt[:-1]
            messages = [
                {"role": "assistant", "content": x.replace('answer: ', 'The answer is ') + "\n"} if i % 2 == 1 else {
                    "role": "user", "content": x.replace('question: ', '') + "\n"} for i, x in enumerate(split_prompt)]

            messages += [{"role": "assistant", "content": " The answer is "}]
            unwanted_tokens_at_the_end = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n",
                                          "<end_of_turn>", "<start_of_turn>", "model", " ", "\n\n", "</s>"]
            unwanted_tokens_embedded = self.tok(unwanted_tokens_at_the_end)["input_ids"]
            unwanted_tokens_embedded = [x for y in unwanted_tokens_embedded for x in y]
            unwanted_tokens_embedded = list(set(unwanted_tokens_embedded))

            input_ids = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            while input_ids[0][-1] in unwanted_tokens_embedded:
                input_ids = input_ids[:, :-1]
            terminators = [
                self.tok.eos_token_id,
                self.tok.convert_tokens_to_ids("<|eot_id|>")
            ]
            with torch.no_grad():
                response = model.generate(input_ids, max_length=(len(input_ids[0]) + length), do_sample=False,
                                          pad_token_id=self.tok.eos_token_id, num_beams=1, temperature=None, top_p=None,
                                          eos_token_id=terminators)
            generated = self.tok.decode(response[0], skip_special_tokens=True)


        else:
            input_ids = \
                self.tok(prompt, padding=True, return_token_type_ids=False, return_tensors="pt")[
                    "input_ids"].to(device)
            with torch.no_grad():
                model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + length), do_sample=False,
                                           pad_token_id=self.tok.eos_token_id, num_beams=1, top_p=None,
                                           temperature=None,
                                           attention_mask=torch.ones_like(input_ids))
            # only new generated tokens
            generated = self.tok.decode(model_out[0], skip_special_tokens=True)
        return generated
