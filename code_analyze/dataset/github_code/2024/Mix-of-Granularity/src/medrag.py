import os
import re
import json
import tqdm
import torch
import time
import argparse
import requests
import transformers
from transformers import AutoTokenizer, AutoModel
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, parent_folder_path)
from utils import RetrievalSystem, build_snippet_score_dict, weighted_merge_snippets
from template import *

from typing import List
from config import config
from openai import OpenAI
from serve_test import answer_internlm

llama3_path = "MoG/llama3"
sys.path.insert(0, llama3_path)
from llama3.llama import Dialog, Tokenizer

if openai.api_key is None:
    from config import config

    openai_api_type = config["openai_api_type"]
    openai_api_base = config["openai_api_base"]
    openai_api_version = config["openai_api_version"]
    openai_api_key = config["openai_api_key"]

import os

tiktoken_cache_dir = "***"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# validate
assert os.path.exists(
    os.path.join(tiktoken_cache_dir, "***")
)


class MedRAG:

    def __init__(
        self,
        llm_name="OpenAI/gpt-3.5-turbo-16k",
        rag=True,
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir="../corpus",
        cache_dir=None,
        llm_local=None,
        api_retry=30,
        api_retry_delay=1,
        rrf_k=100,
        pred_with_router=False,
        router_model=None,
        only_retrieve=False,
        router_merge_k=None,
        threshold=0,
    ):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.api_retry = api_retry
        self.api_retry_delay = api_retry_delay
        self.rrf_k = rrf_k
        self.router_merge_k = router_merge_k
        self.pred_with_router = pred_with_router
        self.router_model = router_model
        self.only_retrieve = only_retrieve
        self.threshold = threshold
        print("[In progress] Initializing MedRAG instance.")
        if rag:
            self.retrieval_system = RetrievalSystem(
                self.cache_dir,
                self.retriever_name,
                self.corpus_name,
                self.db_dir,
                self.pred_with_router,
                self.router_model,
                threshold=self.threshold,
            )
        else:
            self.retrieval_system = None

        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag,
            "one_shot_context": one_shot_context,
            "one_shot_response": one_shot_response,
            "one_shot_rag_context": one_shot_rag_context,
            "one_shot_rag_response": one_shot_rag_response,
        }
        print("Prompt templates loaded")

        if self.llm_name.split("/")[0].lower() == "openai":
            self.model = self.llm_name.split("/")[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "internlm" in self.llm_name.lower():
            self.model = self.llm_name.split("/")[-1]
            self.max_length = 4096
            self.context_length = 3072
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["intern_model_path"], trust_remote_code=True
            )
            print("Tokenizer loaded")
        elif "glm" in self.llm_name.lower():
            self.model = llm_local
            self.max_length = 4096
            self.context_length = 3072
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["glm_tokenizer_path"], trust_remote_code=True
            )
            print("Tokenizer loaded")
        elif "llama3" in self.llm_name.lower():
            self.model = self.llm_name
            self.max_length = 4096
            self.context_length = 3072
            self.tokenizer = Tokenizer(config["llama_tokenizer_path"])
            print("Tokenizer loaded")
        elif "qwen_moe" in self.llm_name.lower():
            self.model = self.llm_name
            self.max_length = 4096
            self.context_length = 3072
            self.tokenizer = AutoTokenizer.from_pretrained(config["qwen_moe_path"])
            print("Tokenizer loaded")
        else:
            print(
                "For other LLMs, the tokenizer will be loaded from the HuggingFace model hub"
            )
            print(f"Loading tokenizer: {self.llm_name}. Cache_dir: {self.cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name, cache_dir=self.cache_dir, trust_remote_code=True
            )
            print("Tokenizer loaded")
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/mistral-instruct.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/meditron.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/pmc_llama.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 2048
                self.context_length = 1024

            print(f"Loading model: {self.llm_name}. Cache_dir: {self.cache_dir}")
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device="cuda:0",
                model_kwargs={"cache_dir": self.cache_dir},
                trust_remote_code=True,
            )
            print("Model loaded.")

        print("[Done] MedRAG instance initialized successfully.\n")

    def answer(
        self, question, options=None, k=32, rrf_k=100, save_dir=None, threshold=0
    ):
        """
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        save_dir (str): directory to save the results
        """
        router = self.router_model

        if options is not None:
            options = "\n".join(
                [key + ". " + options[key] for key in sorted(options.keys())]
            )
        else:
            options = ""  # double check this later!!!!!! See if new prompt tempates are needed.
        # retrieve relevant snippets
        fall_back_cot_flag = True
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                question,
                k=k,
                rrf_k=self.rrf_k,
                threshold=threshold,
            )
            if self.only_retrieve:
                # skip the rest of answer()
                # return the contexts
                return retrieved_snippets, scores
            for retrieval_results_of_one_source in retrieved_snippets[0]:
                if retrieval_results_of_one_source[0] != "NO_TEXT_RETRIEVED":
                    fall_back_cot_flag = False
                    break

            if fall_back_cot_flag:
                contexts = []

            else:
                if self.pred_with_router:
                    # calculate the weights
                    weights = router.run(question)
                    snippet_score_list_dict, snippet_score_dict = (
                        build_snippet_score_dict(
                            retrieved_snippets, scores, weights, router.device
                        )
                    )
                    snippet_merged = weighted_merge_snippets(
                        retrieved_snippets,
                        scores,
                        snippet_score_list_dict,
                        snippet_score_dict,
                        self.router_merge_k,
                        weights,
                        return_separate_list=True,
                    )
                    retrieved_snippets = [[snippet_merged]]

                # combine all the retrieved snippets from differnet retrievers
                _retrieved_snippets = []
                for retriever_result in retrieved_snippets[0]:
                    _retrieved_snippets.extend(retriever_result)
                contexts = [
                    "Document [{:d}] (Title: {:s}) {:s}".format(
                        idx,
                        _retrieved_snippets[idx]["title"],
                        _retrieved_snippets[idx]["content"],
                    )
                    for idx in range(len(_retrieved_snippets))
                ]
            if len(contexts) == 0:
                contexts = [""]

            if "openai" in self.llm_name.lower():
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode("\n".join(contexts))[
                            : self.context_length
                        ]
                    )
                ]
            elif "llama3" in self.llm_name.lower():
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode("\n".join(contexts), bos=True, eos=False)[
                            : self.context_length
                        ]
                    )
                ]
            else:
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode(
                            "\n".join(contexts), add_special_tokens=False
                        )[: self.context_length]
                    )
                ]

        else:
            retrieved_snippets = []
            scores = []
            contexts = []
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(
                question=question, options=options
            )
            # if "internlm" in self.llm_name.lower() or "glm" in self.llm_name.lower():
            if "internlm" in self.llm_name.lower():
                messages = (
                    self.templates["cot_system"]
                    + "\n"
                    + self.templates["one_shot_context"]
                    + "\n"
                    + self.templates["one_shot_response"]
                    + "\n"
                    + prompt_cot
                )
            else:
                messages = [
                    {"role": "system", "content": self.templates["cot_system"]},
                    {"role": "user", "content": self.templates["one_shot_context"]},
                    {
                        "role": "assistant",
                        "content": self.templates["one_shot_response"],
                    },
                    {"role": "user", "content": prompt_cot},
                ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(
                    context=context, question=question, options=options
                )
                # if "internlm" in self.llm_name.lower() or "glm" in self.llm_name.lower():
                if "internlm" in self.llm_name.lower():
                    messages = (
                        self.templates["medrag_system"]
                        + "\n"
                        + self.templates["one_shot_rag_context"]
                        + "\n"
                        + self.templates["one_shot_rag_response"]
                        + "\n"
                        + prompt_medrag
                    )
                else:
                    messages = [
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": self.templates["one_shot_rag_context"]},
                        {
                            "role": "assistant",
                            "content": self.templates["one_shot_rag_response"],
                        },
                        {"role": "user", "content": prompt_medrag},
                    ]
                ans = self.generate(messages)
                answers.append(re.sub("\s+", " ", ans))

        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump({"messages": messages, "answers": answers}, f, indent=4)

        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    # @staticmethod
    def custom_stop(self, stop_str, input_len=0):
        class CustomStoppingCriteria(StoppingCriteria):
            def __init__(self, stop_words, tokenizer, input_len=0):
                super().__init__()
                self.tokenizer = tokenizer
                self.stop_words = stop_words
                self.input_len = input_len

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                tokens = self.tokenizer.decode(input_ids[0][self.input_len :])
                return any(stop in tokens for stop in self.stop_words)

        stopping_criteria = StoppingCriteriaList(
            [CustomStoppingCriteria(stop_str, self.tokenizer, input_len)]
        )
        return stopping_criteria

    def generate_internlm(self, prompt):
        # Retrieve the generated text from the API response
        for retry_attempt in range(50):
            try:
                ans = answer_internlm(prompt)
                break
            except Exception as e:
                if retry_attempt == self.api_retry - 1:
                    print("Failed after all retries.")
                    raise e
                time.sleep(self.api_retry_delay)

        return ans

    def generate_glm(self, prompt):
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        openai_api_key = "EMPTY"
        openai_api_base = config["glm_api_base"]

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        chat_response = client.chat.completions.create(
            model="glm",
            messages=prompt,
        )
        ans = chat_response.choices[0].message.content
        while "\n" in ans:
            ans = ans.replace("\n", "")
        return ans

    def generate_qwen_moe(self, prompt):
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        openai_api_key = "EMPTY"
        openai_api_base = config["qwen_moe_api_base"]

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        chat_response = client.chat.completions.create(
            model="qwen_moe",
            messages=prompt,
        )
        ans = chat_response.choices[0].message.content
        while "\n" in ans:
            ans = ans.replace("\n", "")
        return ans

    def generate_llama3(self, prompt):
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        openai_api_key = "EMPTY"
        openai_api_base = config["llama3_api_base"]

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        chat_response = client.chat.completions.create(
            model="llama3",
            messages=prompt,
        )
        ans = chat_response.choices[0].message.content
        while "\n" in ans:
            ans = ans.replace("\n", "")
        return ans

    def generate(self, messages):
        """
        generate response given messages
        """
        if "openai" in self.llm_name.lower():
            os.environ["http_proxy"] = "***"
            os.environ["https_proxy"] = "***"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            if openai.api_type == "azure":
                for retry_attempt in range(self.api_retry):
                    try:
                        response = openai.ChatCompletion.create(
                            engine=self.model,
                            messages=messages,
                            temperature=0.0,
                        )
                        break
                    except Exception as e:
                        print("Error: ", e)
                        if retry_attempt == self.api_retry - 1:
                            raise e
                        time.sleep(self.api_retry_delay)
            else:
                # for the moment, the implementation of whether it is azure or not is the same
                for retry_attempt in range(self.api_retry):
                    try:
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                        )
                        break
                    except Exception as e:
                        if retry_attempt == self.api_retry - 1:
                            print("Failed after all retries.")
                            raise e
                        time.sleep(self.api_retry_delay)

            ans = (
                response.choices[0].message.content
                if response is not None
                else "No response from the model"
            )
        elif "internlm" in self.llm_name.lower():
            # call internlm to get response via API
            for retry_attempt in range(self.api_retry):
                try:
                    ans = self.generate_internlm(messages)
                    break
                except Exception as e:
                    print("Error in generating response: ", e)
                    if retry_attempt == self.api_retry - 1:
                        raise e
                    time.sleep(self.api_retry_delay)

        elif "glm" in self.llm_name.lower():
            ans = self.generate_glm(messages)
        elif "qwen_moe" in self.llm_name.lower():
            ans = self.generate_qwen_moe(messages)
        elif "llama3" in self.llm_name.lower():
            ans = self.generate_llama3(messages)

        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if "meditron" in self.llm_name.lower():
                stopping_criteria = self.custom_stop(
                    ["###", "User:", "\n\n\n"],
                    input_len=len(
                        self.tokenizer.encode(messages, add_special_tokens=True)
                    ),
                )

            response = self.model(
                prompt,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_length,
                truncation=True,
                stopping_criteria=stopping_criteria,
            )
            ans = response[0]["generated_text"]
        return ans
