import json
import os
from pathlib import Path
from threading import Thread
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from logger import Logger


def load_llm(model_path, tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, device_map="auto",
                                              torch_dtype="auto", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading pretrained model from {model_path}.")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto",
                                                 torch_dtype="auto")
    # print("Used device map:", model.hf_device_map)
    model.eval()
    return model, tokenizer




def prepare_chat_log(prompt: str, initial_system_message: str,
                     chat_log: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """
    Take prompt as first argument, followed by optional chat log
    Args:
    Returns:
    """
    if chat_log is None:
        if initial_system_message is None:
            chat_log = []
        else:
            chat_log = [{'role': 'system', 'content': initial_system_message}]
    chat_log.append({'role': 'user', 'content': prompt})
    return chat_log


def pipeline(model_name: str, model, tokenizer, query: str, chat_log: list[dict[str,str]], initial_system_message: str,
             max_output_tokens: int, temperature:bool = None, do_sample:bool = False, top_p:bool = None, boxed: bool = False) -> tuple[
    str, list[dict[str, str]], str]:
    if "r1_distill" in model_name:
        initial_system_message = None   # recommended according to Usage Recommendations on
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#usage-recommendations and
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        if boxed:
            query += " Put your final answer within \\boxed{}."
    chat_log = prepare_chat_log(query, initial_system_message, chat_log=chat_log)
    inputs = tokenizer.apply_chat_template(chat_log, add_generation_prompt=True, return_tensors="pt").to('cuda')

    if inputs.size(1) > tokenizer.model_max_length:
        answer = "The input sequence is too long. Aborting."
        chat_log.append({'role': 'assistant', 'content': answer})
        return answer, chat_log, answer

    attention_mask = torch.ones(1, inputs.size(1)).to('cuda')

    output = model.generate(
        inputs,
        max_length=inputs.size(1) + max_output_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,  # Handle padding gracefully
        attention_mask=attention_mask,
        do_sample=do_sample,
        top_p=top_p,
    )

    if (output[0][-1] != tokenizer.eos_token_id
            and len(output[0]) - inputs.size(1) == max_output_tokens):
        print(f"WARNING: Max. token length ({max_output_tokens}) exceeded.")

    answer = tokenizer.decode(output[0][inputs.size(1):], skip_special_tokens=True).strip()
    full_answer = answer
    if "r1_distill" in model_name:
        if "r1_distill" in model_name:
            answer = answer.split("</think>")[-1]
            if "boxed{" in answer and answer.rfind("}") != -1:
                answer = answer[:answer.rfind("}")].split("boxed{")[-1]
            answer = answer.split("Answer:**")[-1]
            answer = answer.replace("\_", "_").replace("\'", "'")
            if len(answer.split("**")) == 3:
                answer = answer.split("**")[-2]
        answer = answer.strip()
    chat_log.append({'role': 'assistant', 'content': answer})
    return answer, chat_log, full_answer


def pipeline_batch(model_name: str, model, tokenizer, queries: list[str],
                   system_message: str, max_output_tokens: int, chat_logs: list[list[dict[str, str]]], temperature:bool = None,
                   do_sample:bool = False, top_p:bool = None, boxed: bool = False) -> tuple[
    list[str], list[list[dict[str, str]]], list[str]]:
    if "r1_distill" in model_name:
        # system_message = None   # recommended according to Usage Recommendations on
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#usage-recommendations and
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        if boxed:
            for i in range(len(queries)):
                queries[i] +=  " Put your final answer within \\boxed{}."
    if chat_logs is None:
        chat_logs = [None for _ in range(len(queries))]
    for i in range(len(queries)):
        chat_logs[i] = prepare_chat_log(queries[i], system_message, chat_log=chat_logs[i])

    outputs = []

    prompts = tokenizer.apply_chat_template(chat_logs, add_generation_prompt=True, return_tensors="pt",
                                            padding=True, truncation=False, tokenize=False)
    tokenized_input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to('cuda')

    num_input_tokens = tokenized_input.input_ids.size(1)
    batch_size = int(tokenizer.model_max_length / 2 / num_input_tokens)
    if batch_size < 0 and num_input_tokens <= tokenizer.model_max_length:
        batch_size = 1
    input_batches = torch.split(tokenized_input.input_ids, batch_size)
    attention_batches = torch.split(tokenized_input.attention_mask, batch_size)

    for i in range(len(input_batches)):
        batch_output = model.generate(
            input_batches[i],
            max_length=num_input_tokens + max_output_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_batches[i],
            do_sample=do_sample,
            top_p=top_p
        )
        outputs.extend(batch_output)

    # Decode responses and update chat logs
    for i in range(len(outputs)):
        outputs[i] = outputs[i][num_input_tokens:]

    full_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = []
    for i, answer in enumerate(full_answers):
        if "r1_distill" in model_name:
            answer = answer.split("</think>")[-1]
            if "boxed{" in answer and answer.rfind("}") != -1:
                answer = answer[:answer.rfind("}")].split("boxed{")[-1]
            answer = answer.split("Answer:**")[-1]
            answer = answer.replace("\_", "_").replace("\'", "'")
            if len(answer.split("**")) == 3:
                answer = answer.split("**")[-2]
        answer = answer.strip()
        chat_logs[i].append({'role': 'assistant', 'content': answer})
        answers.append(answer)

    return answers, chat_logs, full_answers


def load_configs_from_file(file_path: str | Path = None) -> dict[str, str | int | float | bool | None]:
    if file_path is None:
        file_path = Path(__file__).parent / 'llm_configs.json'

    with open(file_path) as json_file:
        configs = json.load(json_file)
    if "llm" in configs:
        return configs["llm"]
    return configs


class LlmBridge:
    def __init__(self, model_name: str, configs_path: str | Path = None, logger: Logger=None):
        self.logger = logger
        configs = load_configs_from_file(configs_path)

        self.model_name = model_name
        self.temperature = configs["llm_temperature"]
        self.seed = configs["llm_seed"]

        self.parallelization_mode = configs["llm_parallelization_mode"]
        self.initial_system_message = configs["llm_default_system_message"]

        self.do_sample = configs["llm_do_sample"]
        self.top_p = configs["llm_top_p"]
        self.max_output_tokens = configs["llm_max_output_tokens"]
        if "gpt" in self.model_name:
            load_dotenv()
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif "deepseek" in self.model_name:
            load_dotenv()
            self.client = OpenAI(api_key=os.environ.get("DEEP_SEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        else:
            if "r1_distill" in self.model_name:
                self.max_output_tokens = configs["llm_max_output_tokens_reasoning_model"]
            model_path = configs[self.model_name + "_path"]
            tokenizer_path = configs[self.model_name + "_path"]
            self.model, self.tokenizer = load_llm(model_path, tokenizer_path)

    def forward_to_openai(self, idx: int, answers: list[str], chat_log: list[dict[str, str]]) -> None:
        if self.temperature is None:
            temperature = 0.0
        else:
            temperature = self.temperature
        response = self.client.chat.completions.create(model=self.model_name, messages=chat_log,
                                                       max_tokens=self.max_output_tokens, top_p=self.top_p,
                                                       temperature=temperature, seed=self.seed)
        answers[idx] = response.choices[0].message.content




    # Take question or prompt as first argument, followed by an optional chat log
    def ask_llm(self, question: str, chat_log=None, log: bool = True) -> [str, list[str]]:
        if "gpt" in self.model_name or "deepseek" in self.model_name:
            chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
            response = self.client.chat.completions.create(model=self.model_name, messages=chat_log,
                                                       temperature=self.temperature, seed=self.seed)
            answer = response.choices[0].message.content
            chat_log.append({'role': 'assistant', 'content': answer})
            full_answer = answer
        else:
            answer, chat_log, full_answer = pipeline(self.model_name, self.model, self.tokenizer, question, chat_log,
                                                     self.initial_system_message, self.max_output_tokens, self.temperature,
                                                     self.do_sample, self.top_p, boxed=True)
            if log and self.logger is not None:
                self.logger.log(f"\n[Ask Question]: {question}\n\n[{self.model_name} Full Answer]: {full_answer}\n\n"
                                f"[{self.model_name} Shortened Answer]: {answer}\n")
        return answer, chat_log, full_answer

    def ask_llm_batch(self, questions: list[str], chat_logs:list[dict[str,str]] =None):
        if "gpt" in self.model_name or "deepseek" in self.model_name and self.parallelization_mode == "batch_processing":
            raise ValueError("Batch processing is currently not supported for gpt. "
                             "Use parallelization_mode 'multiprocessing' instead.")
        elif "gpt" in self.model_name or "deepseek" not in self.model_name and self.parallelization_mode == "multiprocessing":
            raise ValueError("Multi-threading is not supported for local models. "
                             "Use parallelization mode 'batch_processing' instead.")

        if self.logger is not None:
            self.logger.log(f"\n[LLM Query 1/{len(questions)} in batch]: {questions[0]}\n")

        if self.parallelization_mode == "sequential":
            if chat_logs is None:
                chat_logs = [None for _ in range(len(questions))]
            answers, chat_logs_new, full_answers = [], [], []
            for i in range(len(questions)):
                answer, chat_log, full_answer = self.ask_llm(questions[i], chat_logs[i], log=False)
                answers.append(answer)
                full_answers.append(full_answer)
                chat_logs[i] = chat_log
            self.logger.log(f"[{self.model_name} Full Answer]: {full_answers[0]}\n")
        else:
            if "gpt" in self.model_name or "deepseek" in self.model_name:
                if chat_logs is None:
                    chat_logs = [None for _ in range(len(questions))]
                procs = []
                answers = [None for _ in range(len(questions))]
                for idx, question in enumerate(questions):
                    chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_logs[idx])
                    chat_logs[idx] = chat_log
                    p = Thread(target=self.forward_to_openai, args=(idx, answers, chat_log))
                    procs.append(p)
                    p.start()
                for p in procs:
                    p.join()
                for i, answer in enumerate(answers):
                    chat_logs[i].append({"role": "assistant", "content": answer})
            else:
                answers, chat_logs, full_answers = pipeline_batch(self.model_name, self.model, self.tokenizer,
                                                                  questions, self.initial_system_message,
                                                                  self.max_output_tokens, chat_logs, self.temperature,
                                                                  self.do_sample, self.top_p, boxed=True)
                self.logger.log(f"[{self.model_name} Full Answer]: {full_answers[0]}\n")
        self.logger.log(f"[{self.model_name} Shortened Answer]: {answers[0]}\n")
        return answers, chat_logs

