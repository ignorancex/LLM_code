# -*- coding: utf-8 -*-
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
import argparse
import json
from threading import Thread
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    LlamaForCausalLM
)
import numpy as np

from supervised_finetuning import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "mistral": (MistralForCausalLM, AutoTokenizer)
}

@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    """Generate answer from prompt with GPT and stream the output"""
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    return generated_text


def get_zero_shot_sentence(sentence):
    print(sentence)
    segments = sentence[0]["content"].split("<input>:")
    sentence[0]["content"] = segments[0] + "<input>:" + segments[-1]
    return sentence


@torch.inference_mode()
def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        prompt_template,
        device,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.0,
):
    """Generate answer from prompt with GPT, batch mode"""
    generated_texts = []
    max_length = 32768
    inputs_tokens = tokenizer.apply_chat_template(sentences[0], return_tensors="pt").to(device)
    inputs_tokens = inputs_tokens[:]
    new_inputs_tokens = []
    for i in range(len(inputs_tokens)):
        new_inputs_tokens.append(inputs_tokens[i].tolist())
    new_inputs_tokens = torch.tensor(new_inputs_tokens).to(inputs_tokens.device)
    generated_result = model.generate(new_inputs_tokens, max_new_tokens=2048, do_sample=True, return_dict_in_generate=True, output_scores=True)
    scores = [t.cpu().numpy() for t in generated_result["scores"]]
    scores = torch.tensor(scores).unsqueeze(0)
    scores = torch.tensor(scores).squeeze(2)
    for gen_sequence in scores:
        max_indices = torch.argmax(gen_sequence, dim=1)
        gen_text = tokenizer.decode(max_indices.tolist(), skip_special_tokens=True)
        stop_str = tokenizer.eos_token or prompt_template.stop_str
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="llama", type=str)
    parser.add_argument('--base_model', default=
    "../output/llama_8k_c_n_weights_a3g10s1r10/checkpoint-372/", type=str)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=
    "../models/llama2-hf-chat-7B/", type=str)
    parser.add_argument('--template_name', default="llama2", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--context_size", type=int, default=8192)
    parser.add_argument('--data_file', default="../data/icl_8k/in_context_learning_METB_20240511_test.jsonl.128k.CLSClusteringS2S", type=str,
                        help="A file that contains instructions (one instruction per line)")
    # parser.add_argument('--data_file', default=None, type=str,
    #                     help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', default=False, help="run in the instruction mode (single-turn)")
    parser.add_argument('--output_file', default='../data/icl_8k/in_context_learning_METB_20240511_test.jsonl.128k.CLSClusteringS2S.llama.dynamic', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--cache_dir', default='../msh-cache/', type=str)
    args = parser.parse_args()
    print(args)
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(args.gpus)
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left', model_max_length=args.context_size)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )
    try:
        pass
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)


    model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    print(tokenizer)
    # test data
    examples = ["introduct Beijing"]
    if args.data_file is None:
        examples = ["introduct Beijing"]
    else:
        def load_testdata(filename):
            data = []
            for line in open(filename, "r", encoding="utf-8"):
                line = line.strip()
                messages = json.loads(line)["messages"]
                data.append(messages[0:-1])
            return data
        examples = load_testdata(args.data_file)

    # Chat
    prompt_template = get_conv_template(args.template_name)
    print(prompt_template)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    if args.interactive:
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        history = []
        while True:
            try:
                query = input(f"{prompt_template.roles[0]}: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "exit":
                print("exit...")
                break
            if query.strip() == "clear":
                history = []
                print("history cleared.")
                continue

            print(f"{prompt_template.roles[1]}: ", end="", flush=True)

            history.append([query, ''])
            prompt = prompt_template.get_prompt(messages=history)
            response = stream_generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                do_print=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            if history:
                history[-1][-1] = response.strip()
    else:
        print("Start inference.")
        counts = 0
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        eval_batch_size = args.eval_batch_size
        for batch in tqdm(
                [
                    examples[i: i + eval_batch_size]
                    for i in range(0, len(examples), eval_batch_size)
                ],
                desc="Generating outputs",
        ):
            try:
                responses = batch_generate_answer(
                    batch,
                    model,
                    tokenizer,
                    prompt_template,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                )
            except:
                responses = [""]
            results = []
            for example, response in zip(batch, responses):
                # print(f"===")
                # print(f"Input: {example}")
                # print(f"Output: {response}\n")
                results.append({"id": counts, "Input": example, "Output": response})
                counts += 1
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for entry in results:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
            # break
        print(f'save to {args.output_file}, size: {counts}')


if __name__ == '__main__':
    main()
