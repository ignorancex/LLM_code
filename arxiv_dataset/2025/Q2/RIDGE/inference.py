import argparse
import jsonlines
import time
import warnings
import os
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from peft import PeftModel

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig

from train import process_dataset


def evaluate(input, model, tokenizer, generation_config, verbose=True):
    inputs = tokenizer(input, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        # output_scores=True,
    )
    outputs = []
    for s in generation_output.sequences:
        output = tokenizer.decode(s, skip_special_tokens=True)
        print("======= ALL OUTPUT =======\n", output) if verbose else None
        output = output.split("[/INST]")[1].strip()
        outputs.append(output)

    return outputs

def main(args):
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    print("Adapter path:", args.adapter_dir)
    print("Output file path:", args.output_file)
    print("Do_sample:", args.do_sample, type(args.do_sample))
    do_sample = True if args.do_sample == 1 else False
    print("Temperature:", args.temperature, type(args.temperature))
    print("Top-p:", args.top_p, type(args.top_p))

    # load dataset
    dataset = load_dataset("json", data_files=args.input_file, download_mode="force_redownload")
    # print(dataset)
    
    # process dataset
    dataset = process_dataset(dataset, inference=True, testing=args.testing)

    model_name = args.model_name_or_path

    # cache_dir = "./cache"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config if args.quantization else None,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # add_eos_token=True,
        # cache_dir=cache_dir,
        # quantization_config=nf4_config
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # decoding parameters
    generation_config = GenerationConfig(
        do_sample=do_sample,
        temperature=args.temperature if do_sample else 1.0,
        num_beams=1,
        top_p=args.top_p if do_sample else 1.0,
        pad_token_id=2,
        max_new_tokens=args.max_len,
    )

    # jsonl writer
    writer = jsonlines.open(args.output_file, "w")

    def batched_data(dataset, batch_size):
        batch = []
        for sample in dataset:
            batch.append(sample["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    all_time = 0
    for batch in tqdm(batched_data(dataset["train"], args.batch_size)):
        prompt = batch
        start_time = time.time()
        outputs = evaluate(
            input=prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            verbose=True,
        )
        generation_time = time.time() - start_time
        print("Generation time per image (sec):", generation_time / len(batch))
        all_time += generation_time
        for prompt, output in zip(batch, outputs):
            writer.write({"input": prompt, "output": output})

        writer._fp.flush() # flush the buffer
    
    writer.close()

    print("Average generation time per image (sec):", all_time / len(dataset["train"]))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--adapter_dir", type=str, default="./models/RIDGE")
    parser.add_argument("--quantization", type=bool, default=False)
    parser.add_argument("--testing", type=bool, default=True)
    parser.add_argument("--input_file", type=str, default="input_files/example.jsonl")
    parser.add_argument("--output_file", type=str, default="output_files/example.jsonl")
    parser.add_argument("--max_len", type=int, default=4000)
    parser.add_argument("--do_sample", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # set random seed
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    main(args)