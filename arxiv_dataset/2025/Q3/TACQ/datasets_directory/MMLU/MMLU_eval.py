import argparse
import datetime
import os
import random
import numpy as np
import pandas as pd
import torch
import gc
import pickle
from datasets_directory.MMLU.MMLU_utils import choices, format_example, gen_prompt, get_indices_for_choices
from datasets_directory.MMLU.logit_extraction_utils import aggregate_hidden_states_for_last_token
import huggingface_hub
from dotenv import load_dotenv
load_dotenv()
from utils.model_utils import load_model
token = os.getenv('HUGGINGFACE_TOKEN')
huggingface_hub.login(token=token)

print("CUDA detection", torch.version.cuda, torch.cuda.device_count())
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def calculate_confidence_from_logits_with_softmax_first(full_logits: torch.Tensor, indices_for_choices: list):
    last_token_logit = full_logits[:,full_logits.shape[1]-1,:].squeeze()
    probabilities_over_all_tokens = torch.softmax(last_token_logit, -1)
    indexes_of_abcd = torch.tensor(indices_for_choices).to(device)  
    probabilities_of_abcd = torch.gather(probabilities_over_all_tokens, dim=0, index=indexes_of_abcd)
    return probabilities_of_abcd

def calculate_confidence_from_logits(full_logits: torch.Tensor, indices_for_choices: list):
    last_token_logit = full_logits[:,full_logits.shape[1]-1,:].squeeze()
    indexes_of_abcd = torch.tensor(indices_for_choices).to(device)  
    logits_of_abcd = torch.gather(last_token_logit, dim=0, index=indexes_of_abcd)
    probabilities_of_abcd = torch.softmax(logits_of_abcd, -1)
    return probabilities_of_abcd

@torch.no_grad()
def eval(args, subject, engine, dev_df, test_df, tokenizer, model, evaluation_results):
    indices_for_choices = get_indices_for_choices(engine)
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    response_hidden_states = []
    response_ABCD_logits = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]

        # Crop training prompt if too long
        prompt_token_length = len(tokenizer(prompt).input_ids)
        while prompt_token_length > args.context_length:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            prompt_token_length = len(tokenizer(prompt).input_ids)
            raise Exception("Prompt too long")

        # Pass prompt to model
        input = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**input, output_hidden_states=True)
        greedy_token = torch.argmax(outputs.logits[:,-1], dim=-1)
        probabilities_for_abcd = calculate_confidence_from_logits(outputs.logits, indices_for_choices).to("cpu").detach().numpy()
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probabilities_for_abcd)]
        hidden_states = outputs.hidden_states

        # Save telemetry about raw confidences from the lm head at the last layer
        cor = pred == label
        cors.append(cor)
        all_probs.append(probabilities_for_abcd)

        # Obtain hidden states for every layer
        aggregated_hidden_states = aggregate_hidden_states_for_last_token(hidden_states)
        response_hidden_states.append((prompt_end, aggregated_hidden_states))

        # Obtain confidences on logits for every layer
        response_ABCD_logits = []
        response_all_layers_logits = model.lm_head(aggregated_hidden_states)
        for i in range(response_all_layers_logits.shape[0]):  # for every layer's hidden state
            gather_indices = torch.tensor(indices_for_choices).to(device)
            response_ABCD_logits.append(torch.gather(response_all_layers_logits[i][0], dim=-1, index=gather_indices))
        response_ABCD_logits = torch.stack(response_ABCD_logits, dim=0)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    evaluation_results[f"{subject} accuracy"] = acc
    print("Average accuracy {:.5f} - {}".format(acc, subject))

    return cors, acc, all_probs, response_hidden_states, response_ABCD_logits

def main(args):
    engines = args.engine
    device = args.device

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    try:
        from datasets_directory.MMLU.categories import subject_to_category_GPT4o, categories_to_subjects
        if args.MMLU_split == "MMLU_MCQA":
            subset_subjects = subjects
        elif args.MMLU_split == "MMLU_STEM":
            subset_subjects = categories_to_subjects["STEM"]
        elif args.MMLU_split == "MMLU_social_sciences":
            subset_subjects = categories_to_subjects["social sciences"]
        elif args.MMLU_split == "MMLU_humanities":
            subset_subjects = categories_to_subjects["humanities"]
        else:
            subset_subjects = subjects
            print(f"Warning: selected subset {args.MMLU_split} does not exist, default to evaluate on full MMLU_MCQA. You are probably conditioned on an none MMLU dataset.")
        subjects = subset_subjects
    except:
        print("Error: Subjects subsetting error.")


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    evaluation_results = {}
    print(subjects)
    print(args)
    for engine in engines:

        # Wipe previous model from memory
        gc.collect()
        torch.cuda.empty_cache()
        if device == "mps":
            torch.mps.empty_cache()

        # Load next model
        model_info = load_model(engine=engine, checkpoints_dir=args.addition_dir, brainfloat=args.brainfloat)
        model, tokenizer = model_info["model"], model_info["tokenizer"]
        print("LLM running:", engine)

        all_cors = []
        for subject in subjects:  # TEST CHANGE ONLY
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)  # TEST CHANGE ONLY

            # Cut test df
            start, stop = int(test_df.shape[0]*args.eval_start_p), int(test_df.shape[0]*args.eval_end_p)
            if args.testing: stop = start+2
            test_df = test_df[start:stop]
            
            cors, acc, probs, response_hidden_states, response_ABCD_logits = eval(args, subject, engine, dev_df, test_df, tokenizer=tokenizer, model=model, evaluation_results=evaluation_results)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)
            if args.save_full_hidden_state:
                pickle.dump(response_hidden_states, open(os.path.join(args.save_dir, "results_{}".format(engine), "{}_hidden_states.pickle".format(subject)), "wb"))


        weighted_acc = np.mean(np.concatenate(all_cors))
        evaluation_results["Average accuracy"] = weighted_acc
        print("Average accuracy: {:.5f}".format(weighted_acc))
        # Save to JSON
        with open(os.path.join(args.save_dir, "results_{}".format(engine), "0_evaluation_results.json"), "w") as f:
            import json
            json.dump(evaluation_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, required=True)
    parser.add_argument("--addition_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--serial_number", type=int, default=0)
    parser.add_argument("--engine",
                        default=["Meta-Llama-3-8B"], nargs="+")
    parser.add_argument("--MMLU_split", type=str, default="MMLU_MCQA")
    parser.add_argument("--context_length", type=int, default=8000)
    parser.add_argument("--save_full_hidden_state", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--eval_start_p", type=float, default=0.75)
    parser.add_argument("--eval_end_p", type=float, default=1.00)
    parser.add_argument("--brainfloat", action="store_true")
    args = parser.parse_args()
    random.seed(int(args.serial_number))
    np.random.seed(int(args.serial_number))
    torch.manual_seed(int(args.serial_number))
    torch.cuda.manual_seed(int(args.serial_number))

    print("Logging_started")
    print(f"{datetime.datetime.now()=}")
    main(args)
    print(f"{datetime.datetime.now()=}")

