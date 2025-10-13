import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM
import copy
import pandas as pd
from dataset_utils.bigbench import get_bb_dataset
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress


class Results:

    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_dict(self):
        return {
            "val_acc": self.val_acc,
            "val_logloss": self.val_logloss,
            "test_acc": self.test_acc,
            "test_logloss": self.test_logloss
        }

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"


class GPTJExperiment:

    def __init__(self, save_dir, logger, device):

        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = device

    def intervene(self, model, tokenizer, dataset, args):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention for layer number {args.lnum}, "
                        f"layer type {args.lname}, rate {args.rate}. Dataset size {dataset_size}.")

        time_edit_start = time.time()
        if args.mode == 'laser':
            print('laser mode')
            model = model.to(args.device)
            model_edit = LaserWrapper.get_edited_model(model=model,
                                                       lname=args.lname,
                                                       lnum=args.lnum,
                                                       rate=args.rate,
                                                       intervention=args.intervention,
                                                       logger=logger,
                                                       in_place=True)
        elif args.mode == '4D_Tucker':  
            print('4D Tucker mode')
            model_edit = model.to(args.device)
            model_edit = LaserWrapper.get_QKVO_edited_model(model=model_edit, 
                                                       lnum=args.lnum, 
                                                       device=args.device,
                                                       qkvo_rank=args.qkvo_rank,
                                                       stack_rank=args.stack_rank,
                                                       head_dim_rank=args.head_dim_rank,
                                                       qkvo_intervention=args.tucker_type, 
                                                       logger=logger,
                                                       in_place=True)
        elif args.mode == '3D_Tucker':
            print('3D Tucker mode')
            model_edit = model.to(args.device)
            model_edit = LaserWrapper.get_3D_Tucker_edited_model(model=model_edit, 
                                                       lnum=args.lnum, 
                                                       device=args.device,
                                                       qkvo_rank=args.qkvo_rank,
                                                       attention_matrix=args.attention_matrix,
                                                       logger=logger,
                                                       in_place=True)
        elif args.mode == '4D_Tucker_laser':
            model_edit = model.to(args.device)
            model_edit = LaserWrapper.get_edited_model(model=model_edit,
                                                       lname='mlp',
                                                       lnum=args.lnum,
                                                       rate=args.rate,
                                                       intervention=args.intervention,
                                                       logger=logger,
                                                       in_place=True)
            
            model_edit = LaserWrapper.get_QKVO_edited_model(model=model_edit, 
                                                       lnum=args.lnum, 
                                                       device=args.device, 
                                                       qkvo_rank=args.qkvo_rank,
                                                       stack_rank=args.stack_rank,
                                                       head_dim_rank=args.head_dim_rank,
                                                       qkvo_intervention=args.tucker_type, 
                                                       logger=logger,
                                                       in_place=True)

        model_edit.to(torch.float16).to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

        
        for i in tqdm(range(0, dataset_size, args.batch_size)):

            if (i % args.batch_size) == 0:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            batch_end = min(i + args.batch_size, dataset_size)
            batch = dataset[i:batch_end]
            
            prompts = [entry[0].strip() for entry in batch]
            answers = [entry[1].strip() for entry in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_and_answers = tokenizer([f"{p} {a}" for p, a in zip(prompts, answers)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            input_and_answers["input_ids"] = input_and_answers["input_ids"].to(torch.long)
            
            inputs = {k: (v.to(torch.float16) if k != "input_ids" else v) for k, v in inputs.items()}
            input_and_answers = {k: (v.to(torch.float16) if k != "input_ids" else v) for k, v in input_and_answers.items()}

            with torch.no_grad():

                generate_ids = model_edit.generate(inputs["input_ids"],
                                                   max_new_tokens=args.max_len,
                                                   min_new_tokens=1,
                                                   pad_token_id=tokenizer.eos_token_id, 
                                                   attention_mask=inputs["attention_mask"])

                generations = tokenizer.batch_decode(generate_ids,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

                # Compute log probability of question + answer
                results = model_edit(input_and_answers["input_ids"], attention_mask=input_and_answers["attention_mask"])
                logits = results.logits  # Batch size x (Question + Answer length) x vocab
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                for j, (prompt, answer, generation, input_and_answer) in enumerate(zip(prompts, answers, generations, input_and_answers["input_ids"])):
                    log_prob = log_probs[j]
                    log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                                                                    question_answer_token_ids=input_and_answer,
                                                                    answer=answer,
                                                                    llm_tokenizer=tokenizer)

                    # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
                    # correct_log_prob_results = [all_log_prob_results[answer_ix] for answer_ix in correct_answers]
                    is_correct = self.metrics.generation_match(generation=generation, answer=answer)
                    f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

                    self.dataset_metric.accept(is_correct=is_correct,
                                               f1pr_score=f1pr_score,
                                               log_prob_results=log_prob_results)
        
                    predictions_ = {
                        "ix": i,
                        "question": prompt,
                        "gold-answer": answer,
                        "generation": "N/A",
                        "correct": is_correct,
                        "f1_score": f1pr_score.f1,
                        "precision": f1pr_score.precision,
                        "recall": f1pr_score.recall,
                        "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                        "white-space-strip": self.strip,              # We ignore white space when checking answer
                        "total_logprob": log_prob_results.total_log_prob,
                        "answer_logprob": log_prob_results.answer_log_prob,
                        "answer_length": log_prob_results.answer_len
                    }
                    predictions.append(predictions_)

        # Save results and terminate
        # self.terminate_and_save(predictions)

        return predictions

    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.lnum}-{args.lname}-{args.rate}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.lnum}-{args.lname}-{args.rate}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")

    @staticmethod
    def get_acc_log_loss(predictions):

        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"])
                            for prediction in predictions])

        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):

        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]

        val_acc, val_logloss = GPTJExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = GPTJExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=9.9, help='rates for intervention')
    parser.add_argument('--qkvo_rank', type=int, default=10, help='rank in the mode of hidden dimension')
    parser.add_argument('--head_dim_rank', type=int, default=10, help='rank in the mode of attention head dimension')
    parser.add_argument('--stack_rank', type=int, default=10, help='rank in the mode of stacking QKVO')
    parser.add_argument('--split', type=str, default="qa_wikidata", help='big bench split to run on')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=10, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                       choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 
                                'fc_in', 'fc_up', 'fc_out', 'None', 'dont', 'attn', 'mlp', 'all'],
                       help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=27, help='Layers to edit', choices=list(range(-1, 28)))
    parser.add_argument('--home_dir', type=str,
                        default="/rds/general/user/yg1221/home/TensorLLM/results/GPTJ_laser_wikiqa",
                        help='Directory where the data is')
    parser.add_argument('--mode', type=str, default="None", choices=['laser', '4D_Tucker', '4D_Tucker_laser', '3D_Tucker'], help="Which mode to intervene the model")
    parser.add_argument('--attention_matrix', type=str, default="None", choices=['Q', 'K', 'V', 'O'], help="Which attention matrix to decompose in 3D_Tucker mode")
    parser.add_argument('--start_rank', type=int, default=10, help='experiment with rank starting from start_rank')
    parser.add_argument('--end_rank', type=int, default=10, help='experiment with rank ending with end_rank')
    parser.add_argument('--start_layer', type=int, default=25, help='experiment with layer starting from start_layer')
    parser.add_argument('--end_layer', type=int, default=25, help='experiment with layer ending from end_layer')
    parser.add_argument('--tucker_type', type=str, default="partial_tucker",
                        choices=['partial_tucker', 'partial_tucker_v2','partial_tucker_v3',
                                 'partial_tucker_v4', 'partial_tucker_v5'], help="what type of intervention to perform")
    parser.add_argument('--device', type=str, default="cuda:0", help='which device to use')
    parser.add_argument('--single_experiment', action='store_true', help='Flag to run a single experiment')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "GPTJ"
    llm_path = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )
    original_state_dict = model.state_dict()

    # Step 3: Create save directory and logger
    home_dir = args.home_dir

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = GPTJExperiment(save_dir=save_dir, logger=logger, device=args.device)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset, _ = get_bb_dataset(args.split)

    # Step 6: Run intervention
    ranks = range(args.start_rank, args.end_rank + 1, 1)
    layers = [layer for layer in range(args.start_layer, args.end_layer+1)]
    
    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
    stack_ranks = [1,2,3,4] if args.mode=='4D_Tucker' or args.mode=='4D_Tucker_laser' else range(0,1)
    
    if args.mode == '3D_Tucker':
        for matrix in ['Q', 'K', 'V', 'O']:
            args.attention_matrix = matrix
            print(f'3D Tucker on {args.attention_matrix} only')
            model.load_state_dict(original_state_dict)
            predictions = experiment.intervene(model=model,
                                                tokenizer=tokenizer,
                                                dataset=dataset,
                                                args=args)
            
            results = experiment.validate(predictions)
            results_dict = results.to_dict()
            
            try:
                results_df
            except NameError:
                results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
                
            # Create a DataFrame with the new results
            new_data = pd.DataFrame([{
                "Layer": args.lnum,
                "Matrix": args.attention_matrix,
                "Rank_qkvo": args.qkvo_rank,
                "Val Acc": results_dict["val_acc"],
                "Val Logloss": results_dict["val_logloss"],
                "Test Acc": results_dict["test_acc"],
                "Test Logloss": results_dict["test_logloss"]
            }])
            
            # Concatenate the new data to the results DataFrame
            results_df = pd.concat([results_df, new_data], ignore_index=True)
            
            # Save the results to a CSV file
            results_df.to_csv(f"{home_dir}/GPTJ_mode{args.mode}_type{args.tucker_type}_lname{args.lname}_rank{args.start_rank}_{args.end_rank}_layer{args.start_layer}_{args.end_layer}_wikiQA_RESULTS.csv", index=False)
            print(f"layer: {args.lnum}; qkvo rank: {args.qkvo_rank}; {results.to_str()}")
            
    elif args.single_experiment == False:
        for layer in layers:
            for rank in ranks:
                for stack_rank in stack_ranks:
                    args.lnum = layer
                    args.qkvo_rank = rank
                    args.stack_rank = stack_rank
                    print(f'layer: {args.lnum}; qkvo rank: {args.qkvo_rank}; stack rank: {args.stack_rank}')
                    model.load_state_dict(original_state_dict)
                    predictions = experiment.intervene(model=model,
                                                    tokenizer=tokenizer,
                                                    dataset=dataset,
                                                    args=args)
                    
                    results = experiment.validate(predictions)
                    results_dict = results.to_dict()
                    
                    try:
                        results_df
                    except NameError:
                        results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
                        
                    # Create a DataFrame with the new results
                    new_data = pd.DataFrame([{
                        "Layer": layer,
                        "Rank_qkvo": rank,
                        "Rank_stack": stack_rank,
                        "Val Acc": results_dict["val_acc"],
                        "Val Logloss": results_dict["val_logloss"],
                        "Test Acc": results_dict["test_acc"],
                        "Test Logloss": results_dict["test_logloss"]
                    }])
                    
                    # Concatenate the new data to the results DataFrame
                    results_df = pd.concat([results_df, new_data], ignore_index=True)
                    
                    # Save the results to a CSV file
                    results_df.to_csv(f"{args.home_dir}/GPTJ_mode{args.mode}_lname{args.lname}_rank{args.start_rank}_{args.end_rank}_layer{args.start_layer}_{args.end_layer}_bbh_wikiqa_RESULTS.csv", index=False)
                    print(f"Layer {layer}, Rank {rank}, {results.to_str()}")
                    
    elif args.single_experiment:
        model.load_state_dict(original_state_dict)
        if args.mode == 'laser':
            print(f'LASER: Layer: {args.lnum}, rate: {args.rate}, lname: {args.lname}')
            predictions = experiment.intervene(model=model,
                                            tokenizer=tokenizer,
                                            dataset=dataset,
                                            args=args)
        else:
            print(f'TensorLLM: Layer: {args.lnum}, Rank_qkvo: {args.qkvo_rank}, Rank_head_dim: {args.head_dim_rank}, Rank_stack: {args.stack_rank}')
            args.tucker_type = 'partial_tucker_v5'
            predictions = experiment.intervene(model=model,
                                            tokenizer=tokenizer,
                                            dataset=dataset,
                                            args=args)
        
        results = experiment.validate(predictions)        
        print(f"{results.to_str()}")
                
    logger.log("Experiment Complete")
