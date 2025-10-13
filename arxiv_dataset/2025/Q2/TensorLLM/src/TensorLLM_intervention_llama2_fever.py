import os
import time
import torch
import pickle
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from dataset_utils.fever import FEVER
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
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


class LlamaExperiment:

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
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        model = model.to(args.device)
        if args.mode == 'laser':
            print(f'laser mode, rate = {args.rate}')
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
                                                       head_dim_rank=args.head_dim_rank,
                                                       stack_rank=args.stack_rank,
                                                       qkvo_intervention=args.tucker_type, 
                                                       logger=logger,
                                                       in_place=True)
        elif args.mode == '4D_Tucker_laser':
            print(f'4D Tucker QKVO + laser FC, laser rate = {args.rate}')
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
                                                       head_dim_rank=args.head_dim_rank,
                                                       qkvo_rank=args.qkvo_rank,
                                                       stack_rank=args.stack_rank,
                                                       qkvo_intervention=args.tucker_type, 
                                                       logger=logger,
                                                       in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        # Answer tokens: true and false
        true_token_ids = tokenizer("true")
        assert len(true_token_ids["input_ids"]) == 2 and true_token_ids["input_ids"][0] == 1
        true_token_id = int(true_token_ids["input_ids"][1])

        false_token_ids = tokenizer("false")
        assert len(false_token_ids["input_ids"]) == 2 and false_token_ids["input_ids"][0] == 1
        false_token_id = int(false_token_ids["input_ids"][1])

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            question = dataset[i]["question"]

            # Answer is either 0 (False) or 1 (True)
            answer_ix = dataset[i]["answer"]
            # Given that we do 1-token look up we do the following:
            # - Compute log-prob of the gold token
            # - Compute top-1, top-5 and top-10 accuracies
            if question.strip().endswith(".") or question.strip().endswith("?"):
                # prompted_question = "Is the following claim true or false: " + question.strip() + " The claim is "
                prompted_question = "Consider the following claim: " + \
                                    question.strip() + " Is this claim true or false. The claim is"
            else:
                # prompted_question = "Is the following claim true or false: " + question.strip() + ". The claim is "
                prompted_question = "Consider the following claim: " + \
                                    question.strip() + ". Is this claim true or false. The claim is"
            assert answer_ix in [0, 1]

            inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Compute log probability of question
                results = model_edit(inputs.input_ids)
                logits = results.logits[0]                                      # question length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

                last_token_logprob = log_prob[-1]                               # vocab

                true_logprob = last_token_logprob[true_token_id].item()
                false_logprob = last_token_logprob[false_token_id].item()

                if answer_ix == 1:     # Answer is True
                    answer_log_prob = true_logprob
                    is_correct = true_logprob > false_logprob
                    answer = "true"
                else:               # Answer is False
                    answer_log_prob = false_logprob
                    is_correct = true_logprob < false_logprob
                    answer = "false"

                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)

                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()

                decoded_tokens = tokenizer.batch_decode(top_k_indices)
                top_k_tokens = [token for token in decoded_tokens]
                assert len(top_k_tokens) == 10

                top_1_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:1]])
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

                # Compute log-prob of question and answer
                selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
                indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

                selected_log_prob = torch.gather(selected_log_prob,
                                                 index=indices,
                                                 dim=1)  # question - 1 x 1
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                       answer_log_prob=answer_log_prob,
                                                       answer_len=1)

            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=logprob_results,
                                       top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})

            # if i % 10 == 0:
            #     print(f"Question: {question} and gold answer {answer}. Predicted top 10 tokens {top_k_tokens}.")

            predictions_ = {
                "ix": i,
                "question": question,
                "prompted-question": prompted_question,
                "gold-answer": answer,
                "gold-answer-ix": answer_ix,
                "generation": top_k_tokens[0],      # We can view the top token as the 1-step generation
                "correct": is_correct,
                "true_logprob": true_logprob,
                "false_logprob": false_logprob,
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_logprob": top_k_logprob,
                "top_10_tokens": top_k_tokens,
                "f1_score": None,
                "precision": None,
                "recall": None,
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": total_log_prob,
                "question_logprob": question_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "question_answer_length": inputs.input_ids.shape[1] + 1
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
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

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

        val_acc, val_logloss = LlamaExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = LlamaExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=8, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None',
                                 'dont', 'all', 'mlp', 'attn'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=28, help='Layers to edit', choices=list(range(-1, 32)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/rds/general/user/yg1221/home/FYP/Llama_models/Llama2/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/rds/general/user/yg1221/home/TensorLLM/results/llama2_fever_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/counterfact",
                        help='Directory where the data is')
    parser.add_argument('--qkvo_rank', type=int, default=10, help='rank in the mode of hidden dimension')
    parser.add_argument('--head_dim_rank', type=int, default=10, help='rank in the mode of attention head dimension')
    parser.add_argument('--stack_rank', type=int, default=10, help='rank in the mode of stacking QKVO')
    parser.add_argument('--mode', type=str, default="None", choices=['laser', '4D_Tucker', '4D_Tucker_laser'], help="Which mode to intervene the model")
    parser.add_argument('--start_rank', type=int, default=10, help='experiment with rank starting from start_rank')
    parser.add_argument('--end_rank', type=int, default=10, help='experiment with rank ending with end_rank')
    parser.add_argument('--start_layer', type=int, default=25, help='experiment with layer starting from start_layer')
    parser.add_argument('--end_layer', type=int, default=25, help='experiment with layer ending from end_layer')
    parser.add_argument('--tucker_type', type=str, default="partial_tucker",
                        choices=['partial_tucker', 'partial_tucker_v2', 'partial_tucker_v3', 
                                 'partial_tucker_v4','partial_tucker_v5'], help="what type of intervention to perform")
    parser.add_argument('--device', type=str, default="cuda:0", help='which device to use')
    parser.add_argument('--single_experiment', action='store_true', help='Flag to run a single experiment')
    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "Llama2-7G"
    llm_path = args.model_path
    tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
    model = LlamaForCausalLM.from_pretrained(llm_path)
    original_state_dict = model.state_dict()

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = LlamaExperiment(save_dir=save_dir, logger=logger, device=args.device)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset_util = FEVER()
    dataset = dataset_util.get_dataset(logger)

    # Step 6: Run intervention
    ranks = range(args.start_rank, args.end_rank + 1, 1)
    layers = [layer for layer in range(args.start_layer, args.end_layer+1)]
    
    results_df = pd.DataFrame(columns=["Layer", "Rank", "Val Acc", "Val Logloss", "Test Acc", "Test Logloss"])
    stack_ranks = range(0,1) if args.mode=='laser' else [1,2,3,4]

    if args.single_experiment == False:
        if args.tucker_type == 'partial_tucker_v5':
            for layer in layers:
                for rank1 in ranks:
                    for rank2 in ranks:
                        for stack_rank in stack_ranks:
                            args.lnum = layer
                            args.qkvo_rank = rank1
                            args.head_dim_rank = rank2
                            args.stack_rank = stack_rank
                            print(f'layer: {args.lnum}; qkvo rank: {args.qkvo_rank}; head dim rank: {args.head_dim_rank}; stack rank: {args.stack_rank}')
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
                                "Rank_qkvo": rank1,
                                "head_dim_rank": rank2,
                                "Rank_stack": stack_rank,
                                "Val Acc": results_dict["val_acc"],
                                "Val Logloss": results_dict["val_logloss"],
                                "Test Acc": results_dict["test_acc"],
                                "Test Logloss": results_dict["test_logloss"]
                            }])
                            
                            # Concatenate the new data to the results DataFrame
                            results_df = pd.concat([results_df, new_data], ignore_index=True)
                            
                            # Save the results to a CSV file
                            results_df.to_csv(f"{args.home_dir}/Llama2_mode{args.mode}_type{args.tucker_type}_lname{args.lname}_rank{args.start_rank}_{args.end_rank}_layer{args.start_layer}_{args.end_layer}_fever_RESULTS.csv", index=False)
                            # print(f"Layer {layer}, Rank {rank}, {results.to_str()}")
        else:
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
                        results_df.to_csv(f"{args.home_dir}/Llama2_mode{args.mode}_type{args.tucker_type}_lname{args.lname}_rank{args.start_rank}_{args.end_rank}_layer{args.start_layer}_{args.end_layer}_fever_RESULTS.csv", index=False)
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