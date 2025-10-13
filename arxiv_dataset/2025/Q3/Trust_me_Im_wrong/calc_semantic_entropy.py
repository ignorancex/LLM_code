import gc
import json
import logging
import random
import datasets
from collections import defaultdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    EntailmentGPT4,
    EntailmentGPT35,
    EntailmentGPT4Turbo,
    EntailmentLlama,
    HuggingfaceModel,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
    cluster_assignment_entropy,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set os environment variables



class SemanticEntropy:
    def __init__(self, model_name, dataset_path, entailment_model, max_new_tokens=10):
        random.seed(0)
        self.model_name = model_name
        print(f"{model_name=} {max_new_tokens=}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model.eval()
        self.tokenizer.padding_side = "left"
        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.pad_token_id = None

        self.max_new_tokens = max_new_tokens
        self.stop_sequences = None
        self.semantic_entropy_generation_model = HuggingfaceModel(
            model_name, stop_sequences="default", max_new_tokens=max_new_tokens
        )

        self.metric = None
        print(f"{entailment_model=}")

        self.entailment_model = self.load_entailment_model(entailment_model, None, None)

    def load_entailment_model(
        self, entailment_model, entailment_cache_id, entailment_cache_only
    ):
        """
        Load the entailment model for the semantic entropy calculation using embeddings.
        """
        # Load entailment model.
        print("Beginning loading for entailment model.")
        if entailment_model == "deberta":
            entailment_model = EntailmentDeberta()
        elif entailment_model == "gpt-4":
            entailment_model = EntailmentGPT4(
                entailment_cache_id, entailment_cache_only
            )
        elif entailment_model == "gpt-3.5":
            entailment_model = EntailmentGPT35(
                entailment_cache_id, entailment_cache_only
            )
        elif entailment_model == "gpt-4-turbo":
            entailment_model = EntailmentGPT4Turbo(
                entailment_cache_id, entailment_cache_only
            )
        elif "llama" in entailment_model.lower():
            entailment_model = EntailmentLlama(
                entailment_cache_id, entailment_cache_only, entailment_model
            )
        else:
            raise ValueError
        print("Entailment model loading complete.")

        return entailment_model


    def generate_answers(
        self,
        model,
        example,
        num_generations,
        max_length,
        stop_sequences=None,
        temperature=1.0,
        compute_acc=False,
    ):
        # This will store all input data and model predictions.
        accuracies, generations, results_dict = [], {}, {}

        question = example[0]
        generations[example[0]] = {"question": question}
        correct_answer = example[1]
        # logging.info("Current question: ".ljust(15) + question)
        full_responses = []
        temp = temperature
        # We sample one low temperature answer on which we will compute the
        # accuracy and args.num_generation high temperature answers which will
        # be used to estimate the entropy variants.
        print(f"{num_generations=} {temperature=}")
        for i in range(num_generations):
            # Temperature for first generation is always `0.1`.

            temperature = 0.1 if i == 0 else temp
            print(f"{temperature=} {i=}")
            predicted_answer, token_log_likelihoods, embedding = model.predict(
                question, temperature
            )
            embedding = embedding.cpu() if embedding is not None else None

            # compute accuracy
            if compute_acc and self.metric is not None:
                acc = self.metric(predicted_answer, example, model)
            else:
                acc = 0.0

            if i == 0:

                accuracies.append(acc)
                most_likely_answer_dict = {
                    "response": predicted_answer,
                    "token_log_likelihoods": token_log_likelihoods,
                    "embedding": embedding,
                    "accuracy": acc,
                }
                generations[example[0]].update(
                    {
                        "most_likely_answer": most_likely_answer_dict,
                    }
                )

            else:
                # logging.info(
                #     "high-t prediction ".ljust(15) + str(i) + " : " + predicted_answer
                # )
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc)
                )

        # Append all predictions for this example to `generations`.
        generations[example[0]]["responses"] = full_responses
        generations[example[0]]["reference"] = correct_answer
        # save the results
        results_dict["accuracies"] = accuracies
        results_dict["generations"] = generations
        results_dict["question"] = question
        results_dict["reference"] = correct_answer

        return results_dict

    def compute_uncertainty_measures(
        self,
        model_generations,
        compute_predictive_entropy,
        entailment_model,
        strict_entailment,
    ):
        # compute semantic entropy uncertainty measure
        result_dict = {}
        result_dict["semantic_ids"] = []

        entropies = defaultdict(list)
        validation_embeddings = []
        p_trues = []
        count = 0
        print(f"len(model_generations)={len(model_generations)} {model_generations=}")
        # Loop over datapoints and compute validation embeddings and entropies.
        for idx, tid in enumerate(model_generations):

            example = model_generations[tid]
            print(f"{example=} {example.keys()=}")
            question = example["question"]
            full_responses = example["responses"]
            most_likely_answer = example["most_likely_answer"]
            print(f"{most_likely_answer=} {full_responses=} {question=}")
            responses = [r[0] for r in full_responses]
            if most_likely_answer["response"] is None or most_likely_answer["token_log_likelihoods"] is None or most_likely_answer["embedding"] is None or\
            (None, None, None, 0.0) in full_responses:
                print("No responses for this example.", idx, tid)
                continue
            validation_embeddings.append(most_likely_answer["embedding"])


            if compute_predictive_entropy:
                # Token log likelihoods. Shape = (n_sample, n_tokens)
                log_liks = [r[1] for r in full_responses]

                if entailment_model == "deberta":
                    responses = [f"{question} {r}" for r in responses]

                # Compute semantic ids.
                semantic_ids = get_semantic_ids(
                    responses,
                    model=entailment_model,
                    strict_entailment=strict_entailment,
                    example=example,
                )

                result_dict["semantic_ids"].append(semantic_ids)
                print(f"{len(log_liks)=} {len(log_liks[0])=} {len(semantic_ids)=}")

                # Compute entropy from frequencies of cluster assignments.
                entropies["cluster_assignment_entropy"].append(
                    cluster_assignment_entropy(semantic_ids)
                )

                # Length normalization of generation probabilities.
                log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

                print(f"{log_liks_agg=} {np.mean(log_liks_agg)=}")

                # Compute naive entropy.
                entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

                # Compute semantic entropy.
                log_likelihood_per_semantic_id = logsumexp_by_id(
                    semantic_ids, log_liks_agg, agg="sum_normalized"
                )
                pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
                entropies["semantic_entropy"].append(pe)


                log_str = (
                    "semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s"
                )
                entropies_fmt = ", ".join(
                    [f"{i}:{j[-1]:.2f}" for i, j in entropies.items()]
                )

                logging.info(80 * "#")
                logging.info("NEW ITEM %d at id=`%s`.", idx, tid)
                logging.info("Context:")
                # logging.info(example["context"])
                logging.info("Question:")
                logging.info(question)
                logging.info("True Answers:")
                logging.info(example["reference"])
                logging.info("Low Temperature Generation:")
                logging.info(most_likely_answer["response"])
                logging.info("Low Temperature Generation Accuracy:")
                logging.info(most_likely_answer["accuracy"])
                logging.info("High Temp Generation:")
                logging.info([r[0] for r in full_responses])
                logging.info("High Temp Generation:")
                logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

                count += 1
        if count == 0:
            print("No valid examples!!!")
            return None, None
        print(f'{entropies["semantic_entropy"]=}')
        # Compute average entropies.
        avg_entropies = {k: np.mean(v) for k, v in entropies.items()}
        print("Average entropies: %s", avg_entropies)

        # Compute average embeddings.
        validation_embeddings = torch.stack(validation_embeddings)
        avg_embedding = torch.mean(validation_embeddings, dim=0)
        print("Average embedding: %s", avg_embedding)

        return avg_entropies, avg_embedding



    def calc_semantic_entropy_per_example(self,prompt:str,answer:str,temp:float=1.0):
        print(f"{prompt=} {answer=} {temp=}")
        results = self.generate_answers(
                    self.semantic_entropy_generation_model,
                    [prompt,answer],
                    num_generations=11,
                    max_length=10,
                    stop_sequences=None,
                    temperature=temp,
                    compute_acc=True,
                )
        avg_entropies, avg_embedding = self.compute_uncertainty_measures(
                    results["generations"],
                    compute_predictive_entropy=True,
                    entailment_model=self.entailment_model,
                    strict_entailment=False,
                )
        return avg_entropies, results["generations"]