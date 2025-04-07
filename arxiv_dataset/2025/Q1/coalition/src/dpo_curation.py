
import json
import torch
from typing import Dict
import pytorch_lightning as pl
import torch.nn.functional as F

class DPOCurationModule(pl.LightningModule):
    def __init__(self, model1, model2, tokenizer, args, cache_path=""):
        super(DPOCurationModule, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokenizer
        self.args = args
        self.cache_path = cache_path

        self.prompt, self.chosen, self.rejected, self.response = [], [], [], []


    def get_seed_prompt(self, generation_mode):
        if generation_mode == "rationale":
            return "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate descriptive reasoning on how to derive the correct answer for the instruction such that the descriptive reasoning will be useful to another AI assistant to generate the correct answer.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif generation_mode == "response":
            return "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate the correct answer for the given instruction.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif generation_mode == "rationale_conditioned_response":
            return "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. Analyse the rationale and come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif generation_mode == "rationale_refinement":
            return "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. The 'Rationale may or may not be correct for the given 'Instruction'. 'Analyse the rationale for its correctness, modify the rationale, and provide the correct elaborate descriptive reasoning or 'Rationale' which will be helpful to come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        else:
            raise NotImplementedError("Response generation mode is not implemented yet")

    def _create_labels(self, tokenized_input: Dict[str, torch.tensor]):

        input_ids = tokenized_input["input_ids"]
        token_type_ids = tokenized_input["token_type_ids"]

        labels = input_ids.new_zeros(input_ids.shape)
        labels[:, :-1] = input_ids[:, 1:].clone()

        shifted_token_type_ids = token_type_ids.new_zeros(token_type_ids.shape)
        shifted_token_type_ids[:, :-1] = token_type_ids[:, 1:].clone()

        labels.masked_fill_(shifted_token_type_ids != 1, -100)
        return labels

    def _get_average_likelihood(self, logits, labels, ignore_index = -100):

        log_probs = F.log_softmax(logits, dim=-1)
        # Create a mask for valid labels
        valid_labels_mask = labels != ignore_index
        
        # Replace -100 in labels with a dummy valid label index, e.g., 0
        # We use this modified labels tensor only for gathering
        labels_masked = labels.clone()
        labels_masked[~valid_labels_mask] = 0  # Replace -100 with 0
        
        # Gather the log probabilities of the labels across the vocabulary dimension
        # Now labels_masked does not contain any -100 values, so gather will not error out
        gathered_log_probs = torch.gather(log_probs, 2, labels_masked.unsqueeze(-1)).squeeze(-1)
        
        # Apply the valid labels mask: zero out contributions from invalid labels (-100 originally)
        gathered_log_probs[~valid_labels_mask] = 0
        
        # Compute the log likelihood for each tensor in the batch by summing over the sequence length
        average_likelihood = torch.exp(gathered_log_probs.sum(dim=1) / torch.sum(valid_labels_mask, dim = 1))

        return average_likelihood

    def get_multimode_average_likelihood_for_response(self, prompt, responses):

        tokenized_prompt = self.tokenizer(prompt, responses, padding="max_length", 
                                            truncation=self.args.truncation, max_length=self.args.turn1_max_length, 
                                            return_attention_mask=self.args.return_attention_mask,
                                            return_token_type_ids=self.args.return_token_type_ids,
                                            return_tensors=self.args.return_tensors)

        prompt_labels = self._create_labels(tokenized_prompt)
        model1_logits = self.model1(input_ids=tokenized_prompt["input_ids"].to(self.device),
                                    attention_mask=tokenized_prompt["attention_mask"].to(self.device),
                                    labels=prompt_labels).logits.detach().cpu()
        model2_logits = self.model2(input_ids=tokenized_prompt["input_ids"].to(self.device),
                                    attention_mask=tokenized_prompt["attention_mask"].to(self.device),
                                    labels=prompt_labels).logits.detach().cpu()
        
        model1_likelihood = self._get_average_likelihood(logits=model1_logits, labels=prompt_labels)
        model2_likelihood = self._get_average_likelihood(logits=model2_logits, labels=prompt_labels)

        average_likelihood = (model1_likelihood + model2_likelihood) / 2

        return average_likelihood


    def forward(self, instructions, responses, rationales1, rationales2, refined_rationales1, refined_rationales2, response_types):

        prompt_without_rationale = [self.get_seed_prompt("response").format(instruction = instruction) for idx, instruction in enumerate(instructions)]

        prompt_rationale1 = [self.get_seed_prompt("rationale_conditioned_response").format(instruction=instruction, rationale=rationale) for idx, (instruction, rationale) in enumerate(zip(instructions, rationales1))]
        prompt_rationale2 = [self.get_seed_prompt("rationale_conditioned_response").format(instruction=instruction, rationale=rationale) for idx, (instruction, rationale) in enumerate(zip(instructions, rationales2))]
        
        prompt_refined_rationale1 = [self.get_seed_prompt("rationale_conditioned_response").format(instruction=instruction, rationale=rationale) for idx, (instruction, rationale) in enumerate(zip(instructions, refined_rationales1))]
        prompt_refined_rationale2 = [self.get_seed_prompt("rationale_conditioned_response").format(instruction=instruction, rationale=rationale) for idx, (instruction, rationale) in enumerate(zip(instructions, refined_rationales2))]
        
        if self.args.filter_samples:

            average_likelihood_without_rationale = self.get_multimode_average_likelihood_for_response(prompt_without_rationale, responses)
            average_likelihoods_turn1_rationale1 = self.get_multimode_average_likelihood_for_response(prompt_rationale1, responses)
            average_likelihoods_turn1_rationale2 = self.get_multimode_average_likelihood_for_response(prompt_rationale2, responses)

            average_likelihoods_turn2_rationale1 = self.get_multimode_average_likelihood_for_response(prompt_refined_rationale1, responses)
            average_likelihoods_turn2_rationale2 = self.get_multimode_average_likelihood_for_response(prompt_refined_rationale2, responses)

            prompt, chosen, rejected, response = [], [], [], []
            for i, response_type in enumerate(response_types):

                # NOTE: The likelihood of ground-truth response without rationale should be less than the likelihood of the chosen response, otherwise don't use the sample
                # however, chosen rationale if the one whose likelihood the ground-truth response conditioned on rationale if greater than the other rationale
                if response_type == "rationale":
                    if average_likelihood_without_rationale[i] < average_likelihoods_turn1_rationale1[i] and average_likelihoods_turn1_rationale1[i] > average_likelihoods_turn1_rationale2[i]:
                        prompt.append(self.get_seed_prompt("rationale").format(instruction=instructions[i]))
                        chosen.append(rationales1[i]); rejected.append(rationales2[i]); response.append(responses[i])
                    elif average_likelihood_without_rationale[i] < average_likelihoods_turn1_rationale2[i] and average_likelihoods_turn1_rationale2[i] > average_likelihoods_turn1_rationale1[i]:
                        prompt.append(self.get_seed_prompt("rationale").format(instruction=instructions[i]))
                        chosen.append(rationales2[i]); rejected.append(rationales1[i]); response.append(responses[i])
                    else:
                        continue

                elif response_type == "rationale_refinement":
                    if average_likelihoods_turn1_rationale1[i] < average_likelihoods_turn2_rationale1[i] and average_likelihoods_turn2_rationale1[i] > average_likelihoods_turn2_rationale2[i]:
                        prompt.append(self.get_seed_prompt("rationale_refinement").format(instruction=instructions[i], rationale=rationales1[i] if self.args.guide_index == 1 else rationales2[i]))
                        chosen.append(refined_rationales1[i]); rejected.append(refined_rationales2[i]); response.append(responses[i])
                    elif average_likelihoods_turn1_rationale2[i] < average_likelihoods_turn2_rationale2[i] and average_likelihoods_turn2_rationale2[i] > average_likelihoods_turn2_rationale1[i]:
                        prompt.append(self.get_seed_prompt("rationale_refinement").format(instruction=instructions[i], rationale=rationales1[i] if self.args.guide_index == 1 else rationales2[i]))
                        chosen.append(refined_rationales2[i]); rejected.append(refined_rationales1[i]); response.append(responses[i])
                    else:
                        continue

        else:

            average_likelihoods_turn1_rationale1 = self.get_multimode_average_likelihood_for_response(prompt_rationale1, responses)
            average_likelihoods_turn1_rationale2 = self.get_multimode_average_likelihood_for_response(prompt_rationale2, responses)

            average_likelihoods_turn2_rationale1 = self.get_multimode_average_likelihood_for_response(prompt_refined_rationale1, responses)
            average_likelihoods_turn2_rationale2 = self.get_multimode_average_likelihood_for_response(prompt_refined_rationale2, responses)

            prompt, chosen, rejected, response = [], [], [], []
            for i, response_type in enumerate(response_types):

                if response_type == "rationale":
                    if average_likelihoods_turn1_rationale1[i] >= average_likelihoods_turn1_rationale2[i]:
                        prompt.append(self.get_seed_prompt("rationale").format(instruction=instructions[i]))
                        chosen.append(rationales1[i]); rejected.append(rationales2[i]); response.append(responses[i])
                    elif average_likelihoods_turn1_rationale2[i] > average_likelihoods_turn1_rationale1[i]:
                        prompt.append(self.get_seed_prompt("rationale").format(instruction=instructions[i]))
                        chosen.append(rationales2[i]); rejected.append(rationales1[i]); response.append(responses[i])

                elif response_type == "rationale_refinement":
                    if average_likelihoods_turn2_rationale1[i] >= average_likelihoods_turn2_rationale2[i]:
                        prompt.append(self.get_seed_prompt("rationale_refinement").format(instruction=instructions[i], rationale=rationales1[i] if self.args.guide_index == 1 else rationales2[i]))
                        chosen.append(refined_rationales1[i]); rejected.append(refined_rationales2[i]); response.append(responses[i])
                    elif average_likelihoods_turn2_rationale2[i] > average_likelihoods_turn2_rationale1[i]:
                        prompt.append(self.get_seed_prompt("rationale_refinement").format(instruction=instructions[i], rationale=rationales1[i] if self.args.guide_index == 1 else rationales2[i]))
                        chosen.append(refined_rationales2[i]); rejected.append(refined_rationales1[i]); response.append(responses[i])


            
        return prompt, chosen, rejected, response

    def validation_step(self, batch, batch_idx):
        instructions, responses, rationales1, rationales2, refined_rationales1, refined_rationales2, response_types = \
            batch["instructions"], batch["responses"], batch["rationales1"], batch["rationales2"], batch["refined_rationales1"], batch["refined_rationales2"], batch["response_types"]

        prompt, chosen, rejected, response = self.forward(instructions, responses, rationales1, rationales2, refined_rationales1, refined_rationales2, response_types)

        self.prompt.extend(prompt); self.chosen.extend(chosen); self.rejected.extend(rejected); self.response.extend(response)

    def on_validation_epoch_end(self) -> None:

        dataset = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected, "response": self.response}
        with open(f"{self.cache_path}_{self.device}.json", "w") as f:
            json.dump(dataset, f, indent=4)
            