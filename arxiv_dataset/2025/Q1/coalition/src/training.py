
import torch
import torch.nn as nn
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam

class WarmupInstructionTuningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, args, **kwargs):
        super(WarmupInstructionTuningModule, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.criterion = nn.CrossEntropyLoss()

        self.seed_prompts = {
            "response": "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate the correct answer for the given instruction.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: ",

            "rationale": "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate descriptive reasoning on how to derive the correct answer for the instruction such that the descriptive reasoning will be useful to another AI assistant to generate the correct answer.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: ",

            "rationale_conditioned_response": "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. Analyse the rationale and come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: ",

            "rationale_refinement": "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. The 'Rationale may or may not be correct for the given 'Instruction'. 'Analyse the rationale for its correctness, modify the rationale, and provide the correct elaborate descriptive reasoning or 'Rationale' which will be helpful to come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        }

    def _get_batch_inputs(self, batch):

        instructions, rationales, gt_rationales, responses, response_types = batch["instructions"], batch["rationales"], batch["gt_rationales"], batch["responses"], batch["response_types"]

        formatted_instructions = [self.seed_prompts[response_type].format(instruction = instruction, rationale = rationale) for idx, (instruction, rationale, response_type) in enumerate(zip(instructions, rationales, response_types))]
        
        formatted_responses = [response if (response_type == "response" or response_type == "rationale_conditioned_response") else gt_rationale for response, gt_rationale, response_type in zip(responses, gt_rationales, response_types)]
        
        tokenized_instructions = self.tokenizer(formatted_instructions, formatted_responses, padding = "max_length", 
                                                truncation = self.args.truncation, max_length = self.args.turn1_max_length, 
                                                return_attention_mask = self.args.return_attention_mask,
                                                return_token_type_ids = self.args.return_token_type_ids,
                                                return_tensors = self.args.return_tensors)
        
        input_ids, attention_mask = tokenized_instructions["input_ids"], tokenized_instructions["attention_mask"]
        labels = self._create_labels(tokenized_instructions)

        return input_ids, attention_mask, labels

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).logits
        return logits

    def training_step(self, batch, batch_idx, *args, **kwargs):
        
        input_ids, attention_mask, labels = self._get_batch_inputs(batch)
        logits = self(input_ids, attention_mask)
        loss = self._compute_loss(logits, labels.to(self.device))
        self.log("train/loss", loss.item(), prog_bar = True, sync_dist = True)

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.learning_rate)
        optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr = self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.args.max_epochs)

        return [optimizer], [scheduler]

    def _compute_loss(self, logits, labels):
        return self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

    def _create_labels(self, tokenized_input):

        input_ids = tokenized_input["input_ids"]
        token_type_ids = tokenized_input["token_type_ids"]
    
        labels = input_ids.new_zeros(input_ids.shape)
        labels[:, :-1] = input_ids[:, 1:].clone()
    
        shifted_token_type_ids = token_type_ids.new_zeros(token_type_ids.shape)
        shifted_token_type_ids[:, :-1] = token_type_ids[:, 1:].clone()
    
        labels.masked_fill_(shifted_token_type_ids != 1, -100)
        return labels