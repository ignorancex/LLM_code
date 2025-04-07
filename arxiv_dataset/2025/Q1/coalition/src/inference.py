
import json
import torch
import pytorch_lightning as pl


class InferenceModule(pl.LightningModule):
    def __init__(self, model, tokenizer, generation_mode="rationale", cache_path="", args=None):
        super(InferenceModule, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.generation_mode = generation_mode
        self.cache_path = cache_path
        self.args = args

        self.instructions, self.rationales, self.gt_rationales, self.responses, self.response_types = [], [], [], [], []
        if self.generation_mode == "rationale_refinement":
            self.refined_rationales = []

        if self.generation_mode == "rationale":
            self.seed_prompt = "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate descriptive reasoning on how to derive the correct answer for the instruction such that the descriptive reasoning will be useful to another AI assistant to generate the correct answer.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif self.generation_mode == "response":
            self.seed_prompt = "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction'. Generate the correct answer for the given instruction.\n'Instruction' - {instruction}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif self.generation_mode == "rationale_conditioned_response":
            self.seed_prompt = "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. Analyse the rationale and come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        elif self.generation_mode == "rationale_refinement":
            self.seed_prompt = "You are an AI assistant 'M'. Provide a response to the given instruction denoted by <Task Description>.\n\n[TASK DESCRIPTION STARTS]\n<Task Description>: In this task, you will be given an 'Instruction' and a rationale denoted by 'Rationale'. The 'Rationale may or may not be correct for the given 'Instruction'. 'Analyse the rationale for its correctness, modify the rationale, and provide the correct elaborate descriptive reasoning or 'Rationale' which will be helpful to come up with the correct answer for the given instruction.\n'Instruction' - {instruction}\n'Rationale' - {rationale}\n[TASK DESCRIPTION ENDS]\n\nFor the given <Task Description>, give your response. [MASTER RESPONSE BEGINS]: "
        else:
            raise NotImplementedError("Response generation mode is not implemented yet")


    def forward(self, batch):
        instructions, rationales = batch["instructions"], batch["rationales"]
        if self.generation_mode == "rationale_conditioned_response" or self.generation_mode == "rationale_refinement":
            formatted_instructions = [self.seed_prompt.format(instruction=instruction, rationale=rationale) for idx, (instruction, rationale) in enumerate(zip(instructions, rationales))]
        else:
            formatted_instructions = [self.seed_prompt.format(instruction = instruction) for idx, instruction in enumerate(instructions)]
        tokenized_instructions = self.tokenizer(formatted_instructions, padding = "max_length", 
                                                truncation = self.args.truncation, max_length = self.args.turn1_max_length, 
                                                return_attention_mask = self.args.return_attention_mask,
                                                return_token_type_ids = self.args.return_token_type_ids,
                                                return_tensors = self.args.return_tensors)
        
        rationale_ids = self.model.generate(input_ids = tokenized_instructions["input_ids"].to(self.device), 
                                            attention_mask = tokenized_instructions["attention_mask"].to(self.device),
                                            num_beams = 3, early_stopping = True, max_new_tokens = 256, pad_token_id = self.tokenizer.eos_token_id)

        rationales = [self.tokenizer.decode(output_ids[len(tokenized_instructions["input_ids"][0]) - 1:], skip_special_tokens=True).split(self.tokenizer.eos_token)[0].strip() for output_ids in rationale_ids]
        
        return rationales

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        instructions, rationales, gt_rationales, responses, response_types = batch["instructions"], batch["rationales"], batch["gt_rationales"], batch["responses"], batch["response_types"]
        refined_rationales = self.forward(batch)

        self.instructions.extend(instructions)
        self.rationales.extend(rationales)
        self.gt_rationales.extend(gt_rationales)
        self.responses.extend(responses)
        self.response_types.extend(response_types)

        if self.generation_mode == "rationale_refinement":
            self.refined_rationales.extend(refined_rationales)
            

    def on_predict_epoch_end(self):
        if self.generation_mode == "rationale_refinement":
            dataset = {"instructions": self.instructions, "responses": self.responses, "gt_rationales": self.gt_rationales, 
                   "rationales": self.rationales, "refined_rationales": self.refined_rationales, "response_types": self.response_types}
        else:
            dataset = {"instructions": self.instructions, "responses": self.responses, "gt_rationales": self.gt_rationales, 
                   "rationales": self.rationales, "response_types": self.response_types}
            
        with open(f"{self.cache_path}_{self.device}.json", "w") as f:
            json.dump(dataset, f) 


    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr = 1e-6)