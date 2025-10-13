import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class Generator:

    def __init__(self, model, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, device='cuda:0'):

        self.model = model
        self.device = device

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.min_length = min_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.temperature = temperature

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate(self, prompt):

        outputs = self.model.llama_model.generate(
            inputs_embeds=prompt.context_embs[0],
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=self.num_beams,
            do_sample=True,
            min_length=self.min_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        return output_text, output_token.cpu().numpy()
        
    def generates(self, prompt):
        # Assuming `prompt.context_embs` is a list of embeddings for multiple prompts (batch)
        input_embeds = prompt.context_embs  # List of embeddings
        batch_size = len(input_embeds)
        max_len = max([emb.shape[1] for emb in input_embeds])
        emb_dim = input_embeds[0].shape[2]
        dtype = input_embeds[0].dtype
        device = input_embeds[0].device


        output_texts = []
        output_tokens = []

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(input_embeds):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            do_sample=False,
            min_length=self.min_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            # stopping_criteria=stopping_criteria,
        )

        output_texts = []
        output_tokens = []
        for output_token in outputs:
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()

            output_texts.append(output_text)
            output_tokens.append(output_token.cpu().numpy())

        return output_texts, output_tokens

        # for emb in input_embeds:
        #     # Process each embedding one by one
        #     outputs = self.model.llama_model.generate(
        #         inputs_embeds=emb,
        #         max_new_tokens=self.max_new_tokens,
        #         stopping_criteria=self.stopping_criteria,
        #         num_beams=self.num_beams,
        #         do_sample=True,
        #         min_length=self.min_length,
        #         top_p=self.top_p,
        #         repetition_penalty=self.repetition_penalty,
        #         length_penalty=self.length_penalty,
        #         temperature=self.temperature,
        #     )

        #     output_token = outputs[0]
        #     if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        #         output_token = output_token[1:]
        #     if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        #         output_token = output_token[1:]
        #     output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        #     output_text = output_text.split('###')[0]  # remove the stop sign '###'
        #     output_text = output_text.split('Assistant:')[-1].strip()

        #     output_texts.append(output_text)
        #     output_tokens.append(output_token.cpu().numpy())

        # return output_texts, output_tokens