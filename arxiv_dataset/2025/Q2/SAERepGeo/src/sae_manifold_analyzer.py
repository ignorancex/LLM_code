import torch
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class SAEManifoldAnalyzer:
    def __init__(self, model, sae, target_hook: str, model_name: str):
        self.model = model.to(device)
        self.sae = sae.to(device)
        self.target_hook = target_hook
        self.model_name = model_name
        self.device = device
        self.d_model = model.cfg.d_model
        self.d_sae = sae.cfg.d_sae
        logger.info(f"{model_name}: hook={self.target_hook}, d_in={self.d_model}, d_sae={self.d_sae}")

    def extract_common_activations(self, tokens_list: List[torch.Tensor], concept_name: str, activation_threshold=0.01, batch_size=10):
        all_resid_posts = []
        all_latents = []
        all_masks = []

        if len(tokens_list) < 1:
            logger.error(f"error：tokens_list is empty。")
            return None, None, None

        concept_filter_tokens = {
            "months": ["spring", "summer", "fall", "winter", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
            "days": ["one", "two", "three", "four", "five", "six", "seven"],
            "elements": ["period", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "life", "air"],
            "color": ["follows", "one", "two", "three", "four", "five", "six", "seven"],
            "planets": ["small", "blue", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"],
            "number": ["after", "succeeds", "lucky"],
            "alphabet": ["succeeds", "follows", "letter", "Letter"],
            "phonetic_symbol": ["symbol", "Symbol", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"],
            "constellations": ["sign", "earth", "air", "symbolizes", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
        }
        common_filter_tokens = [
            '<|endoftext|>', '<pad>', '<bos>', 'The', 'in', 'at', 'id', '.', 'sign',
            'element', 'Element', 'the', 'Sign', 'comes', 'after', 'follows', 'on', 'an', 'of', 'ends'
        ]
        filter_tokens = set(common_filter_tokens)
        if concept_name in concept_filter_tokens:
            filter_tokens.update(concept_filter_tokens[concept_name])

        for tokens in tokens_list:
            tokens = tokens.to(self.device)
            num_prompts = tokens.shape[0]
            for i in range(0, num_prompts, batch_size):
                batch_tokens = tokens[i:i+batch_size]
                self.model.reset_hooks()
                self.sae.reset_hooks()
                self.model.add_sae(self.sae)
                _, cache = self.model.run_with_cache(batch_tokens, names_filter=[self.target_hook + ".hook_sae_input"])
                resid_post = cache[self.target_hook + ".hook_sae_input"].to(torch.float32)
                latents = self.sae.encode(resid_post)

                tokenizer = self.model.tokenizer
                mask = torch.ones_like(batch_tokens, dtype=torch.bool, device=self.device)
                for token in filter_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None:
                        mask &= (batch_tokens != token_id)
                all_resid_posts.append(resid_post)
                all_latents.append(latents)
                all_masks.append(mask)

        all_resid_posts = torch.cat(all_resid_posts, dim=0) if all_resid_posts else None
        all_latents = torch.cat(all_latents, dim=0) if all_latents else None
        all_masks = torch.cat(all_masks, dim=0) if all_masks else None

        all_latents_flat = torch.cat([latents[mask].flatten() for latents, mask in zip([all_latents], [all_masks])])
        if len(all_latents_flat) > 0:
            activation_threshold = torch.quantile(torch.abs(all_latents_flat), 0.5).item()
            activation_threshold = max(activation_threshold, 0.01)
            logger.info(f"{self.model_name}: dynamic activation threshold {activation_threshold:.6f}")
            self.activation_threshold = activation_threshold

        activation_masks = []
        latents_filtered = all_latents.clone()
        latents_filtered[~all_masks] = 0
        activation_masks.append(torch.abs(latents_filtered) > activation_threshold)

        active_count = torch.zeros(self.d_sae, device=self.device, dtype=torch.long)
        for mask in activation_masks:
            active_per_prompt = (mask.sum(dim=1) > 0).long()
            active_count += (active_per_prompt.sum(dim=0) > 0).long()

        common_active = active_count >= 1
        common_indices = torch.where(common_active)[0]

        if len(common_indices) == 0:
            logger.warning(f"Warning：{self.model_name} Features not found activation in at least 1 prompt")
            return None, None, None

        logger.info(f"{self.model_name}: Extracted {len(common_indices)}  activation features")
        return all_resid_posts, common_indices, all_masks