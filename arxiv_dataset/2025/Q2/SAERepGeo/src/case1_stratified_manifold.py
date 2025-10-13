import os
import torch
import pickle
import json
import logging
from sae_manifold_analyzer import SAEManifoldAnalyzer
from sae_lens import HookedSAETransformer, SAE

logger = logging.getLogger(__name__)


def generate_case1_caches(model_configs, concept_prompts, case1_output_dir, device):
    results = {}
    tokens_dict = {}
    noise_levels = [0.0]  # Only zero noise as per notebook

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()

    for concept_name in concept_prompts.keys():
        concept_results = {}
        tokens_dict[concept_name] = {}
        for model_name, config in model_configs.items():
            try:
                logger.info(f"Generating {model_name} - {concept_name}'s Case 1 cache")
                model = HookedSAETransformer.from_pretrained(config["model_name"]).to(device)
                sae = SAE.from_pretrained(config["release"], config["sae_id"], device=device)[0]

                tokens = model.to_tokens(concept_prompts[concept_name], prepend_bos=True).to(device)
                tokens_dict[concept_name][model_name] = tokens

                analyzer = SAEManifoldAnalyzer(model, sae, config["hook"], model_name)
                common_resid, common_indices, mask = analyzer.extract_common_activations([tokens], concept_name,
                                                                                         batch_size=10)

                result = {}
                if common_resid is not None:
                    latents_encoded = sae.encode(common_resid)
                    result[0.0] = {
                        'latents': latents_encoded.clone().cpu(),
                    }
                concept_results[model_name] = result

                model = model.cpu()
                sae = sae.cpu()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Failure to generate {model_name} - {concept_name}: {e}")
                continue
        results[concept_name] = concept_results

    os.makedirs(case1_output_dir, exist_ok=True)
    with open(os.path.join(case1_output_dir, 'results_zero_noise.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(case1_output_dir, 'tokens_dict_zero_noise.pkl'), 'wb') as f:
        pickle.dump(tokens_dict, f)
    logger.info(f"Save Case 1's cache to {case1_output_dir}")
    return results, tokens_dict