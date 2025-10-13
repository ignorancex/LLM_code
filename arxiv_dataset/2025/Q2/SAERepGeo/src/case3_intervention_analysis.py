import os
import torch
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import ot
from sae_manifold_analyzer import SAEManifoldAnalyzer
from sae_lens import HookedSAETransformer, SAE

logger = logging.getLogger(__name__)


class CustomSAE(SAE):
    def run_time_activation_ln_out(self, acts):
        return acts

    def decode(self, latents):
        feature_acts = self.run_time_activation_ln_out(latents)
        return self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec


def run_intervention_analysis(model_configs, case2_3_concepts, case1_output_dir, case2_output_dir, case3_output_dir,
                              tokens_dict, lambda_mse=1.0, device='cpu'):
    os.makedirs(case3_output_dir, exist_ok=True)

    analysis_results = []
    csv_file = os.path.join(case3_output_dir, "intervention_analysis_results_icd_monge.csv")
    if os.path.exists(csv_file):
        try:
            if os.path.getsize(csv_file) > 0:
                existing_df = pd.read_csv(csv_file)
                analysis_results = existing_df.to_dict('records')
                logger.info(f"Loaded existing CSV file: {csv_file}")
            else:
                logger.warning(f"CSV file {csv_file} is empty, creating new file")
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file {csv_file} is empty or invalid, creating new file")
    else:
        logger.info(f"No existing CSV file found, creating new file")

    results_file = os.path.join(case1_output_dir, 'results_zero_noise.pkl')
    tokens_dict_file = os.path.join(case1_output_dir, 'tokens_dict_zero_noise.pkl')
    residual_cache_file = os.path.join(case2_output_dir, 'residual_cache.pkl')

    if not (os.path.exists(results_file) and os.path.exists(tokens_dict_file) and os.path.exists(residual_cache_file)):
        logger.error("Cache files missing, please run generate_case1_caches and generate_case2_cache first")
        return analysis_results

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    with open(tokens_dict_file, 'rb') as f:
        tokens_dict = pickle.load(f)
    with open(residual_cache_file, 'rb') as f:
        residual_cache = pickle.load(f)

    logger.info(f"Successfully loaded cache: {results_file}, {tokens_dict_file}, {residual_cache_file}")

    def compute_inter_cluster_distance(latents: np.ndarray, labels: np.ndarray, unique_labels: List[int]) -> float:
        if len(unique_labels) < 2:
            return 0.0
        centers = np.array([np.mean(latents[labels == l], axis=0) for l in unique_labels])
        dist_matrix = ot.dist(centers, centers, metric='euclidean')
        n = len(unique_labels)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        distances = dist_matrix[mask]
        if len(distances) == 0:
            return 0.0
        mean_distance = np.mean(distances)
        return mean_distance

    def compute_reconstruction_loss(original_resid, intervened_latents, sae, mask, device):
        intervened_latents = torch.tensor(intervened_latents, dtype=torch.float32, device=device)
        intervened_latents = intervened_latents.reshape(original_resid.shape[0], original_resid.shape[1], -1)
        intervened_latents = (intervened_latents - intervened_latents.mean(dim=-1, keepdim=True)) / (
                    intervened_latents.std(dim=-1, keepdim=True) + 1e-8)
        original_resid = (original_resid - original_resid.mean(dim=-1, keepdim=True)) / (
                    original_resid.std(dim=-1, keepdim=True) + 1e-8)

        with torch.no_grad():
            recon_resid = sae.decode(intervened_latents)

        mask = mask.unsqueeze(-1).expand_as(recon_resid)
        recon_resid_masked = recon_resid[mask]
        original_resid_masked = original_resid[mask]

        mse_loss = torch.mean(
            (recon_resid_masked - original_resid_masked) ** 2).item() if recon_resid_masked.numel() > 0 else 0.0
        return mse_loss

    def intervene_gromov_monge(latents, labels, sae, resid_post, mask, analyzer, alpha=1.0, n_iterations=10,
                               lambda_mse=1.0, device='cpu'):
        latents_flat = latents.copy()
        unique_labels = sorted(set(labels) - {-1})

        if len(unique_labels) < 2:
            logger.warning("Fewer than two valid clusters, unable to perform intervention")
            return latents_flat, 0.0, 0.0, 0.0, 0.0

        original_inter_cluster_dist = compute_inter_cluster_distance(latents_flat, labels, unique_labels)
        logger.info(f"Original Inter-Cluster Distance: {original_inter_cluster_dist:.4f}")

        activation_threshold = analyzer.activation_threshold if hasattr(analyzer, 'activation_threshold') else 0.001
        logger.info(f"Using activation threshold: {activation_threshold:.6f}")

        initial_active_count = np.sum(np.any(np.abs(latents_flat) > activation_threshold, axis=0))
        logger.info(f"Initial Active Features: {initial_active_count}")

        centers = np.array([np.mean(latents_flat[labels == l], axis=0) for l in unique_labels])
        new_centers = centers.copy()
        best_latents_dgw = latents_flat.copy()
        best_latents_aedp = latents_flat.copy()
        best_loss_dgw = float('inf')
        best_loss_aedp = float('inf')
        best_inter_cluster_dist_dgw = original_inter_cluster_dist
        best_inter_cluster_dist_aedp = original_inter_cluster_dist

        for iteration in range(n_iterations):
            intervened_latents = latents_flat.copy()
            for l, center in zip(unique_labels, new_centers):
                mask_cluster = labels == l
                intervened_latents[mask_cluster] += (center - np.mean(latents_flat[mask_cluster], axis=0))

            current_active_count = np.sum(np.any(np.abs(intervened_latents) > activation_threshold, axis=0))

            inter_cluster_dist = compute_inter_cluster_distance(intervened_latents, labels, unique_labels)

            icd_loss_dgw = 1.0 / (inter_cluster_dist + 1e-8)
            icd_loss_aedp = inter_cluster_dist

            mse_loss = compute_reconstruction_loss(resid_post, intervened_latents, sae, mask, device)

            total_loss_dgw = icd_loss_dgw + lambda_mse * mse_loss
            total_loss_aedp = icd_loss_aedp + lambda_mse * mse_loss

            if total_loss_dgw < best_loss_dgw:
                best_loss_dgw = total_loss_dgw
                best_latents_dgw = intervened_latents.copy()
                best_inter_cluster_dist_dgw = inter_cluster_dist

            if total_loss_aedp < best_loss_aedp:
                best_loss_aedp = total_loss_aedp
                best_latents_aedp = intervened_latents.copy()
                best_inter_cluster_dist_aedp = inter_cluster_dist

            grad = np.random.randn(*centers.shape) * 0.01
            new_centers += alpha * grad

            logger.info(
                f"Iteration {iteration + 1}/{n_iterations}: Active Features: {current_active_count}, d_GW Loss={icd_loss_dgw:.4f}, AEDP^-1 Loss={icd_loss_aedp:.4f}, MSE={mse_loss:.4f}, Total_dGW={total_loss_dgw:.4f}, Total_AEDP={total_loss_aedp:.4f}, Inter-Cluster Distance={inter_cluster_dist:.4f}")

        logger.info(
            f"Best Inter-Cluster Distance (d_GW): {best_inter_cluster_dist_dgw:.4f}, (AEDP^-1): {best_inter_cluster_dist_aedp:.4f} (Original: {original_inter_cluster_dist:.4f})")
        return best_latents_dgw, best_latents_aedp, original_inter_cluster_dist, best_inter_cluster_dist_dgw, best_inter_cluster_dist_aedp

    scale_factors = [0.5, 0.8, 1.0, 1.2, 1.5]

    for concept_name in case2_3_concepts:
        for model_name in model_configs.keys():
            try:
                if concept_name not in results or model_name not in results[concept_name] or 0.0 not in \
                        results[concept_name][model_name]:
                    logger.warning(f"Results missing for {model_name} - {concept_name}")
                    continue

                latents = results[concept_name][model_name][0.0]['latents']
                if latents is None:
                    logger.warning(f"Skipping {model_name} - {concept_name}: No latent representations")
                    continue

                latents_flat = latents.reshape(-1, latents.shape[-1]).cpu().numpy()

                labels_file = os.path.join(case2_output_dir,
                                           f"clusters_{model_name.replace('/', '_')}_{concept_name}_latents.npy")
                if not os.path.exists(labels_file):
                    logger.warning(f"Skipping {model_name} - {concept_name}: No cluster labels file {labels_file}")
                    continue
                labels = np.load(labels_file)

                if concept_name not in residual_cache or model_name not in residual_cache[concept_name]:
                    logger.warning(f"Skipping {model_name} - {concept_name}: No residual representations")
                    continue
                resid_post = residual_cache[concept_name][model_name].to(device)

                config = model_configs[model_name]
                model = HookedSAETransformer.from_pretrained(config["model_name"]).to(device)
                if model_name == "GPT2-Small Layer 11":
                    sae = CustomSAE.from_pretrained(config["release"], config["sae_id"], device=device)[0]
                else:
                    sae = SAE.from_pretrained(config["release"], config["sae_id"], device=device)[0]

                analyzer = SAEManifoldAnalyzer(model, sae, config["hook"], model_name)
                logger.info(f"Calling extract_common_activations for {model_name} - {concept_name}")
                _, _, mask = analyzer.extract_common_activations([tokens_dict[concept_name][model_name]], concept_name)
                if mask is None:
                    logger.warning(f"Skipping {model_name} - {concept_name}: Unable to extract mask")
                    continue

                icd_monge_results = {}
                for alpha in scale_factors:
                    best_latents_dgw, best_latents_aedp, original_inter_cluster_dist, best_inter_cluster_dist_dgw, best_inter_cluster_dist_aedp = intervene_gromov_monge(
                        latents_flat, labels, sae, resid_post, mask, analyzer,
                        alpha=alpha, n_iterations=10, lambda_mse=lambda_mse, device=device
                    )
                    mse_loss_dgw = compute_reconstruction_loss(resid_post, best_latents_dgw, sae, mask, device)
                    mse_loss_aedp = compute_reconstruction_loss(resid_post, best_latents_aedp, sae, mask, device)
                    icd_monge_results[alpha] = {
                        'dgw': {
                            'intervened_latents': best_latents_dgw,
                            'mse_loss': mse_loss_dgw,
                            'original_inter_cluster_dist': original_inter_cluster_dist,
                            'best_inter_cluster_dist': best_inter_cluster_dist_dgw
                        },
                        'aedp': {
                            'intervened_latents': best_latents_aedp,
                            'mse_loss': mse_loss_aedp,
                            'original_inter_cluster_dist': original_inter_cluster_dist,
                            'best_inter_cluster_dist': best_inter_cluster_dist_aedp
                        }
                    }

                for alpha in scale_factors:
                    for loss_type in ['d_GW', 'AEDP^-1']:
                        result_key = 'dgw' if loss_type == 'd_GW' else 'aedp'
                        analysis_results.append({
                            'concept': concept_name,
                            'model': model_name,
                            'intervention': 'icd_monge',
                            'loss_type': loss_type,
                            'scale_factor': float(alpha),
                            'mse_loss': float(icd_monge_results[alpha][result_key]['mse_loss']),
                            'original_inter_cluster_dist': float(
                                icd_monge_results[alpha][result_key]['original_inter_cluster_dist']),
                            'best_inter_cluster_dist': float(
                                icd_monge_results[alpha][result_key]['best_inter_cluster_dist'])
                        })

                model = model.cpu()
                sae = sae.cpu()
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Analysis failed for {model_name} - {concept_name}: {e}")
                continue

    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv(csv_file, index=False, mode='w', encoding='utf-8')
    logger.info("\nIntervention analysis results:")
    logger.info(results_df.to_string())

    return analysis_results