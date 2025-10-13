import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
import torch
from skdim.id import TwoNN
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import procrustes
from sae_manifold_analyzer import SAEManifoldAnalyzer
from sae_lens import HookedSAETransformer, SAE

logger = logging.getLogger(__name__)


def generate_case2_cache(model_configs, case2_3_concepts, concept_prompts, case2_output_dir, device):
    residual_cache = {}
    for concept_name in case2_3_concepts:
        if concept_name not in concept_prompts:
            logger.warning(f"Jump {concept_name}：No such concept in concept_prompts")
            continue
        residual_cache[concept_name] = {}
        for model_name, config in model_configs.items():
            cache_file = os.path.join(case2_output_dir, f"resid_{model_name.replace('/', '_')}_{concept_name}.pt")
            try:
                logger.info(f"Generating {model_name} - {concept_name}'s Case 2 cache")
                model = HookedSAETransformer.from_pretrained(config["model_name"]).to(device)
                sae = SAE.from_pretrained(config["release"], config["sae_id"], device=device)[0]

                tokens = model.to_tokens(concept_prompts[concept_name], prepend_bos=True).to(device)

                analyzer = SAEManifoldAnalyzer(model, sae, config["hook"], model_name)
                common_resid, common_indices, mask = analyzer.extract_common_activations([tokens], concept_name)

                if common_resid is not None:
                    residual_cache[concept_name][model_name] = common_resid.cpu()
                    torch.save(common_resid.cpu(), cache_file)
                    logger.info(f"Save the residual repreentation to {cache_file}")
                else:
                    logger.warning(f"Warning：Can't extract the residual representation of {model_name} - {concept_name}")

                model = model.cpu()
                sae = sae.cpu()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Failure to generate {model_name} - {concept_name} : {e}")
                continue

    os.makedirs(case2_output_dir, exist_ok=True)
    with open(os.path.join(case2_output_dir, 'residual_cache.pkl'), 'wb') as f:
        pickle.dump(residual_cache, f)
    logger.info(f"Save Case 2's cache to {case2_output_dir}/residual_cache.pkl")
    return residual_cache


def analyze_representation_structure(model_configs, case2_3_concepts, case1_output_dir, case2_output_dir):
    results_file = os.path.join(case1_output_dir, 'results_zero_noise.pkl')
    tokens_dict_file = os.path.join(case1_output_dir, 'tokens_dict_zero_noise.pkl')
    residual_cache_file = os.path.join(case2_output_dir, 'residual_cache.pkl')

    if not (os.path.exists(results_file) and os.path.exists(tokens_dict_file) and os.path.exists(residual_cache_file)):
        logger.error("The cache file is missing，please run generate_case1_caches and generate_case2_cache")
        return [], {}

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    with open(tokens_dict_file, 'rb') as f:
        tokens_dict = pickle.load(f)
    with open(residual_cache_file, 'rb') as f:
        residual_cache = pickle.load(f)

    logger.info(f"Successfully loaded the cache: {results_file}, {tokens_dict_file}, {residual_cache_file}")

    analysis_results = []

    def run_hdbscan_clustering(points, min_cluster_size=10):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, n_clusters

    def estimate_intrinsic_dimension(points):
        try:
            id_estimator = TwoNN()
            dim = id_estimator.fit_transform(points)
            return dim
        except:
            return np.nan

    def compute_mst_weight(labels, points):
        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) < 2:
            return np.nan
        centers = np.array([np.mean(points[labels == l], axis=0) for l in unique_labels])
        dist_matrix = euclidean_distances(centers)
        mst = minimum_spanning_tree(dist_matrix).toarray()
        return mst.sum()

    def compute_procrustes_disparity(labels_resid, labels_latents, points_resid, points_latents):
        common_labels = set(labels_resid).intersection(set(labels_latents)) - {-1}
        if len(common_labels) < 2:
            return np.nan
        centers_resid = np.array([np.mean(points_resid[labels_resid == l], axis=0) for l in common_labels])
        centers_latents = np.array([np.mean(points_latents[labels_latents == l], axis=0) for l in common_labels])
        try:
            _, _, disparity = procrustes(centers_resid, centers_latents)
            return disparity
        except:
            return np.nan

    for concept_name in case2_3_concepts:
        for model_name in model_configs.keys():
            try:
                latents = results.get(concept_name, {}).get(model_name, {}).get(0.0, {}).get('latents', None)
                if latents is None:
                    logger.warning(f"Jump {model_name} - {concept_name}：No sparse representation")
                    continue

                resid = residual_cache.get(concept_name, {}).get(model_name, None)
                if resid is None:
                    logger.warning(f"Jump {model_name} - {concept_name}：No residual representation")
                    continue

                resid_flat = resid.reshape(-1, resid.shape[-1]).cpu().numpy()
                latents_flat = latents.reshape(-1, latents.shape[-1]).cpu().numpy()

                scaler = StandardScaler()
                resid_flat = scaler.fit_transform(resid_flat)
                latents_flat = scaler.fit_transform(latents_flat)
                resid_flat = resid_flat / np.linalg.norm(resid_flat, axis=1, keepdims=True)
                latents_flat = latents_flat / np.linalg.norm(latents_flat, axis=1, keepdims=True)

                reducer = UMAP(n_components=50, random_state=42, n_neighbors=15, min_dist=0.1, metric='euclidean')
                resid_reduced = reducer.fit_transform(resid_flat)
                latents_reduced = reducer.fit_transform(latents_flat)

                labels_resid, n_clusters_resid = run_hdbscan_clustering(resid_reduced)
                labels_latents, n_clusters_latents = run_hdbscan_clustering(latents_reduced)

                local_dims_resid = {}
                local_dims_latents = {}
                for label in set(labels_resid) - {-1}:
                    cluster_points = resid_reduced[labels_resid == label]
                    local_dims_resid[label] = estimate_intrinsic_dimension(cluster_points)

                for label in set(labels_latents) - {-1}:
                    cluster_points = latents_reduced[labels_latents == label]
                    local_dims_latents[label] = estimate_intrinsic_dimension(cluster_points)

                mst_weight_resid = compute_mst_weight(labels_resid, resid_reduced)
                mst_weight_latents = compute_mst_weight(labels_latents, latents_reduced)
                procrustes_disparity = compute_procrustes_disparity(labels_resid, labels_latents, resid_reduced,
                                                                    latents_reduced)

                result = {
                    'concept': concept_name,
                    'model': model_name,
                    'n_clusters_resid': n_clusters_resid,
                    'n_clusters_latents': n_clusters_latents,
                    'local_dims_resid': local_dims_resid,
                    'local_dims_latents': local_dims_latents,
                    'mst_weight_resid': mst_weight_resid,
                    'mst_weight_latents': mst_weight_latents,
                    'procrustes_disparity': procrustes_disparity,
                    'labels_resid': labels_resid,
                    'labels_latents': labels_latents
                }
                analysis_results.append(result)

                np.save(
                    os.path.join(case2_output_dir, f"clusters_{model_name.replace('/', '_')}_{concept_name}_resid.npy"),
                    labels_resid)
                np.save(os.path.join(case2_output_dir,
                                     f"clusters_{model_name.replace('/', '_')}_{concept_name}_latents.npy"),
                        labels_latents)

            except Exception as e:
                logger.error(f"Failure to analyze {model_name} - {concept_name} : {e}")
                continue

    results_df = []
    for result in analysis_results:
        avg_dim_resid = np.mean([d for d in result['local_dims_resid'].values() if not np.isnan(d)]) if result[
            'local_dims_resid'] else np.nan
        avg_dim_latents = np.mean([d for d in result['local_dims_latents'].values() if not np.isnan(d)]) if result[
            'local_dims_latents'] else np.nan

        results_df.append({
            'Concept': result['concept'],
            'Model': result['model'],
            'Clusters Resid': int(result['n_clusters_resid']),
            'Clusters Latents': int(result['n_clusters_latents']),
            'Avg Dim Resid': float(avg_dim_resid),
            'Avg Dim Latents': float(avg_dim_latents),
            'MST Weight Resid': float(result['mst_weight_resid']) if not np.isnan(
                result['mst_weight_resid']) else np.nan,
            'MST Weight Latents': float(result['mst_weight_latents']) if not np.isnan(
                result['mst_weight_latents']) else np.nan,
            'Procrustes Disparity': float(result['procrustes_disparity']) if not np.isnan(
                result['procrustes_disparity']) else np.nan
        })

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(os.path.join(case2_output_dir, "clustering_analysis_results.csv"), index=False, encoding='utf-8')
    logger.info("\nClustering and structural analysis results：")
    logger.info(results_df.to_string())

    return analysis_results, residual_cache