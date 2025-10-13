from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import numpy as np
import numpy as np
from soar.llm_utils import  check_resp_ok, check_one_out





def filter_invalid_grid(data,data2test):
    data_filtered={}
    for k in tqdm(data.keys()):
        data_filtered[k]=[]
        for dic in data[k]:
            try:
                if check_resp_ok(dic,data2test[k]):
                    dic["predicted_test_output"] = convert_grids_to_list(dic["predicted_test_output"])
                    dic["predicted_train_output"] = convert_grids_to_list(dic["predicted_train_output"])
                    data_filtered[k].append(dic)
            except Exception as e:
                print(f"Error checking response for key {k}: {e}")       
            
    return data_filtered


def filter_invalid_grid_soft(data):
    data_filtered={}
    for k in tqdm(data.keys()):
        data_filtered[k]=[]
        for dic in data[k]:
            all_grids = dic["predicted_test_output"] + dic["predicted_train_output"]

            try:
                if all(check_one_out(grid) for grid in all_grids):
                    data_filtered[k].append(dic)
            except:
                pass
    return data_filtered


def convert_grid_to_list(grid):
    grid = np.array(grid)
    grid = grid.astype(int)
    return grid.tolist()

def convert_grids_to_list(list_grids):
    return [convert_grid_to_list(grid) for grid in list_grids]


def rm_duplicate_resps(model, list_resp, threshold = 0.9):
    if len(list_resp) == 0:
        return list_resp
    list_code = [resp["code"].strip() for resp in list_resp]
    list_feat = model.encode(list_code,batch_size=1,transfer_to_cpu=True,convert_to_numpy=True)#encode_in_batches(model, list_code)
    # list_feat = model.encode(list_code)
    # Normalize the vectors (required for cosine similarity)
    normalized_embeddings = list_feat / np.linalg.norm(list_feat, axis=1, keepdims=True)

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    duplicate_indices = []

    # Only check each pair once (upper triangle)
    for i in range(len(cosine_similarity_matrix)):
        if i in duplicate_indices:
            continue
        for j in range(i+1, len(cosine_similarity_matrix)):

            if cosine_similarity_matrix[i, j] > threshold:
                # Keep the first document, mark the second as duplicate
                duplicate_indices.append(j)

    # Get unique duplicates
    duplicate_indices = list(set(duplicate_indices))
    unique_idx_resp = [idx for idx in range(len(list_code)) if idx not in duplicate_indices]
    list_resp = [list_resp[i] for i in unique_idx_resp]
    return list_resp

# from concurrent.futures import ProcessPoolExecutor, as_completed
# from functools import partial
# from datasketch import MinHash, MinHashLSH


# class FastDuplicateDetector:
#     def __init__(self, 
#                  similarity_threshold: float = 0.9,
#                  num_perms: int = 128,
#                  shingle_size: int = 12,disable_tqdm=True):
#         self.disable_tqdm=disable_tqdm
#         self.similarity_threshold = similarity_threshold
#         self.num_perms = num_perms
#         self.shingle_size = shingle_size
#         self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perms)
#         self.shingle_cache: Dict[int, Set[str]] = {}  # Cache for shingles
#         self.minhash_cache: Dict[int, MinHash] = {}   # Cache for minhashes

#     @staticmethod
#     def preprocess(string: str) -> str:
#         """Preprocess the input string."""
#         return string.lower().strip()

#     def _shingle(self, string: str) -> Set[str]:
#         """Create shingles from string with caching."""
#         string = self.preprocess(string)
#         shings = {
#             string[i : i + self.shingle_size] 
#             for i in range(len(string) - self.shingle_size + 1)
#         }
#         return shings

#     def create_minhash(self, text: str) -> MinHash:
#         """Create MinHash object with caching."""
#         m = MinHash(num_perm=self.num_perms)
#         shingles = self._shingle(text)
#         # Use numpy for faster processing
#         np_shingles = np.array([s.encode('utf8') for s in shingles])
#         for s in np_shingles:
#             m.update(s)
#         return m

#     def process_codes(self, code_snippets: List[str]) -> Dict[int, List[str]]:
#         """Process all code snippets in batch."""
#         # Pre-compute all shingles and minhashes
#         for idx, code in enumerate(tqdm(code_snippets, desc="Creating MinHashes",disable=self.disable_tqdm)):
#             minhash = self.create_minhash(code)
#             self.minhash_cache[idx] = minhash
#             self.shingle_cache[idx] = self._shingle(code)
#             self.lsh.insert(str(idx), minhash)

#         # Find duplicates efficiently
#         dup_dict = defaultdict(list)
#         for idx in tqdm(range(len(code_snippets)), desc="Finding duplicates",disable=self.disable_tqdm):
#             dups = self.lsh.query(self.minhash_cache[idx])
#             if dups:
#                 dup_dict[idx] = dups

#         return dict(dup_dict)

#     def calculate_similarities(self, 
#                              code_snippets: List[str], 
#                              dup_dict: Dict[int, List[str]]) -> List[float]:
#         """Calculate Jaccard similarities efficiently."""
#         similarities = []
        
#         for id_, dups in tqdm(dup_dict.items(), desc="Calculating similarities",disable=self.disable_tqdm):
#             if dups:
#                 query_shingles = self.shingle_cache[id_]
#                 for dup_id in dups:
#                     indexed_shingles = self.shingle_cache[int(dup_id)]
#                     # Efficient set operations
#                     intersection = len(query_shingles & indexed_shingles)
#                     union = len(query_shingles | indexed_shingles)
#                     sim = intersection / union if union > 0 else 0
#                     similarities.append(sim)
        
#         return similarities

#     def plot_similarities(self, similarities: List[float]) -> None:
#         """Plot histogram of similarities."""
#         plt.figure(figsize=(10, 6))
#         plt.hist(similarities, bins=100, edgecolor='black')
#         plt.title('Distribution of Jaccard Similarities')
#         plt.xlabel('Similarity')
#         plt.ylabel('Frequency')
#         plt.grid(True, alpha=0.3)
#         plt.show()


#     def get_unique_indices(self, 
#                           code_snippets: List[str], 
#                           dup_dict: Dict[int, List[str]], 
#                           similarity_threshold: float = 0.8,
#                           min_keep: int = 100) -> Tuple[List[int], List[Tuple[int, int, float]]]:
#         """
#         Determine which code snippets to keep by removing duplicates.
#         Returns both the indices to keep and the duplicate pairs for reference.
        
#         Args:
#             code_snippets: List of code snippets
#             dup_dict: Dictionary of duplicate relationships
#             similarity_threshold: Minimum similarity to consider as duplicate
#             min_keep: Minimum number of snippets to keep

            
#         Returns:
#             - List of indices to keep
#             - List of tuples (original_idx, duplicate_idx, similarity)
#         """
#         indices_to_remove = set()
#         duplicate_pairs = []
        
#         # Create graph of duplicate relationships
#         for id_, dups in tqdm(dup_dict.items(), desc="Analyzing duplicates",disable=self.disable_tqdm):
#             if dups:
#                 query_shingles = self.shingle_cache[id_]
#                 for dup_id in dups:
#                     dup_id = int(dup_id)
#                     if id_ != dup_id and dup_id not in indices_to_remove:
#                         indexed_shingles = self.shingle_cache[dup_id]
#                         # Calculate similarity
#                         intersection = len(query_shingles & indexed_shingles)
#                         union = len(query_shingles | indexed_shingles)
#                         similarity = intersection / union if union > 0 else 0
                        
#                         if similarity >= similarity_threshold:
#                             # Store duplicate information
#                             duplicate_pairs.append((id_, dup_id, similarity))
                            
#                             # Keep the shorter code (usually more concise)
#                             if len(code_snippets[id_]) <= len(code_snippets[dup_id]):
#                                 indices_to_remove.add(dup_id)
#                             else:
#                                 indices_to_remove.add(id_)
#                                 break  # Skip remaining comparisons for this id_
        
#         # Create list of indices to keep
#         indices_to_keep = [i for i in range(len(code_snippets)) if i not in indices_to_remove]
#         # if not indices_to_keep:
#         #     # Keep the shortest snippet if all would be removed
#         #     shortest_idx = min(range(len(code_snippets)), key=lambda i: len(code_snippets[i]))
#         #     indices_to_keep = [shortest_idx]
#     # Ensure we keep at least min_keep snippets (or all if fewer exist)
#         if len(indices_to_keep) < min(min_keep, len(code_snippets)):
#             # Calculate how many additional indices we need
#             additional_needed = min(min_keep, len(code_snippets)) - len(indices_to_keep)
            
#             # Get indices that were removed and are available to add back
#             removed_indices = list(indices_to_remove)
            
#             # If we need more indices and have removed indices available
#             if removed_indices:
#                 # Randomly select additional indices from the removed ones
#                 import random
#                 additional_indices = random.sample(removed_indices, 
#                                                 min(additional_needed, len(removed_indices)))
#                 indices_to_keep.extend(additional_indices)
                
#                 # Remove the restored indices from indices_to_remove
#                 indices_to_remove.difference_update(additional_indices)

#         return indices_to_keep, duplicate_pairs

#     def print_duplicate_summary(self, 
#                               code_snippets: List[str], 
#                               indices_to_keep: List[int], 
#                               duplicate_pairs: List[Tuple[int, int, float]]) -> None:
#         """
#         Print a summary of the deduplication results.
#         """
#         print("\n=== Deduplication Summary ===")
#         print(f"Original number of snippets: {len(code_snippets)}")
#         print(f"Number of snippets to keep: {len(indices_to_keep)}")
#         print(f"Number of duplicates removed: {len(code_snippets) - len(indices_to_keep)}")
        
#         print("\n=== Sample Duplicate Pairs ===")
#         for orig_idx, dup_idx, similarity in sorted(duplicate_pairs, key=lambda x: x[2], reverse=True)[:5]:
#             print(f"\nSimilarity: {similarity:.4f}")
#             print(f"Original (Index {orig_idx}):")
#             print(code_snippets[orig_idx][:100] + "..." if len(code_snippets[orig_idx]) > 100 else code_snippets[orig_idx])
#             print(f"Duplicate (Index {dup_idx}):")
#             print(code_snippets[dup_idx][:100] + "..." if len(code_snippets[dup_idx]) > 100 else code_snippets[dup_idx])
#             print("-" * 50)


#     def compute_similarity_matrix(self, code_snippets: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
#         """
#         Compute full similarity matrix between all code snippets.
        
#         Returns:
#             - similarity_matrix: numpy array with pairwise similarities
#             - similar_pairs: list of tuples (i, j, similarity) where similarity > threshold
#         """
#         n = len(code_snippets)
#         similarity_matrix = np.zeros((n, n))
#         similar_pairs = []

#         # Pre-compute all minhashes and shingles if not already cached
#         for idx, code in enumerate(tqdm(code_snippets, desc="Computing hashes",disable=self.disable_tqdm)):
#             if idx not in self.minhash_cache:
#                 self.minhash_cache[idx] = self.create_minhash(code)
#             if idx not in self.shingle_cache:
#                 self.shingle_cache[idx] = self._shingle(code)

#         # Compute full similarity matrix
#         for i in tqdm(range(n), desc="Computing similarity matrix",disable=self.disable_tqdm):
#             # Diagonal elements are 1 (self-similarity)
#             similarity_matrix[i, i] = 1.0
            
#             # Compute upper triangle of matrix
#             for j in range(i + 1, n):
#                 # First use MinHash estimation for quick filtering
#                 minhash_sim = self.minhash_cache[i].jaccard(self.minhash_cache[j])
                
#                 # If MinHash similarity is above threshold, compute exact Jaccard
#                 if minhash_sim >= self.similarity_threshold:
#                     # Compute exact Jaccard similarity
#                     intersection = len(self.shingle_cache[i] & self.shingle_cache[j])
#                     union = len(self.shingle_cache[i] | self.shingle_cache[j])
#                     exact_sim = intersection / union if union > 0 else 0
                    
#                     similarity_matrix[i, j] = exact_sim
#                     similarity_matrix[j, i] = exact_sim  # Matrix is symmetric
                    
#                     if exact_sim >= self.similarity_threshold:
#                         similar_pairs.append((i, j, exact_sim))
#                 else:
#                     similarity_matrix[i, j] = minhash_sim
#                     similarity_matrix[j, i] = minhash_sim

#         return similarity_matrix, similar_pairs

#     def plot_similarity_matrix(self, 
#                              similarity_matrix: np.ndarray, 
#                              figsize: Tuple[int, int] = (12, 10),
#                              snippet_length: int = 30) -> None:
#         """
#         Plot the similarity matrix as a heatmap.
        
#         Args:
#             similarity_matrix: The computed similarity matrix
#             figsize: Size of the figure (width, height)
#             snippet_length: Length of code snippets to show in labels
#         """
#         plt.figure(figsize=figsize)
        
#         # Create labels (truncated code snippets)
#         # labels = [f"#{i}: {code[:snippet_length]}..." 
#         #          for i, code in enumerate(code_snippets)]
        
#         # Create heatmap
#         sns.heatmap(similarity_matrix, 
#                 #    xticklabels=labels,
#                 #    yticklabels=labels,
#                    cmap='YlOrRd',
#                    vmin=0, 
#                    vmax=1)
        
#         plt.title('Code Similarity Matrix')
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plt.show()

#     def print_similarity_analysis(self, 
#                                 similar_pairs: List[Tuple[int, int, float]], 
#                                 code_snippets: List[str]) -> None:
#         """
#         Print detailed analysis of similar code pairs.
#         """
#         print("\n=== Similarity Analysis ===")
#         print(f"Found {len(similar_pairs)} similar pairs above threshold {self.similarity_threshold}")
        
#         # Sort by similarity
#         sorted_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)
        
#         print("\nTop similar pairs:")
#         for i, j, sim in sorted_pairs[:5]:  # Show top 5 pairs
#             print(f"\nSimilarity: {sim:.4f}")
#             print(f"Snippet #{i}:")
#             print(code_snippets[i][:100] + "..." if len(code_snippets[i]) > 100 else code_snippets[i])
#             print(f"\nSnippet #{j}:")
#             print(code_snippets[j][:100] + "..." if len(code_snippets[j]) > 100 else code_snippets[j])
#             print("-" * 50)
# def remove_code_duplicate(code_snippets,similarity_threshold=0.8):
#     num_perms = 128 
#     if similarity_threshold >= 0.96:
#         # Increase number of permutations bug otherwise 
#         num_perms = 768 
#     detector = FastDuplicateDetector(shingle_size=4,similarity_threshold=min(0.7,similarity_threshold), num_perms = num_perms)
#     dup_dict = detector.process_codes(code_snippets)
#     indices_to_keep, _ = detector.get_unique_indices(
#         code_snippets,
#         dup_dict,
#         similarity_threshold=similarity_threshold
#     )
#     return indices_to_keep



# def remove_code_duplicate_multiprocessing(data,similarity_threshold=0.8,max_workers=16):
#     deduplication_data={}
#     remove_code_duplicate_ = partial(remove_code_duplicate,similarity_threshold=similarity_threshold)
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
#         futures = {}
#         # Only pass the code snippets list to the process
#         for k in data:
#             code_snippets = [i["code"] for i in data[k]]
#             futures[executor.submit(remove_code_duplicate_, code_snippets)] = k
        
#         for future in as_completed(futures):
#             k = futures[future]
#             indices_to_keep = future.result()
#             deduplication_data[k] = [data[k][i] for i in indices_to_keep]
#     return deduplication_data

# def remove_code_duplicate_monoprocessing(data,similarity_threshold=0.8):

#     deduplication_data={}
#     for k in data:
#         code_snippets = [i["code"] for i in data[k]]
#         indices_to_keep = remove_code_duplicate(code_snippets,similarity_threshold=similarity_threshold)
#         deduplication_data[k] = [data[k][i] for i in indices_to_keep]
#     return deduplication_data