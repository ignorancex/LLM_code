import os
import jax
import jax.numpy as jnp
import numpy as np
import traceback
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class SaeVisData:
    """
    JAX compatible version of SAEDashboard's SaeVisData.
    Stores feature data and statistics for visualization.
    """
    feature_data_dict: Dict[int, Dict] = field(default_factory=dict)
    feature_stats: Dict = field(default_factory=dict)

    def update(self, other):
        """
        Updates data with data from another SaeVisData object

        Handles merging of feature data with special care for synthetic/dummy data
        to ensure real data takes precedence over synthetic data from previous updates.
        """
        if other is None:
            return

        # Handle feature data dictionary updates carefully
        for feature_idx, new_feature_data in other.feature_data_dict.items():
            if feature_idx not in self.feature_data_dict:
                # If this feature doesn't exist yet, just add it
                self.feature_data_dict[feature_idx] = new_feature_data
            else:
                # Feature already exists - need to intelligently merge
                # For sequence data, keep only real data and discard synthetic data
                existing_data = self.feature_data_dict[feature_idx]

                # Check if the existing sequence data appears to be synthetic
                existing_seq_groups = existing_data.get("sequence_data", {}).get("seq_group_data", [])
                new_seq_groups = new_feature_data.get("sequence_data", {}).get("seq_group_data", [])

                # If existing data is synthetic (identified by title) and new data is not,
                # replace it completely; otherwise, use standard update
                is_existing_synthetic = any(
                    "SYNTHETIC DATA" in group.get("title", "")
                    for group in existing_seq_groups
                )
                is_new_synthetic = any(
                    "SYNTHETIC DATA" in group.get("title", "")
                    for group in new_seq_groups
                )

                # If the new data is not synthetic, but the existing is, replace it
                if not is_new_synthetic and is_existing_synthetic:
                    print(f"Replacing synthetic sequence data for feature {feature_idx} with real data")
                    existing_data["sequence_data"] = new_feature_data["sequence_data"]
                # If both are real data, keep the existing and add new non-duplicates
                elif not is_new_synthetic and not is_existing_synthetic:
                    # Both real data - create a merged set of groups
                    print(f"Merging real sequence data for feature {feature_idx}")
                    existing_groups = existing_data.get("sequence_data", {}).get("seq_group_data", [])
                    new_groups = new_feature_data.get("sequence_data", {}).get("seq_group_data", [])

                    # Keep existing groups and add non-duplicate new ones
                    existing_titles = {group.get("title", "") for group in existing_groups}
                    for group in new_groups:
                        if group.get("title", "") not in existing_titles:
                            existing_groups.append(group)

                    existing_data["sequence_data"]["seq_group_data"] = existing_groups

                # For histograms - replace if existing appears to be synthetic
                if "acts_histogram_data" in existing_data and "acts_histogram_data" in new_feature_data:
                    existing_heights = existing_data["acts_histogram_data"].get("bar_heights", [])
                    existing_values = existing_data["acts_histogram_data"].get("bar_values", [])
                    new_heights = new_feature_data["acts_histogram_data"].get("bar_heights", [])
                    new_values = new_feature_data["acts_histogram_data"].get("bar_values", [])

                    # Check if existing histogram data is synthetic (all same values or very small)
                    is_existing_synthetic = (
                        len(existing_heights) == 0 or
                        (len(existing_heights) > 0 and all(h == existing_heights[0] for h in existing_heights)) or
                        (len(existing_values) > 0 and max(existing_values) < 1e-5)
                    )

                    # Check if new histogram data is synthetic
                    is_new_synthetic = (
                        len(new_heights) == 0 or
                        (len(new_heights) > 0 and all(h == new_heights[0] for h in new_heights)) or
                        (len(new_values) > 0 and max(new_values) < 1e-5)
                    )

                    # If existing is synthetic but new is not, replace it
                    if is_existing_synthetic and not is_new_synthetic:
                        print(f"Replacing synthetic histogram data for feature {feature_idx} with real data")
                        existing_data["acts_histogram_data"] = new_feature_data["acts_histogram_data"]

        # Merge feature stats, filter out dummy values (values very close to zero that were added as synthetic data)
        if not self.feature_stats:
            # Even when getting first stats, filter out potential synthetic values
            if "max" in other.feature_stats:
                other.feature_stats["max"] = [
                    v for v in other.feature_stats["max"]
                    if abs(v) > 1e-5  # Filter out tiny synthetic values
                ]
            if "frac_nonzero" in other.feature_stats:
                other.feature_stats["frac_nonzero"] = [
                    v for v in other.feature_stats["frac_nonzero"]
                    if v > 0.01  # Filter out very low densities (likely synthetic)
                ]
            # For quantile data, filter out sets that are all tiny values
            if "quantile_data" in other.feature_stats:
                other.feature_stats["quantile_data"] = [
                    q_set for q_set in other.feature_stats["quantile_data"]
                    if any(abs(q) > 1e-5 for q in q_set)  # Keep if any real values
                ]
            self.feature_stats = other.feature_stats
        else:
            # Filter out synthetic values before merging
            if "max" in other.feature_stats:
                real_max_values = [v for v in other.feature_stats["max"] if abs(v) > 1e-5]
                if real_max_values:  # Only extend if there's real data
                    self.feature_stats["max"].extend(real_max_values)

            if "frac_nonzero" in other.feature_stats:
                real_density_values = [v for v in other.feature_stats["frac_nonzero"] if v > 0.01]
                if real_density_values:  # Only extend if there's real data
                    self.feature_stats["frac_nonzero"].extend(real_density_values)

            # For quantile data, only include non-synthetic values
            if "quantile_data" in other.feature_stats:
                real_quantiles = []
                for quantile_set in other.feature_stats["quantile_data"]:
                    # Check if the quantile data looks real (not all tiny values)
                    if any(abs(q) > 1e-5 for q in quantile_set):
                        real_quantiles.append(quantile_set)

                if real_quantiles:  # Only extend if there's real data
                    self.feature_stats["quantile_data"].extend(real_quantiles)

    def save_json(self, filename: str | Path) -> None:
        """
        Saves data to a JSON file.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.suffix == ".json", "Filename must have a .json extension"

        # Import the adapter conversion function
        try:
            from sae_dashboard_adapter import convert_to_native_python
            # Convert data to native Python types to ensure JSON serialization works
            data_dict = convert_to_native_python(self.to_dict())
        except ImportError:
            print("Warning: sae_dashboard_adapter not found. Trying direct JSON serialization.")
            data_dict = self.to_dict()

        try:
            with open(filename, "w") as f:
                json.dump(data_dict, f)
            print(f"Successfully saved data to {filename}")
        except Exception as e:
            print(f"Error saving data to JSON: {e}")
            raise

    def to_dict(self) -> dict:
        """
        Converts data to a dictionary for serialization
        """
        return {
            "feature_data_dict": self.feature_data_dict,
            "feature_stats": self.feature_stats
        }

class RollingCorrCoef:
    """
    JAX compatible version of SAEDashboard's RollingCorrCoef.
    Tracks correlations between features and neurons.
    """
    def __init__(self, indices=None, with_self=False):
        self.n = 0
        self.X = None
        self.Y = None
        self.indices = indices
        self.with_self = with_self

        # Following are initialized when update is first called
        self.x_sum = None
        self.xy_sum = None
        self.x2_sum = None
        self.y_sum = None
        self.y2_sum = None

    def update(self, x, y):
        """
        Updates correlation statistics with new data

        Args:
            x: Feature activations [n_features, n_samples]
            y: Model activations [n_neurons, n_samples] or feature activations
        """
        # Get values of x and y, and check for consistency
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        X, Nx = x.shape
        Y, Ny = y.shape
        assert Nx == Ny, "Error: x and y should have the same size in the last dimension"

        if self.with_self:
            assert X == Y, "If with_self is True, then x and y should be the same shape"

        if self.X is not None:
            assert X == self.X, "Error: updating with different sized dataset."
        if self.Y is not None:
            assert Y == self.Y, "Error: updating with different sized dataset."

        self.X = X
        self.Y = Y

        # If this is the first update step, initialize the sums
        if self.n == 0:
            self.x_sum = jnp.zeros(X)
            self.xy_sum = jnp.zeros((X, Y))
            self.x2_sum = jnp.zeros(X)
            if not self.with_self:
                self.y_sum = jnp.zeros(Y)
                self.y2_sum = jnp.zeros(Y)

        # Update the sums
        self.n += x.shape[-1]
        self.x_sum = self.x_sum + jnp.sum(x, axis=1)
        self.xy_sum = self.xy_sum + jnp.einsum('in,jn->ij', x, y)
        self.x2_sum = self.x2_sum + jnp.sum(x**2, axis=1)

        if not self.with_self:
            self.y_sum = self.y_sum + jnp.sum(y, axis=1)
            self.y2_sum = self.y2_sum + jnp.sum(y**2, axis=1)

    def corrcoef(self):
        """
        Computes correlation coefficients between x and y

        Returns:
            tuple: (pearson correlation, cosine similarity)
        """
        # Get y_sum and y2_sum (to deal with the cases when with_self is True/False)
        if self.with_self:
            self.y_sum = self.x_sum
            self.y2_sum = self.x2_sum

        # Compute cosine similarity
        cossim_numer = self.xy_sum
        cossim_denom = jnp.sqrt(jnp.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        # Compute pearson correlation
        pearson_numer = self.n * self.xy_sum - jnp.outer(self.x_sum, self.y_sum)
        pearson_denom = jnp.sqrt(
            jnp.outer(
                self.n * self.x2_sum - self.x_sum**2,
                self.n * self.y2_sum - self.y_sum**2
            )
        ) + 1e-6
        pearson = pearson_numer / pearson_denom

        # If with_self, exclude the diagonal
        if self.with_self:
            d = cossim.shape[0]
            mask = jnp.eye(d)
            cossim = cossim * (1 - mask)
            pearson = pearson * (1 - mask)

        return pearson, cossim

    def topk_pearson(self, k, largest=True):
        """
        Gets the top-k pearson correlations

        Args:
            k: Number of top indices to take
            largest: If True, take largest k indices, otherwise smallest

        Returns:
            tuple: (indices, pearson values, cosine similarity values)
        """
        # Get correlation coefficients
        pearson, cossim = self.corrcoef()

        # For each row, find the indices with the top-k pearson values
        if largest:
            topk_indices = jnp.argsort(-pearson, axis=1)[:, :k]
        else:
            topk_indices = jnp.argsort(pearson, axis=1)[:, :k]

        # Get the corresponding pearson and cossim values
        rows = jnp.arange(pearson.shape[0])[:, None]
        topk_pearson_values = pearson[rows, topk_indices]
        topk_cossim_values = cossim[rows, topk_indices]

        # If indices were supplied, use them
        if self.indices is not None:
            adjusted_indices = []
            for row in topk_indices:
                adjusted_indices.append([self.indices[i] for i in row])
            return adjusted_indices, topk_pearson_values.tolist(), topk_cossim_values.tolist()

        return topk_indices.tolist(), topk_pearson_values.tolist(), topk_cossim_values.tolist()

def compute_feature_statistics(data, batch_size=None):
    """
    Calculates statistics for feature activations

    Args:
        data: Feature activations [n_features, n_samples]
        batch_size: Batch size for processing

    Returns:
        dict: Statistics including max, frac_nonzero, and quantile data
    """
    if data is None:
        return {
            "max": [],
            "frac_nonzero": [],
            "quantile_data": [],
            "quantiles": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        }

    # Define quantiles
    quantiles = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    if batch_size is None or batch_size == 0:
        batch_size = data.shape[0]

    n_features = data.shape[0]
    max_values = []
    frac_nonzero = []
    quantile_data = []

    # Process in batches to avoid memory issues
    for i in range(0, n_features, batch_size):
        batch = data[i:min(i + batch_size, n_features)]

        # Calculate max values
        max_values.extend(jnp.max(batch, axis=1).tolist())

        # Calculate fraction of non-zero values
        frac_nonzero.extend(jnp.mean(jnp.abs(batch) > 1e-6, axis=1).tolist())

        # Calculate quantiles
        batch_quantiles = []
        for feat in batch:
            feat_quantiles = [float(jnp.quantile(feat, q)) for q in quantiles]

            # Strip out zeros from beginning
            first_nonzero = next((i for i, x in enumerate(feat_quantiles) if abs(x) > 1e-6), len(feat_quantiles))
            feat_quantiles = feat_quantiles[first_nonzero:]

            # Round values for smaller JSON
            feat_quantiles = [round(q, 6) for q in feat_quantiles]
            batch_quantiles.append(feat_quantiles)

        quantile_data.extend(batch_quantiles)

    return {
        "max": max_values,
        "frac_nonzero": frac_nonzero,
        "quantile_data": quantile_data,
        "quantiles": [round(q, 6) for q in quantiles]
    }

def compute_histogram_data(data, n_bins=40):
    """
    Computes histogram data for visualization

    Args:
        data: 1D array of values
        n_bins: Number of bins for the histogram

    Returns:
        dict: Histogram data including bar heights, values, and tick values
    """
    # Check for empty or constant data
    if data.size == 0:
        # Create synthetic data for empty histograms
        print("Warning: Empty data provided for histogram. Using synthetic values.")
        synthetic_data = jnp.linspace(1e-10, 1e-9, n_bins)
        return {
            "bar_heights": [1] * n_bins,
            "bar_values": synthetic_data.tolist(),
            "tick_vals": [0.0, 2e-10, 4e-10, 6e-10, 8e-10, 1e-9]
        }

    # Check for zeros - replace with tiny values if all zeros
    if jnp.all(data == 0):
        print("Warning: All zeros in histogram data. Using synthetic non-zero values.")
        data = jnp.linspace(1e-10, 1e-9, max(100, data.size))

    # If all values are the same, add a small variation to avoid division by zero
    if jnp.allclose(data, data[0]):
        print("Warning: All values in histogram data are identical. Adding small variation.")
        value = float(data[0])
        # For very small values, use an even smaller variation
        variation = max(0.01 * abs(value), 0.000001)
        data = jnp.concatenate([data, jnp.array([value + variation, value - variation])])

    # Get min and max of data
    max_value = float(jnp.max(data))
    min_value = float(jnp.min(data))

    # Ensure min and max are different
    if max_value == min_value:
        max_value += max(0.001, max_value * 0.01)  # Add 1% or at least 0.001
        min_value -= max(0.001, abs(min_value) * 0.01)  # Subtract 1% or at least 0.001

    # Create bins
    bin_edges = jnp.linspace(min_value, max_value, n_bins + 1)
    bin_size = (max_value - min_value) / n_bins

    # Calculate histogram
    counts, _ = jnp.histogram(data, bins=bin_edges)
    bar_heights = counts.tolist()
    bar_values = [(bin_edges[i] + bin_size / 2) for i in range(n_bins)]

    # For small values, use more precision in rounding
    if max_value < 0.1:
        bar_values = [round(x, 8) for x in bar_values]
    else:
        bar_values = [round(x, 5) for x in bar_values]

    # Create tick values based on the range of the data
    data_range = max_value - min_value

    # For very small ranges, use scientific notation adaptive ticks
    if data_range < 0.001:
        # Find a reasonable scale
        scale = 10**int(jnp.floor(jnp.log10(data_range)))

        # Create 5 ticks spanning the range
        tick_vals = [
            round(min_value, 8),
            round(min_value + 0.25 * data_range, 8),
            round(min_value + 0.5 * data_range, 8),
            round(min_value + 0.75 * data_range, 8),
            round(max_value, 8)
        ]
    else:
        # For normal ranges, create standard ticks
        if max_value > -min_value:
            tickrange = 0.1 * max(int(1e-4 + max_value / (3 * 0.1)), 1) + 1e-6
            num_positive_ticks = 3
            num_negative_ticks = max(1, int(-min_value / tickrange))
        else:
            tickrange = 0.1 * max(int(1e-4 + -min_value / (3 * 0.1)), 1) + 1e-6
            num_negative_ticks = 3
            num_positive_ticks = max(1, int(max_value / tickrange))

        # Create tick values
        neg_ticks = [-tickrange * i for i in range(1, 1 + num_negative_ticks)]
        neg_ticks.reverse()
        tick_vals = neg_ticks + [0] + [tickrange * i for i in range(1, 1 + num_positive_ticks)]

        # Use more precision for small values
        if max(abs(max_value), abs(min_value)) < 0.1:
            tick_vals = [round(t, 8) for t in tick_vals]
        else:
            tick_vals = [round(t, 5) for t in tick_vals]

    return {
        "bar_heights": bar_heights,
        "bar_values": bar_values,
        "tick_vals": tick_vals
    }

def get_logits_table_data(logit_vector, n_rows=10):
    """
    Gets most positively and negatively affected tokens from logit vector

    Args:
        logit_vector: Vector of logit effects
        n_rows: Number of tokens to include

    Returns:
        dict: Token IDs and logit values for most positive and negative effects
    """
    # Make sure we have data to work with
    if logit_vector.size == 0:
        return {
            "top_token_ids": list(range(n_rows)),
            "top_logits": [0.0] * n_rows,
            "bottom_token_ids": list(range(n_rows)),
            "bottom_logits": [0.0] * n_rows
        }

    # Get top and bottom indices
    try:
        top_indices = jnp.argsort(-logit_vector)[:n_rows]
        bottom_indices = jnp.argsort(logit_vector)[:n_rows]

        # Get corresponding values
        top_logits = logit_vector[top_indices]
        bottom_logits = logit_vector[bottom_indices]

        return {
            "top_token_ids": top_indices.tolist(),
            "top_logits": top_logits.tolist(),
            "bottom_token_ids": bottom_indices.tolist(),
            "bottom_logits": bottom_logits.tolist()
        }
    except Exception as e:
        print(f"Warning: Error extracting logits table data: {e}")
        return {
            "top_token_ids": list(range(n_rows)),
            "top_logits": [0.0] * n_rows,
            "bottom_token_ids": list(range(n_rows)),
            "bottom_logits": [0.0] * n_rows
        }

def get_features_table_data(feature_out_dir, corrcoef_neurons, corrcoef_encoder, n_rows=5):
    """
    Gets neuron alignment and correlation data for feature tables

    Args:
        feature_out_dir: Decoder weights for the features
        corrcoef_neurons: Correlations between features and neurons
        corrcoef_encoder: Correlations between features
        n_rows: Number of rows to include in each table

    Returns:
        dict: Data for feature tables
    """
    # Check if feature_out_dir is empty
    if feature_out_dir.size == 0 or feature_out_dir.shape[0] == 0:
        print("Warning: Empty feature output directions. Creating empty data.")
        # Return empty data structure
        return {
            "neuron_alignment_indices": [],
            "neuron_alignment_values": [],
            "neuron_alignment_l1": [],
            "correlated_neurons_indices": [],
            "correlated_neurons_pearson": [],
            "correlated_neurons_cossim": [],
            "correlated_features_indices": [],
            "correlated_features_pearson": [],
            "correlated_features_cossim": [],
            "correlated_b_features_indices": [],
            "correlated_b_features_pearson": [],
            "correlated_b_features_cossim": []
        }

    # Get L1 norms - avoid division by zero
    l1_norms = jnp.linalg.norm(feature_out_dir, ord=1, axis=1)
    # Add small epsilon to avoid division by zero
    l1_fractions = feature_out_dir / (l1_norms[:, None] + 1e-6)

    # Neuron alignment - find which dimensions have largest weights
    neuron_indices = jnp.argsort(-jnp.abs(feature_out_dir), axis=1)[:, :n_rows]

    # Get corresponding values and l1 fractions
    neuron_values = jnp.take_along_axis(feature_out_dir, neuron_indices, axis=1)
    neuron_l1 = jnp.take_along_axis(l1_fractions, neuron_indices, axis=1)

    # Handle correlation data carefully
    try:
        # Get correlated neurons
        neurons_indices, neurons_pearson, neurons_cossim = corrcoef_neurons.topk_pearson(k=n_rows)
    except Exception as e:
        print(f"Warning: Error getting neuron correlations: {e}")
        # Create empty data
        neurons_indices = []
        neurons_pearson = []
        neurons_cossim = []

    try:
        # Get correlated features - ask for n_rows+1 to account for self-correlation
        features_indices, features_pearson, features_cossim = corrcoef_encoder.topk_pearson(k=n_rows + 1)

        # Exclude self-correlation (first entry since it's sorted)
        features_indices = [row[1:] for row in features_indices]
        features_pearson = [row[1:] for row in features_pearson]
        features_cossim = [row[1:] for row in features_cossim]
    except Exception as e:
        print(f"Warning: Error getting feature correlations: {e}")
        # Create empty data
        features_indices = []
        features_pearson = []
        features_cossim = []

    # Convert JAX arrays to lists
    try:
        neuron_indices_list = neuron_indices.tolist()
        neuron_values_list = neuron_values.tolist()
        neuron_l1_list = neuron_l1.tolist()
    except Exception as e:
        print(f"Warning: Error converting neuron alignment to lists: {e}")
        neuron_indices_list = []
        neuron_values_list = []
        neuron_l1_list = []

    return {
        "neuron_alignment_indices": neuron_indices_list,
        "neuron_alignment_values": neuron_values_list,
        "neuron_alignment_l1": neuron_l1_list,
        "correlated_neurons_indices": neurons_indices,
        "correlated_neurons_pearson": neurons_pearson,
        "correlated_neurons_cossim": neurons_cossim,
        "correlated_features_indices": features_indices,
        "correlated_features_pearson": features_pearson,
        "correlated_features_cossim": features_cossim,
        # B-features are not used in our implementation
        "correlated_b_features_indices": [],
        "correlated_b_features_pearson": [],
        "correlated_b_features_cossim": []
    }

def direct_effect_feature_ablation_experiment(
    sae_model,
    partial_model,
    feature_idx,
    feat_acts,
    intermediate_activations,
    feature_resid_dir,
    norms=None
):
    """
    Performs a feature ablation experiment to determine direct effect on logits

    Args:
        sae_model: The MoE SAE model
        partial_model: The partial language model (typically layers 20+ of the full model)
        feature_idx: Index of the feature to ablate
        feat_acts: Feature activation values [batch_size, seq_length]
        intermediate_activations: Original model activations [batch_size, seq_length, input_dim]
        feature_resid_dir: The residual direction for this feature [input_dim]
        norms: Optional normalization factors [batch_size, seq_length]

    Returns:
        contribution_to_logprobs: Difference in log probabilities when feature is ablated
    """
    # Check if there are any non-zero activations
    if not jnp.any(jnp.abs(feat_acts) > 1e-8):
        # Get vocab size from the model if possible, otherwise use a reasonable default
        vocab_size = 50257  # Default size for many models
        return jnp.zeros((*feat_acts.shape, vocab_size))

    # Calculate the feature's contribution to the residual stream
    # The einsum handles both 2D and 3D cases for feat_acts
    if feat_acts.ndim == 2:
        # Shape: [batch_size, seq_length, input_dim]
        feature_contribution = jnp.einsum('bs,d->bsd', feat_acts, feature_resid_dir)
    else:
        # Handle the case where feat_acts is 1D (single sequence)
        # Shape: [batch_size, input_dim]
        feature_contribution = jnp.einsum('b,d->bd', feat_acts, feature_resid_dir)

    # Ablate the feature by subtracting its contribution
    ablated_activations = intermediate_activations - feature_contribution

    # Get logits for original and ablated activations by running them through the partial model
    try:
        original_logits = run_partial_model(partial_model, intermediate_activations, norms)
        ablated_logits = run_partial_model(partial_model, ablated_activations, norms)

        # Calculate difference in log softmax probabilities
        original_log_probs = jax.nn.log_softmax(original_logits, axis=-1)
        ablated_log_probs = jax.nn.log_softmax(ablated_logits, axis=-1)
        contribution_to_logprobs = original_log_probs - ablated_log_probs
    except Exception as e:
        print(f"Warning: Error in ablation experiment: {e}")
        # Return zeros with appropriate shape
        if intermediate_activations.ndim > 2:
            batch_size, seq_length, _ = intermediate_activations.shape
            vocab_size = 50257  # Default size for many models
            contribution_to_logprobs = jnp.zeros((batch_size, seq_length, vocab_size))
        else:
            # Single sequence case
            batch_size, _ = intermediate_activations.shape
            vocab_size = 50257
            contribution_to_logprobs = jnp.zeros((batch_size, vocab_size))

    return contribution_to_logprobs

def get_sequence_data(token_ids, token_strings, feat_acts, qualifying_token_index, original_index=0,
                     loss_contribution=None, top_token_ids=None, top_logits=None,
                     bottom_token_ids=None, bottom_logits=None, token_logits=None):
    """
    Creates sequence data for visualization

    Args:
        token_ids: Array of token IDs
        token_strings: List of token strings
        feat_acts: Feature activation values for each token
        qualifying_token_index: Index of the token that qualified for this sequence
        original_index: Original index in the dataset
        loss_contribution: Optional loss contributions from ablation
        top_token_ids: Optional IDs of top affected tokens
        top_logits: Optional logit values for top affected tokens
        bottom_token_ids: Optional IDs of bottom affected tokens
        bottom_logits: Optional logit values for bottom affected tokens
        token_logits: Optional logit values for each token

    Returns:
        dict: Sequence data for visualization
    """
    # Round feature activations for smaller JSON
    feat_acts_rounded = [round(float(act), 4) for act in feat_acts]

    # Base sequence data
    seq_data = {
        "original_index": int(original_index),
        "qualifying_token_index": int(qualifying_token_index),
        "token_ids": token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids),
        "feat_acts": feat_acts_rounded,
        "token_strings": token_strings if isinstance(token_strings, list) else [str(ts) for ts in token_strings]
    }

    # Add ablation data if available
    if loss_contribution is not None:
        seq_data["loss_contribution"] = [round(float(lc), 4) for lc in loss_contribution]
    else:
        seq_data["loss_contribution"] = []

    if token_logits is not None:
        seq_data["token_logits"] = [round(float(tl), 4) for tl in token_logits]
    else:
        seq_data["token_logits"] = []

    if top_token_ids is not None and top_logits is not None:
        seq_data["top_token_ids"] = top_token_ids
        seq_data["top_logits"] = [[round(float(tl), 4) for tl in tls] for tls in top_logits]
    else:
        seq_data["top_token_ids"] = []
        seq_data["top_logits"] = []

    if bottom_token_ids is not None and bottom_logits is not None:
        seq_data["bottom_token_ids"] = bottom_token_ids
        seq_data["bottom_logits"] = [[round(float(bl), 4) for bl in bls] for bls in bottom_logits]
    else:
        seq_data["bottom_token_ids"] = []
        seq_data["bottom_logits"] = []

    return seq_data

def get_sequence_groups(token_ids, token_strings, feat_acts, buffer_size=5,
                     perform_ablation=False, feature_idx=None, intermediate_activations=None,
                     feature_resid_dir=None, sae_model=None, partial_model=None, norms=None,
                     max_sequences=10, n_quantiles=5):
    """
    Creates sequence groups for visualization, including TOP ACTIVATIONS and quantile groups
    similar to how SAEDashboard does it.

    Args:
        token_ids: Array of token IDs [batch, seq]
        token_strings: Array of token strings [batch, seq] with the same shape as token_ids
        feat_acts: Feature activation values [batch, seq]
        buffer_size: Size of context window around activating tokens
        perform_ablation: Whether to perform ablation experiments
        feature_idx: Index of the feature (needed for ablation)
        intermediate_activations: Original activations (needed for ablation)
        feature_resid_dir: Feature residual direction (needed for ablation)
        sae_model: MoE SAE model (needed for ablation)
        partial_model: Partial language model for forward passes (needed for ablation)
        norms: Optional normalization factors (needed for ablation)
        max_sequences: Maximum number of sequences to generate per group
        n_quantiles: Number of quantile groups to generate

    Returns:
        list: List of sequence group data dictionaries for visualization
    """
    # Check if feat_acts is empty
    if feat_acts.size == 0:
        print("Warning: No feature activations provided. Creating minimal sequence data with non-zero values.")
        # Create synthetic data instead of empty data to prevent visualization crashes
        return [
            {
                "title": "SYNTHETIC DATA (NO ACTIVATIONS)",
                "seq_data": [{
                    "original_index": 0,
                    "qualifying_token_index": 0,
                    "token_ids": [0, 1, 2, 3, 4],
                    "feat_acts": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
                    "token_strings": ["[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"],
                    "loss_contribution": [],
                    "token_logits": [],
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                }]
            }
        ]

    # Find max activating tokens - handle the case where all values are 0
    max_act = float(jnp.max(feat_acts))

    # If max is exactly 0, no activations at all - use a tiny value
    if max_act == 0:
        print(f"Warning: All activations are zero for feature {feature_idx}. Using tiny epsilon value.")
        # Add a small epsilon to all values to avoid zeros - this allows visualization to proceed
        feat_acts = jnp.ones_like(feat_acts) * 1e-10
        max_act = 1e-10  # Set max to the epsilon value

    print(f"Generating sequence groups for feature {feature_idx}, max activation: {max_act:.8f}")

    # For MoE models, we need to be aware that there's an offset activation threshold
    # The threshold is 1.0/sqrt(input_dim) which is about 0.0208 for 2304-dimensional input
    # When working with normalized data, this threshold might never be crossed
    # We'll use a relative threshold based on the max activation instead

    # Get a reasonable threshold - either 5% of max or 0.000001, whichever is larger
    act_threshold = max(max_act * 0.05, 0.000001)
    print(f"Using activation threshold: {act_threshold:.8f}")

    # Check if any activations exceed our threshold
    if jnp.sum(feat_acts > act_threshold) == 0:
        print(f"Warning: No activations above threshold for feature {feature_idx}. Creating minimal sequence data.")
        # Instead of returning empty data, create a single sequence with synthetic data
        # This prevents downstream crashes while allowing visualization to proceed
        return [
            {
                "title": "SYNTHETIC DATA (NO ACTIVATIONS)",
                "seq_data": [{
                    "original_index": 0,
                    "qualifying_token_index": 0,
                    "token_ids": [0, 1, 2, 3, 4],
                    "feat_acts": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
                    "token_strings": ["[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"],
                    "loss_contribution": [],
                    "token_logits": [],
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                }]
            }
        ]

    # Initialize list of sequence groups to return
    sequence_groups = []

    # 1. Create TOP ACTIVATIONS group
    # ---------------------------------
    # Get indices of top activations
    flat_indices = jnp.argsort(-feat_acts.flatten())[:max_sequences]

    try:
        batch_indices, seq_indices = jnp.unravel_index(flat_indices, feat_acts.shape)
    except Exception as e:
        print(f"Warning: Error unraveling indices: {e}")
        return [
            {
                "title": f"ERROR: {str(e)}",
                "seq_data": []
            }
        ]

    top_sequences_data = []

    # For each top activation, extract a sequence with context
    for batch_idx, seq_idx in zip(batch_indices, seq_indices):
        try:
            # Calculate start and end indices for the sequence
            start_idx = max(0, seq_idx - buffer_size)
            end_idx = min(token_ids.shape[1], seq_idx + buffer_size + 1)

            # Extract token IDs and feature activations for the sequence
            seq_token_ids = token_ids[batch_idx, start_idx:end_idx]
            seq_feat_acts = feat_acts[batch_idx, start_idx:end_idx]

            # Extract token strings if available
            if hasattr(token_strings, "shape"):
                seq_token_strings = token_strings[batch_idx, start_idx:end_idx]
            elif callable(token_strings):
                # If token_strings is a function, apply it to token IDs
                seq_token_strings = token_strings(seq_token_ids)
            else:
                # Default to token IDs as strings
                seq_token_strings = [str(tid) for tid in seq_token_ids]

            # Base parameters for sequence data
            seq_data_params = {
                "token_ids": seq_token_ids,
                "token_strings": seq_token_strings,
                "feat_acts": seq_feat_acts,
                "qualifying_token_index": seq_idx - start_idx,  # Adjust for the window
                "original_index": int(batch_idx)
            }

            # If ablation is enabled, compute additional data
            if perform_ablation and partial_model is not None and feature_resid_dir is not None:
                try:
                    # Extract slice of intermediate activations
                    seq_intermediate_acts = intermediate_activations[batch_idx, start_idx:end_idx]

                    # Extract norms if provided
                    seq_norms = None
                    if norms is not None:
                        seq_norms = norms[batch_idx, start_idx:end_idx]

                    # Perform ablation experiment
                    logprob_diffs = direct_effect_feature_ablation_experiment(
                        sae_model,
                        partial_model,
                        feature_idx,
                        seq_feat_acts,
                        seq_intermediate_acts,
                        feature_resid_dir,
                        seq_norms
                    )

                    # Calculate loss contribution (effect on the next token)
                    loss_contribution = []
                    for i in range(logprob_diffs.shape[0] - 1):  # -1 because we're looking at next token
                        if i + 1 < len(seq_token_ids):
                            next_token_id = int(seq_token_ids[i + 1])
                            # Negative because we want decrease in loss to be positive contribution
                            loss_contribution.append(-float(logprob_diffs[i, next_token_id]))

                    # Add final token or pad with zeros
                    while len(loss_contribution) < len(seq_feat_acts):
                        loss_contribution.append(0.0)

                    # For each position, find tokens most affected
                    top_k = 3  # Number of top tokens to track
                    top_token_ids = []
                    top_logits = []
                    bottom_token_ids = []
                    bottom_logits = []

                    for i in range(min(logprob_diffs.shape[0], len(seq_feat_acts))):
                        # Top (most increased) tokens
                        top_indices = jnp.argsort(-logprob_diffs[i])[:top_k]
                        top_values = logprob_diffs[i][top_indices]

                        # Bottom (most decreased) tokens
                        bottom_indices = jnp.argsort(logprob_diffs[i])[:top_k]
                        bottom_values = logprob_diffs[i][bottom_indices]

                        top_token_ids.append(top_indices.tolist())
                        top_logits.append([float(v) for v in top_values])
                        bottom_token_ids.append(bottom_indices.tolist())
                        bottom_logits.append([float(v) for v in bottom_values])

                    # Calculate direct logit effect
                    try:
                        single_feature_dir = jnp.reshape(feature_resid_dir, (1, -1))
                        token_logits = jnp.matmul(single_feature_dir, unembedding_matrix)[0]
                    except Exception:
                        # Create empty token logits
                        token_logits = jnp.zeros(10)

                    # Update sequence data parameters with ablation results
                    seq_data_params.update({
                        "loss_contribution": loss_contribution,
                        "top_token_ids": top_token_ids,
                        "top_logits": top_logits,
                        "bottom_token_ids": bottom_token_ids,
                        "bottom_logits": bottom_logits,
                        "token_logits": token_logits.tolist()
                    })
                except Exception as e:
                    print(f"Warning: Error performing ablation: {e}")
                    # Skip ablation for this sequence

            # Create sequence data
            top_sequences_data.append(get_sequence_data(**seq_data_params))

        except Exception as e:
            print(f"Warning: Error extracting sequence data: {e}")
            # Skip this sequence

    # Create a title that reflects the true magnitude
    max_str = f"{max_act:.6f}" if max_act < 0.01 else f"{max_act:.3f}"

    # Add the TOP ACTIVATIONS group
    sequence_groups.append({
        "title": f"TOP ACTIVATIONS<br>MAX = {max_str}",
        "seq_data": top_sequences_data
    })

    # 2. Create QUANTILE groups if n_quantiles > 0
    # --------------------------------------------
    if n_quantiles > 0:
        # Generate quantile boundaries, similarly to SAEDashboard's sequence_data_generator.py
        # Create linspace from 0 to max activation
        try:
            quantiles = jnp.linspace(0, max_act, n_quantiles + 1)

            # For each quantile range (except the last), create a group
            for i in range(n_quantiles):
                lower, upper = float(quantiles[i]), float(quantiles[i + 1])

                # Skip ranges with no activations
                mask = (feat_acts >= lower) & (feat_acts <= upper)
                num_in_range = jnp.sum(mask)

                # If there are activations in this range, create a quantile group
                if num_in_range > 0:
                    # Calculate percentage of activations in this range
                    pct = float(num_in_range) / feat_acts.size

                    # Get flat indices of activations in this range
                    flat_indices_in_range = jnp.where(mask.flatten())[0]

                    # Take a random sample of up to max_sequences
                    if len(flat_indices_in_range) > max_sequences:
                        # Create a random permutation and take the first max_sequences elements
                        permutation = jax.random.permutation(
                            jax.random.PRNGKey(i), # Use quantile index as seed for reproducibility
                            len(flat_indices_in_range)
                        )[:max_sequences]
                        flat_indices_in_range = flat_indices_in_range[permutation]

                    # Get batch and sequence indices
                    try:
                        batch_indices, seq_indices = jnp.unravel_index(flat_indices_in_range, feat_acts.shape)
                    except Exception as e:
                        print(f"Warning: Error unraveling indices for quantile {i}: {e}")
                        continue

                    # Create sequences for this quantile group
                    quantile_sequences_data = []

                    # Process each sequence in this quantile group
                    for batch_idx, seq_idx in zip(batch_indices, seq_indices):
                        try:
                            # Calculate start and end indices for the sequence
                            start_idx = max(0, seq_idx - buffer_size)
                            end_idx = min(token_ids.shape[1], seq_idx + buffer_size + 1)

                            # Extract token IDs and feature activations for the sequence
                            seq_token_ids = token_ids[batch_idx, start_idx:end_idx]
                            seq_feat_acts = feat_acts[batch_idx, start_idx:end_idx]

                            # Extract token strings if available
                            if hasattr(token_strings, "shape"):
                                seq_token_strings = token_strings[batch_idx, start_idx:end_idx]
                            elif callable(token_strings):
                                # If token_strings is a function, apply it to token IDs
                                seq_token_strings = token_strings(seq_token_ids)
                            else:
                                # Default to token IDs as strings
                                seq_token_strings = [str(tid) for tid in seq_token_ids]

                            # Base parameters for sequence data
                            seq_data_params = {
                                "token_ids": seq_token_ids,
                                "token_strings": seq_token_strings,
                                "feat_acts": seq_feat_acts,
                                "qualifying_token_index": seq_idx - start_idx,  # Adjust for the window
                                "original_index": int(batch_idx)
                            }

                            # If ablation is enabled, compute additional data
                            if perform_ablation and partial_model is not None and feature_resid_dir is not None:
                                try:
                                    # Extract slice of intermediate activations
                                    seq_intermediate_acts = intermediate_activations[batch_idx, start_idx:end_idx]

                                    # Extract norms if provided
                                    seq_norms = None
                                    if norms is not None:
                                        seq_norms = norms[batch_idx, start_idx:end_idx]

                                    # Perform ablation experiment
                                    logprob_diffs = direct_effect_feature_ablation_experiment(
                                        sae_model,
                                        partial_model,
                                        feature_idx,
                                        seq_feat_acts,
                                        seq_intermediate_acts,
                                        feature_resid_dir,
                                        seq_norms
                                    )

                                    # Calculate loss contribution (effect on the next token)
                                    loss_contribution = []
                                    for i in range(logprob_diffs.shape[0] - 1):  # -1 because we're looking at next token
                                        if i + 1 < len(seq_token_ids):
                                            next_token_id = int(seq_token_ids[i + 1])
                                            # Negative because we want decrease in loss to be positive contribution
                                            loss_contribution.append(-float(logprob_diffs[i, next_token_id]))

                                    # Add final token or pad with zeros
                                    while len(loss_contribution) < len(seq_feat_acts):
                                        loss_contribution.append(0.0)

                                    # For each position, find tokens most affected
                                    top_k = 3  # Number of top tokens to track
                                    top_token_ids = []
                                    top_logits = []
                                    bottom_token_ids = []
                                    bottom_logits = []

                                    for j in range(min(logprob_diffs.shape[0], len(seq_feat_acts))):
                                        # Top (most increased) tokens
                                        top_indices = jnp.argsort(-logprob_diffs[j])[:top_k]
                                        top_values = logprob_diffs[j][top_indices]

                                        # Bottom (most decreased) tokens
                                        bottom_indices = jnp.argsort(logprob_diffs[j])[:top_k]
                                        bottom_values = logprob_diffs[j][bottom_indices]

                                        top_token_ids.append(top_indices.tolist())
                                        top_logits.append([float(v) for v in top_values])
                                        bottom_token_ids.append(bottom_indices.tolist())
                                        bottom_logits.append([float(v) for v in bottom_values])

                                    # Calculate direct logit effect
                                    try:
                                        single_feature_dir = jnp.reshape(feature_resid_dir, (1, -1))
                                        token_logits = jnp.matmul(single_feature_dir, unembedding_matrix)[0]
                                    except Exception:
                                        # Create empty token logits
                                        token_logits = jnp.zeros(10)

                                    # Update sequence data parameters with ablation results
                                    seq_data_params.update({
                                        "loss_contribution": loss_contribution,
                                        "top_token_ids": top_token_ids,
                                        "top_logits": top_logits,
                                        "bottom_token_ids": bottom_token_ids,
                                        "bottom_logits": bottom_logits,
                                        "token_logits": token_logits.tolist()
                                    })
                                except Exception as e:
                                    print(f"Warning: Error performing ablation for quantile sequence: {e}")
                                    # Skip ablation for this sequence

                            # Create sequence data
                            quantile_sequences_data.append(get_sequence_data(**seq_data_params))

                        except Exception as e:
                            print(f"Warning: Error extracting quantile sequence data: {e}")
                            # Skip this sequence

                    # Format the title similar to SAEDashboard's format
                    # Use more precision for small values
                    if lower < 0.01 or upper < 0.01:
                        lower_str, upper_str = f"{lower:.6f}", f"{upper:.6f}"
                    else:
                        lower_str, upper_str = f"{lower:.3f}", f"{upper:.3f}"

                    # Add the quantile group
                    sequence_groups.append({
                        "title": f"INTERVAL {lower_str} - {upper_str}<br>CONTAINS {pct:.3%}",
                        "seq_data": quantile_sequences_data
                    })
                else:
                    print(f"Skipping empty quantile range: {lower:.6f} - {upper:.6f}")
        except Exception as e:
            print(f"Error creating quantile groups: {e}")
            # Continue with just the top activations group

    # Return the list of sequence groups
    return sequence_groups

def run_partial_model(model, reconstructed_activations, norms=None):
    """
    Runs the partial model to get logits from reconstructed activations

    Args:
        model: The partial language model (typically layers 20+ of the full model)
        reconstructed_activations: The activations after reconstruction
        norms: Optional pre-computed norms for normalization

    Returns:
        Logits from running the partial model on normalized activations
    """
    # Check if model is None
    if model is None:
        print("Warning: No partial model provided")
        # Return zeros with appropriate shape
        vocab_size = 50257  # Default size for many models
        if reconstructed_activations.ndim > 2:
            return jnp.zeros((*reconstructed_activations.shape[:-1], vocab_size))
        else:
            return jnp.zeros((reconstructed_activations.shape[0], vocab_size))

    # Store original shape for reshaping output
    original_shape = reconstructed_activations.shape
    need_reshape = len(original_shape) > 2

    # If we have batch and sequence dimensions, flatten them for processing
    if need_reshape:
        # Flatten batch and sequence dimensions: [batch, seq, dim] -> [batch*seq, dim]
        batch_size, seq_length, input_dim = original_shape
        reconstructed_activations = reconstructed_activations.reshape(-1, input_dim)

        if norms is not None:
            # Flatten norms too if provided
            norms = norms.reshape(-1)

    # Normalize the reconstructed activations using their standard deviation
    # (this matches the LayerNorm behavior in transformers)
    try:
        normalized_acts = reconstructed_activations / (jnp.std(reconstructed_activations, axis=-1, keepdims=True) + 1e-6)

        # If norms are provided, use them to denormalize the activations
        # This is because the input was already normalized, so we need to "undo" that
        if norms is not None:
            # This is the critical part: multiply by norms to denormalize
            normalized_acts = normalized_acts * norms[..., None]

        # Run the actual partial model on the processed activations
        try:
            # Try to use penzai.nx.wrap if available
            try:
                import penzai
                from penzai import pz
                wrapped_acts = pz.nx.wrap(normalized_acts).tag("batch", "embedding")
                logits = model(wrapped_acts)
            except (ImportError, AttributeError):
                # Fall back to a simpler approach if penzai isn't available
                logits = model(normalized_acts)
        except Exception as e:
            print(f"Warning: Error running partial model: {e}")
            # Create fallback logits
            vocab_size = 50257  # Default size for many models
            if need_reshape:
                logits = jnp.zeros((batch_size * seq_length, vocab_size))
            else:
                logits = jnp.zeros((normalized_acts.shape[0], vocab_size))
    except Exception as e:
        print(f"Warning: Error normalizing activations: {e}")
        # Return zeros with appropriate shape
        vocab_size = 50257  # Default size for many models
        if need_reshape:
            logits = jnp.zeros((batch_size * seq_length, vocab_size))
        else:
            logits = jnp.zeros((reconstructed_activations.shape[0], vocab_size))

    # Reshape output back to original dimensions if needed
    if need_reshape:
        vocab_size = logits.shape[-1]
        logits = logits.reshape(batch_size, seq_length, vocab_size)

    return logits

def init_data(feature_indices=None):
    """
    Initializes the SAE visualization data structure

    Args:
        feature_indices: Optional list of feature indices to pre-initialize

    Returns:
        SaeVisData: Empty data structure for visualization

    Note:
        To save visualizations using SAEDashboard, use the functions in sae_dashboard_adapter.py.
        Example:
            from sae_dashboard_adapter import save_feature_centric_vis
            save_feature_centric_vis(data, "output.html", tokenizer)
    """
    data = SaeVisData()

    # If feature indices are provided, pre-initialize structures
    if feature_indices:
        # Create placeholder feature data
        for idx in feature_indices:
            # Initialize empty feature data structures with the same defaults
            # that SAEDashboard uses
            feature_data = {
                # Empty feature tables data
                "feature_tables_data": {
                    "neuron_alignment_indices": [],
                    "neuron_alignment_values": [],
                    "neuron_alignment_l1": [],
                    "correlated_neurons_indices": [],
                    "correlated_neurons_pearson": [],
                    "correlated_neurons_cossim": [],
                    "correlated_features_indices": [],
                    "correlated_features_pearson": [],
                    "correlated_features_cossim": [],
                    "correlated_b_features_indices": [],
                    "correlated_b_features_pearson": [],
                    "correlated_b_features_cossim": []
                },

                # Empty acts histogram data
                "acts_histogram_data": {
                    "bar_heights": [],
                    "bar_values": [],
                    "tick_vals": [],
                    "title": "ACTIVATIONS<br>DENSITY = 0.000%"
                },

                # Empty logits histogram data
                "logits_histogram_data": {
                    "bar_heights": [],
                    "bar_values": [],
                    "tick_vals": []
                },

                # Empty logits table data
                "logits_table_data": {
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                },

                # Empty sequence data
                "sequence_data": {
                    "seq_group_data": []
                }
            }

            # Add to feature data dictionary
            data.feature_data_dict[idx] = feature_data

        # Initialize feature stats
        data.feature_stats = compute_feature_statistics(None)

    return data

def update_data(data, model, batch, feature_indices, partial_model=None, perform_ablation=False, unembedding_matrix=None):
    """
    Updates SAE visualization data with a new batch

    Args:
        data: SaeVisData object to update
        model: MoE SAE model
        batch: Tuple of (token_ids, token_strings, intermediate_activations, norms) where
               norms is an optional array of normalization factors [batch_size, seq_length]
        feature_indices: List of feature indices to visualize
        partial_model: Optional partial language model (layers 20+) used for ablation
        perform_ablation: Whether to perform feature ablation experiments
        unembedding_matrix: The model's unembedding matrix for projecting to logit space.
                           Shape: [hidden_dim, vocab_size]

    Returns:
        SaeVisData: Updated data structure for visualization
    """
    # Unpack the batch with support for an optional norms component
    if len(batch) == 4:
        token_ids, token_strings, intermediate_activations, norms = batch
    else:
        token_ids, token_strings, intermediate_activations = batch
        norms = None

    # Try to get the unembedding matrix from the global scope if not provided
    if unembedding_matrix is None:
        try:
            # Import from get_logits to get the global unembed_matrix if available
            from get_logits import unembed_matrix
            unembedding_matrix = unembed_matrix
        except (ImportError, AttributeError):
            pass

    # Verify we have what we need for ablation
    if perform_ablation and partial_model is None:
        print("Warning: Ablation requested but no partial_model provided. Disabling ablation.")
        perform_ablation = False

    # Convert to JAX arrays if needed
    token_ids = jnp.array(token_ids) if not isinstance(token_ids, jnp.ndarray) else token_ids
    intermediate_activations = jnp.array(intermediate_activations) if not isinstance(intermediate_activations, jnp.ndarray) else intermediate_activations

    # Step 1: Get sparse codes for the intermediate activations
    # Handle sequence dimension properly
    original_shape = intermediate_activations.shape

    if intermediate_activations.ndim > 2:
        # Extract dimensions
        batch_size, seq_length, input_dim = intermediate_activations.shape

        # Reshape to flatten batch and sequence: [batch_size * seq_length, input_dim]
        flattened_activations = intermediate_activations.reshape(-1, input_dim)

        # Apply vmap over the flattened batch dimension
        sparse_codes_data = jax.vmap(model.encode)(flattened_activations)
        top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = sparse_codes_data

        # Debug output to see if features are being activated
        # Note: The model applies leaky_offset_relu internally, so we need to check for the
        # specific value offset_val = 1.0/sqrt(input_dim) which is about 0.0208 for dim=2304
        offset_val = 1.0/jnp.sqrt(jnp.array(flattened_activations.shape[-1]))
        nonzero_features = jnp.sum(top_level_latent_codes > offset_val)
        max_feature_act = jnp.max(top_level_latent_codes)
        print(f"DEBUG: Top level features - nonzero past threshold {offset_val:.5f}: {nonzero_features}")
        print(f"DEBUG: Top level features - max value: {max_feature_act:.6f}")

        # Reshape results to account for batch and sequence dimensions
        # Will be reshaped again for features later
    else:
        # For 2D input (just batch dimension), apply vmap directly
        sparse_codes_data = jax.vmap(model.encode)(intermediate_activations)
        top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = sparse_codes_data

        # Debug output to see if features are being activated
        # Note: The model applies leaky_offset_relu internally, so we need to check for the
        # specific value offset_val = 1.0/sqrt(input_dim) which is about 0.0208 for dim=2304
        offset_val = 1.0/jnp.sqrt(jnp.array(intermediate_activations.shape[-1]))
        nonzero_features = jnp.sum(top_level_latent_codes > offset_val)
        max_feature_act = jnp.max(top_level_latent_codes)
        print(f"DEBUG: Top level features - nonzero past threshold {offset_val:.5f}: {nonzero_features}")
        print(f"DEBUG: Top level features - max value: {max_feature_act:.6f}")

    # Step 2: Extract feature activations and map them (MoE specific)
    # Map top level feature indices (0 to num_experts-1)
    top_level_indices = [idx for idx in feature_indices if idx < model.num_experts]
    # Map expert level indices (num_experts to num_experts*atoms_per_subspace)
    expert_indices = [idx - model.num_experts for idx in feature_indices
                     if idx >= model.num_experts]

    # Print debug info about which features we're extracting
    print(f"Extracting {len(top_level_indices)} top-level features: {top_level_indices}")
    print(f"Extracting {len(expert_indices)} expert-level features: {expert_indices}")

    # Store correlations for feature statistics
    corr_neurons = RollingCorrCoef()
    corr_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)

    # Create features objects
    if intermediate_activations.ndim > 2:
        batch_size, seq_length, _ = intermediate_activations.shape
    else:
        batch_size = intermediate_activations.shape[0]
        seq_length = 1

    # For the MoE model, we need to track both top-level and expert-level features
    all_feat_acts = []

    # Handle top-level features
    if top_level_indices:
        # For top-level features, extract from top_level_latent_codes
        # NOTE: Make sure indices are within range
        valid_indices = [idx for idx in top_level_indices if idx < top_level_latent_codes.shape[1]]
        if len(valid_indices) != len(top_level_indices):
            print(f"WARNING: Some top-level indices were out of bounds! Using {len(valid_indices)} valid indices.")

        if len(valid_indices) > 0:
            top_level_acts = top_level_latent_codes[:, valid_indices]

            # Reshape if needed (sequence vs batch processing)
            if intermediate_activations.ndim > 2:
                # Reshape from [batch_size*seq_length, n_features] to [batch_size, seq_length, n_features]
                top_level_acts = top_level_acts.reshape(batch_size, seq_length, len(valid_indices))
            else:
                top_level_acts = top_level_acts.reshape(batch_size, 1, len(valid_indices))

            print(f"Top-level acts shape: {top_level_acts.shape}, max: {jnp.max(top_level_acts)}")
            all_feat_acts.append(top_level_acts)

    # Handle expert-level features
    if expert_indices:
        # Debug information
        print(f"MoE model parameters: num_experts={model.num_experts}, atoms_per_subspace={model.atoms_per_subspace}")

        # Extract expert-level features from expert_specific_codes
        try:
            # expert_specific_codes shape: [batch_size*seq_length, num_experts, atoms_per_subspace]
            # We need to extract specific atoms for specific experts
            all_expert_acts = []

            for expert_idx in expert_indices:
                # Calculate which expert this belongs to (top-level feature index)
                expert = expert_idx // model.atoms_per_subspace
                # Calculate which atom within that expert
                atom = expert_idx % model.atoms_per_subspace

                # Make sure the expert index is valid
                if expert < model.num_experts:
                    try:
                        # The issue: We can't directly index expert_specific_codes by expert index
                        # We need to check top_k_indices to see if this expert is in the top k

                        # Initialize with zeros (in case the expert doesn't appear in any top_k)
                        atom_acts = jnp.zeros(expert_specific_codes.shape[0])

                        # Debug shapes of relevant tensors
                        print(f"DEBUG: top_k_indices shape: {top_k_indices.shape}, dtype: {top_k_indices.dtype}")
                        print(f"DEBUG: expert_specific_codes shape: {expert_specific_codes.shape}, atoms_per_subspace: {model.atoms_per_subspace}")
                        print(f"DEBUG: Looking for expert {expert} in top_k_indices, atom {atom} in atoms_per_subspace")

                        # Create a mask for all positions where the expert appears in top_k_indices
                        # This is a 2D boolean array of shape [batch_size, k]
                        expert_mask = top_k_indices == expert

                        # Count occurrences for debugging
                        expert_count = jnp.sum(expert_mask)
                        print(f"DEBUG: Expert {expert} appears {expert_count} times in top_k_indices")

                        # Find whether each batch item has the expert in any position
                        has_expert = jnp.any(expert_mask, axis=1)
                        expert_batch_count = jnp.sum(has_expert)
                        print(f"DEBUG: Expert {expert} appears in {expert_batch_count} batch items")

                        # For batches that have the expert, find the first position where it appears
                        # This gives us the first True position (or 0 if none exists)
                        first_pos = jnp.argmax(expert_mask, axis=1)

                        # Create batch indices for all samples
                        batch_indices = jnp.arange(expert_mask.shape[0])

                        # Only keep batch indices and positions where the expert exists
                        valid_batch_indices = batch_indices[has_expert]
                        valid_expert_positions = first_pos[has_expert]

                        # Debug the valid indices
                        if valid_batch_indices.size > 0:
                            print(f"DEBUG: First few valid batch indices: {valid_batch_indices[:5]}")
                            print(f"DEBUG: First few valid expert positions: {valid_expert_positions[:5]}")

                        # Get the activations only where the expert is present
                        if valid_batch_indices.size > 0:
                            # Check if atom index is within bounds
                            if atom < expert_specific_codes.shape[2]:
                                # Use JAX's functional-style update
                                atom_acts = atom_acts.at[valid_batch_indices].set(
                                    expert_specific_codes[valid_batch_indices, valid_expert_positions, atom]
                                )
                            else:
                                print(f"WARNING: Atom index {atom} is out of bounds for expert_specific_codes (max: {expert_specific_codes.shape[2]-1})")
                                # Use epsilon values instead of zeros for out-of-bounds atoms
                                atom_acts = jnp.ones_like(atom_acts) * 1e-10

                            # Debug the non-zero values
                            nonzero_count = jnp.sum(atom_acts > 0)
                            print(f"DEBUG: After update, atom_acts has {nonzero_count} non-zero values")

                            # Ensure we have at least some tiny value to prevent downstream crashes
                            if nonzero_count == 0:
                                print(f"WARNING: No non-zero values for expert {expert}, atom {atom}. Adding small epsilon values.")
                                # Add a small epsilon to all values to avoid zeros
                                # Use a value that's very small but non-zero (won't affect visualization)
                                atom_acts = jnp.ones_like(atom_acts) * 1e-10
                        else:
                            print(f"DEBUG: Expert {expert} not found in any batch item's top_k_indices")
                            # Add small epsilon values when expert isn't found
                            atom_acts = jnp.ones_like(atom_acts) * 1e-10

                        # Reshape to match expected format
                        if intermediate_activations.ndim > 2:
                            # Reshape to [batch_size, seq_length, 1]
                            atom_acts = atom_acts.reshape(batch_size, seq_length, 1)
                        else:
                            # Reshape to [batch_size, 1, 1]
                            atom_acts = atom_acts.reshape(batch_size, 1, 1)

                        all_expert_acts.append(atom_acts)
                        print(f"Expert {expert}, Atom {atom} acts shape: {atom_acts.shape}, max: {jnp.max(atom_acts)}")
                    except Exception as e:
                        print(f"ERROR processing expert {expert}, atom {atom}: {e}")
                        traceback.print_exc()
                        # Use a placeholder with small epsilon values instead of zeros
                        # This prevents downstream crashes
                        if intermediate_activations.ndim > 2:
                            placeholder = jnp.ones((batch_size, seq_length, 1)) * 1e-10
                        else:
                            placeholder = jnp.ones((batch_size, 1, 1)) * 1e-10
                        all_expert_acts.append(placeholder)
                else:
                    print(f"WARNING: Expert index {expert} is out of bounds ({expert_specific_codes.shape[1]} experts available)")
                    # Use a placeholder with small epsilon values for out-of-bounds experts
                    # This prevents downstream crashes from pure zeros
                    if intermediate_activations.ndim > 2:
                        placeholder = jnp.ones((batch_size, seq_length, 1)) * 1e-10
                    else:
                        placeholder = jnp.ones((batch_size, 1, 1)) * 1e-10
                    all_expert_acts.append(placeholder)

            # Concatenate all expert activations if any were collected
            if all_expert_acts:
                expert_acts = jnp.concatenate(all_expert_acts, axis=-1)
                print(f"Expert-level acts shape: {expert_acts.shape}, max: {jnp.max(expert_acts)}")
                all_feat_acts.append(expert_acts)
        except Exception as e:
            # If there's an error extracting expert activations, log it and create placeholders
            print(f"ERROR extracting expert-level activations: {e}")

            # Create placeholder activations with zeros
            if intermediate_activations.ndim > 2:
                placeholder = jnp.zeros((batch_size, seq_length, len(expert_indices)))
            else:
                placeholder = jnp.zeros((batch_size, 1, len(expert_indices)))

            print(f"Created placeholder expert acts with shape {placeholder.shape}, values = 0")
            all_feat_acts.append(placeholder)

    # Combine all feature activations
    if all_feat_acts:
        all_feat_acts = jnp.concatenate(all_feat_acts, axis=-1)
        print(f"Combined feature activations shape: {all_feat_acts.shape}, max: {jnp.max(all_feat_acts)}")
    else:
        # If no features were requested or found, create empty tensor
        all_feat_acts = jnp.zeros((batch_size, seq_length, 0))
        print("WARNING: No feature activations found or requested!")

    # Update correlation statistics
    flat_acts = all_feat_acts.reshape(-1, all_feat_acts.shape[-1]).T
    flat_inputs = intermediate_activations.reshape(-1, intermediate_activations.shape[-1]).T

    corr_neurons.update(flat_acts, flat_inputs)
    corr_encoder.update(flat_acts, flat_acts)

    # Step 3: Get feature output directions for each feature
    feature_output_directions = []
    feature_resid_directions = []

    # Handle top level feature directions
    if top_level_indices:
        top_level_decoder = model.get_top_level_decoder()
        for idx in top_level_indices:
            # Extract decoder weights for this feature (analogous to W_dec[feature_indices] in the original code)
            feature_dir = top_level_decoder[:, idx]
            feature_output_directions.append(feature_dir)

            # For top-level features, residual direction is the same as output direction
            # This is equivalent to to_resid_direction when the hook point is in the residual stream
            feature_resid_directions.append(feature_dir)

    # Handle expert level feature directions
    if expert_indices:
        for expert_idx in expert_indices:
            # Calculate which top-level feature this belongs to
            top_feature = expert_idx // model.atoms_per_subspace
            # Calculate which atom within that expert
            atom_idx = expert_idx % model.atoms_per_subspace

            # Get the output direction from the model structure
            # W_up[expert, input_dim, subspace_dim] @ decoder_weights[expert, subspace_dim, atoms]
            # This creates the equivalent of W_dec for expert-level features
            feature_dir = jnp.dot(
                model.W_up[top_feature, :, :],
                model.decoder_weights[top_feature, :, atom_idx]
            )
            feature_output_directions.append(feature_dir)

            # In our MoE implementation, we're already mapping to residual stream space
            # This is equivalent to the to_resid_direction transformation in the original code
            feature_resid_directions.append(feature_dir)

    feature_output_directions = jnp.stack(feature_output_directions)
    feature_resid_directions = jnp.stack(feature_resid_directions)

    # Step 4: Calculate logit effects if unembedding matrix is provided
    if unembedding_matrix is not None:
        # Check and fix dimension mismatch
        if unembedding_matrix.shape[0] != feature_resid_directions.shape[-1]:
            # Try to transpose if dimensions would match after transposing
            if unembedding_matrix.shape[1] == feature_resid_directions.shape[-1]:
                print(f"Transposing unembedding matrix from shape {unembedding_matrix.shape} to match feature dimensions {feature_resid_directions.shape[-1]}")
                unembedding_matrix = unembedding_matrix.T
            else:
                # Log warning and set to None if dimensions still don't match
                print(f"WARNING: Unembedding matrix shape mismatch: expected first dimension {feature_resid_directions.shape[-1]}, got {unembedding_matrix.shape}")
                unembedding_matrix = None

        if unembedding_matrix is not None:
            # Project to logit space similar to transformer_lens_wrapper.py
            # This is equivalent to `feature_resid_dir @ model.W_U` in the original implementation
            logit_effects = jnp.matmul(feature_resid_directions, unembedding_matrix)
        else:
            vocab_size = 50257  # Default size for many models
            logit_effects = jnp.zeros((len(feature_indices), vocab_size))
    else:
        # If unembedding matrix isn't available, log warning and use zeros
        print("WARNING: No unembedding matrix provided. Logit effects will be zeros.")
        print("Consider passing the unembedding_matrix parameter to update_data().")
        vocab_size = 50257  # Default size for many models
        logit_effects = jnp.zeros((len(feature_indices), vocab_size))

    # Step 5: For each feature, generate the visualization data
    new_data = SaeVisData()

    # A. Get overall feature statistics
    new_data.feature_stats = compute_feature_statistics(
        flat_acts,
        batch_size=min(64, flat_acts.shape[0])
    )

    # B. Get feature tables data
    feature_tables_data = get_features_table_data(
        feature_output_directions,
        corr_neurons,
        corr_encoder,
        n_rows=5
    )

    # For each feature, create feature data
    for i, feature_idx in enumerate(feature_indices):
        # Extract feature's activations
        feature_acts = all_feat_acts[..., i]

        # For MoE models, we need to adapt to different activation scales
        # The model internally uses a leaky_offset_relu with a threshold that depends on input dimension
        # When working with normalized data, the activations might be very small

        # Debug info
        max_act_value = float(jnp.max(feature_acts))

        # If max activation is exactly zero, add a small epsilon to prevent crashes
        if max_act_value == 0:
            print(f"WARNING: Feature {feature_idx} has all zero activations. Adding tiny epsilon values.")
            # Create a copy with small non-zero values
            feature_acts = jnp.ones_like(feature_acts) * 1e-10
            # Update max accordingly
            max_act_value = 1e-10

        print(f"Feature {feature_idx} max activation: {max_act_value:.6f}")

        # Use a relative threshold based on max activation value
        # Either 5% of max or 0.000001, whichever is larger
        activation_threshold = max(max_act_value * 0.05, 0.000001)

        # Calculate nonzero percentage - make sure we have non-zero data
        nonzero_mask = feature_acts > activation_threshold
        frac_nonzero = float(jnp.sum(nonzero_mask) / (feature_acts.size or 1))  # Avoid div by zero

        # Get non-zero activations for histogram - handle empty case
        if jnp.sum(nonzero_mask) > 0:
            nonzero_acts = feature_acts[nonzero_mask]
            print(f"Feature {feature_idx} has {jnp.sum(nonzero_mask)} activations above threshold {activation_threshold:.8f}")
        else:
            # Generate at least some data for visualization
            # Use the actual distribution of data, just in case there's structure in small values
            positive_mask = feature_acts > 0
            if jnp.sum(positive_mask) > 0:
                nonzero_acts = feature_acts[positive_mask]
                print(f"Using {jnp.sum(positive_mask)} positive activations for visualization")
            else:
                # Generate synthetic data to prevent crashes
                # Create an array with small epsilon values with some minor variations
                # This prevents downstream crashes while still being visually near-zero
                base_val = 1e-6
                # Generate 100 small values with tiny variations to avoid singularities
                nonzero_acts = jnp.ones(100) * base_val + jnp.linspace(0, base_val/10, 100)
                print(f"Warning: Feature {feature_idx} has no positive activations. Using synthetic non-zero values.")

            # Set density to a small value to avoid confusion
            frac_nonzero = 0.001

        # If ablation is enabled, prepare the necessary arguments for sequence data generation
        sequence_group_args = {
            "token_ids": token_ids,
            "token_strings": token_strings,
            "feat_acts": feature_acts,
            "buffer_size": 5,
            "perform_ablation": perform_ablation,
            "n_quantiles": 5  # Explicitly set to 5 to ensure quantile groups are generated
        }

        if perform_ablation:
            sequence_group_args.update({
                "feature_idx": feature_idx,
                "intermediate_activations": intermediate_activations,
                "feature_resid_dir": feature_resid_directions[i],
                "sae_model": model,
                "partial_model": partial_model,
                "norms": norms
            })

        # Feature data dictionary - with better error handling
        try:
            # Try to extract feature tables data safely
            feature_tables = {}
            for k in feature_tables_data:
                try:
                    if isinstance(feature_tables_data[k], list) and feature_tables_data[k] and isinstance(feature_tables_data[k][0], list):
                        feature_tables[k] = [v[i] for v in feature_tables_data[k]] if i < len(feature_tables_data[k][0]) else []
                    else:
                        feature_tables[k] = feature_tables_data[k][i] if i < len(feature_tables_data[k]) else []
                except Exception as e:
                    print(f"Warning: Error processing feature tables for key {k}: {e}")
                    # Create fallback data
                    feature_tables[k] = []
        except Exception as e:
            print(f"Warning: Error processing feature tables data: {e}")
            # Create empty feature tables data
            feature_tables = {
                "neuron_alignment_indices": [],
                "neuron_alignment_values": [],
                "neuron_alignment_l1": [],
                "correlated_neurons_indices": [],
                "correlated_neurons_pearson": [],
                "correlated_neurons_cossim": [],
                "correlated_features_indices": [],
                "correlated_features_pearson": [],
                "correlated_features_cossim": [],
                "correlated_b_features_indices": [],
                "correlated_b_features_pearson": [],
                "correlated_b_features_cossim": []
            }

        # Generate histogram data with better error handling
        try:
            acts_histogram = compute_histogram_data(nonzero_acts, n_bins=40)
            # Format the density based on the scale (use scientific notation for very small values)
            density_str = f"{frac_nonzero:.3%}" if frac_nonzero >= 0.001 else f"{frac_nonzero:.2e}"
            # Format the max based on scale (use scientific notation for very small values)
            max_str = f"{max_act_value:.3f}" if max_act_value >= 0.001 else f"{max_act_value:.2e}"
            acts_histogram["title"] = f"ACTIVATIONS<br>DENSITY = {density_str}<br>MAX = {max_str}"
        except Exception as e:
            print(f"Warning: Error computing activation histogram: {e}")
            # Create fallback data
            acts_histogram = {
                "bar_heights": [0] * 40,
                "bar_values": [0.1 * i for i in range(40)],
                "tick_vals": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "title": f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}<br>MAX = {max_act_value:.6f}"
            }

        try:
            logits_histogram = compute_histogram_data(logit_effects[i], n_bins=40)
        except Exception as e:
            print(f"Warning: Error computing logits histogram: {e}")
            # Create fallback data
            logits_histogram = {
                "bar_heights": [0] * 40,
                "bar_values": [0.1 * i for i in range(40)],
                "tick_vals": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            }

        try:
            logits_table = get_logits_table_data(logit_effects[i], n_rows=10)  # Increased to 10 to match SAEDashboard expectations
        except Exception as e:
            print(f"Warning: Error computing logits table: {e}")
            # Create fallback data with 10 entries as expected by SAEDashboard
            logits_table = {
                "top_token_ids": list(range(10)),
                "top_logits": [0.1] * 10,
                "bottom_token_ids": list(range(10)),
                "bottom_logits": [-0.1] * 10
            }

        try:
            # Get n_quantiles parameter or use default of 5
            n_quantiles = sequence_group_args.get('n_quantiles', 5)
            print(f"Calling get_sequence_groups with n_quantiles={n_quantiles}")

            # Now get_sequence_groups returns a list of groups, not a single group
            sequence_groups = get_sequence_groups(**sequence_group_args)
            sequence_data = {"seq_group_data": sequence_groups}

            # Print detailed info about each group
            print(f"Generated {len(sequence_groups)} sequence groups for feature {feature_idx}:")
            for i, group in enumerate(sequence_groups):
                title = group.get('title', 'No Title')
                seq_count = len(group.get('seq_data', []))
                print(f"  Group {i}: '{title}' with {seq_count} sequences")
        except Exception as e:
            print(f"Warning: Error generating sequence data: {e}")
            traceback.print_exc()
            # Create minimal fallback data
            sequence_data = {"seq_group_data": []}

        # Assemble the feature data dictionary
        feature_data = {
            "feature_tables_data": feature_tables,
            "acts_histogram_data": acts_histogram,
            "logits_histogram_data": logits_histogram,
            "logits_table_data": logits_table,
            "sequence_data": sequence_data
        }

        # Add to feature data dictionary
        new_data.feature_data_dict[feature_idx] = feature_data

    # Update the provided data with new data
    # Before updating, check if we generated any real (non-synthetic) data
    has_real_data = False
    for feature_idx, feature_data in new_data.feature_data_dict.items():
        # Check sequence data for any non-synthetic groups
        seq_groups = feature_data.get("sequence_data", {}).get("seq_group_data", [])
        is_all_synthetic = all(
            "SYNTHETIC DATA" in group.get("title", "")
            for group in seq_groups if group.get("title")
        )

        # Check histogram data for non-synthetic values
        acts_histogram = feature_data.get("acts_histogram_data", {})
        values = acts_histogram.get("bar_values", [])
        has_large_values = any(abs(v) > 1e-5 for v in values) if values else False

        # If we have non-synthetic data for any feature, consider this a real update
        if not is_all_synthetic or has_large_values:
            has_real_data = True
            print(f"Real (non-synthetic) data generated for feature {feature_idx}")
            break

    if not has_real_data:
        print("NOTICE: This update contains only synthetic data. If you see this repeatedly, "
              "it suggests no features are activating in your input.")

    # Always update, but our improved SaeVisData.update method will handle synthetic data properly
    data.update(new_data)

    return data


def pregenerate_sae_data(model, precomputed_data, feature_indices, unembedding_matrix=None, batch_size=1000):
    """
    Generates SAE visualization data from precomputed activations using CPU-based processing
    for datasets too large to fit in GPU memory.

    Args:
        model: MoE SAE model (only used for feature output directions, not encoding)
        precomputed_data: Tuple of (top_level_latent_codes, expert_specific_codes, top_k_indices,
                          top_k_values, token_ids, token_strings) where:
                            - top_level_latent_codes: [total_tokens, num_experts] numpy array
                            - expert_specific_codes: [total_tokens, k, atoms_per_subspace] numpy array
                            - top_k_indices: [total_tokens, k] numpy array of expert indices
                            - top_k_values: [total_tokens, k] numpy array of activation values
                            - token_ids: [total_tokens] array of token IDs
                            - token_strings: [total_tokens] array of token strings
        feature_indices: List of feature indices to visualize
        unembedding_matrix: The model's unembedding matrix for projecting to logit space.
                           Shape: [hidden_dim, vocab_size]
        batch_size: Size of batches to process at once to manage memory usage

    Returns:
        SaeVisData: Data structure for visualization

    Note:
        This function does not perform ablation/loss effect calculations as that would
        require a partial model and running reconstruction through it.
    """
    import numpy as np  # Use numpy for CPU processing

    # Unpack the precomputed data
    top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values, token_ids, token_strings = precomputed_data
    # Convert jax arrays to numpy if needed
    if hasattr(top_level_latent_codes, 'device_buffer'):
        print("Converting JAX arrays to NumPy")
        top_level_latent_codes = np.array(top_level_latent_codes)
        expert_specific_codes = np.array(expert_specific_codes)
        top_k_indices = np.array(top_k_indices)
        top_k_values = np.array(top_k_values)
        token_ids = np.array(token_ids)


    # Initialize the SaeVisData object
    data = SaeVisData()

    # Print debug info
    print(f"Processing {top_level_latent_codes.shape[0]} tokens in batches of {batch_size}")
    print(f"Top level latent codes shape: {top_level_latent_codes.shape}")
    print(f"Expert specific codes shape: {expert_specific_codes.shape}")
    print(f"Top k indices shape: {top_k_indices.shape}")

    # Step 1: Get feature output directions - this still uses JAX/model
    # ----------------------------------------------------------------
    feature_output_directions = []
    feature_resid_directions = []

    # Handle top level feature directions
    top_level_indices = [idx for idx in feature_indices if idx < model.num_experts]
    if top_level_indices:
        top_level_decoder = model.get_top_level_decoder()
        for idx in top_level_indices:
            # Extract decoder weights for this feature
            feature_dir = top_level_decoder[:, idx]
            feature_output_directions.append(feature_dir)
            feature_resid_directions.append(feature_dir)  # Same for top-level features

    # Handle expert level feature directions
    expert_indices = [idx - model.num_experts for idx in feature_indices if idx >= model.num_experts]
    if expert_indices:
        for expert_idx in expert_indices:
            # Calculate which expert this belongs to
            top_feature = expert_idx // model.atoms_per_subspace
            atom_idx = expert_idx % model.atoms_per_subspace

            # Get output direction
            feature_dir = jnp.dot(
                model.W_up[top_feature, :, :],
                model.decoder_weights[top_feature, :, atom_idx]
            )
            feature_output_directions.append(feature_dir)
            feature_resid_directions.append(feature_dir)

    # Convert to numpy arrays
    feature_output_directions = np.array(feature_output_directions)
    feature_resid_directions = np.array(feature_resid_directions)

    # Step 2: Calculate logit effects if unembedding matrix is provided
    # ----------------------------------------------------------------
    if unembedding_matrix is not None:
        # Check dimensions
        if unembedding_matrix.shape[0] != feature_resid_directions.shape[1]:
            if unembedding_matrix.shape[1] == feature_resid_directions.shape[1]:
                print(f"Transposing unembedding matrix to match feature dimensions")
                unembedding_matrix = unembedding_matrix.T
            else:
                print(f"WARNING: Unembedding matrix shape mismatch, ignoring")
                unembedding_matrix = None

        if unembedding_matrix is not None:
            # Convert to numpy if needed
            if hasattr(unembedding_matrix, 'device_buffer'):
                unembedding_matrix = np.array(unembedding_matrix)

            # Project to logit space
            logit_effects = np.matmul(feature_resid_directions, unembedding_matrix)
        else:
            vocab_size = 50257
            logit_effects = np.zeros((len(feature_indices), vocab_size))
    else:
        print("WARNING: No unembedding matrix provided. Logit effects will be zeros.")
        vocab_size = 50257
        logit_effects = np.zeros((len(feature_indices), vocab_size))

    # Step 3: Extract feature activations in batches
    # ---------------------------------------------
    # Initialize storage for all feature activations
    total_tokens = top_level_latent_codes.shape[0]
    all_feature_acts = np.zeros((total_tokens, len(feature_indices)))

    # Create a mapping from feature indices to array columns
    feature_to_column = {idx: pos for pos, idx in enumerate(feature_indices)}
    print(f"Feature to column mapping created with {len(feature_to_column)} entries")

    # Process in batches to manage memory
    first_batch = True
    total_batches = (total_tokens + batch_size - 1) // batch_size
    processed_tokens = 0
    
    for batch_start in range(0, total_tokens, batch_size):
        batch_end = min(batch_start + batch_size, total_tokens)
        batch_size_actual = batch_end - batch_start
        processed_tokens += batch_size_actual
        print(f"Processing batch {batch_start}-{batch_end} of {total_tokens} (batch {(batch_start//batch_size)+1}/{total_batches})")

        # Extract batch data
        batch_top_level = top_level_latent_codes[batch_start:batch_end]
        batch_expert_codes = expert_specific_codes[batch_start:batch_end]
        batch_top_k = top_k_indices[batch_start:batch_end]
        
        # On first batch, verify data dimensions
        if first_batch:
            print(f"Batch data dimensions:")
            print(f"  - batch_top_level: {batch_top_level.shape}")
            print(f"  - batch_expert_codes: {batch_expert_codes.shape}")
            print(f"  - batch_top_k: {batch_top_k.shape}")
            first_batch = False

        # Extract top-level features directly to their correct columns
        for idx in top_level_indices:
            # Use the mapping to find the correct column
            column_idx = feature_to_column[idx]
            all_feature_acts[batch_start:batch_end, column_idx] = batch_top_level[:, idx]

        # Extract expert-level features directly to their correct columns
        for expert_idx in expert_indices:
            # Use the mapping to find the correct column
            column_idx = feature_to_column[expert_idx + model.num_experts]
            
            expert = expert_idx // model.atoms_per_subspace
            atom = expert_idx % model.atoms_per_subspace

            # Initialize with zeros
            batch_acts = np.zeros(batch_end - batch_start)

            # For each sample in batch, check if our expert is in top-k
            for j in range(batch_end - batch_start):
                # Find where this expert appears in top-k indices for this sample
                matches = np.where(batch_top_k[j] == expert)[0]

                # If found, get the activation
                if len(matches) > 0:
                    # Use first occurrence
                    relative_expert_idx = matches[0]
                    # Get activation for this sample
                    if batch_expert_codes.shape[2] > atom:  # Check atom index is in bounds
                        batch_acts[j] = batch_expert_codes[j, relative_expert_idx, atom]# + batch_top_level[j, expert]

            # Add to overall activations using the correct column
            all_feature_acts[batch_start:batch_end, column_idx] = batch_acts
    
    # Verify all tokens were processed
    print(f"Processed {processed_tokens} tokens out of {total_tokens} total tokens ({processed_tokens/total_tokens:.2%})")
    
    # Check for max value discrepancies in top-level features
    print("Checking for max value discrepancies in top-level features...")
    for idx in top_level_indices:
        column_idx = feature_to_column[idx]
        from_acts = float(np.max(all_feature_acts[:, column_idx]))
        direct_max = float(np.max(top_level_latent_codes[:, idx]))
        if abs(from_acts - direct_max) > 1e-6:
            print(f"Feature {idx} max discrepancy:")
            print(f"  - From all_feature_acts[:, {column_idx}]: {from_acts}")
            print(f"  - Direct from top_level_latent_codes[:, {idx}]: {direct_max}")

    # Step 4: Compute statistics and create visualization data
    # -------------------------------------------------------
    print("Computing statistics for all features")
    new_data = SaeVisData()

    # Get overall feature statistics
    feature_stats = {
        "max": [],
        "frac_nonzero": [],
        "quantile_data": [],
        "quantiles": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    }

    # Compute stats directly with numpy
    for i in range(all_feature_acts.shape[1]):
        feat_acts = all_feature_acts[:, i]

        # Calculate max
        max_val = float(np.max(feat_acts))
        feature_stats["max"].append(max_val)

        # Calculate fraction of non-zero values
        threshold = 1e-6
        frac_nonzero = float(np.mean(np.abs(feat_acts) > threshold))
        feature_stats["frac_nonzero"].append(frac_nonzero)

        # Calculate quantiles
        quantiles = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        feat_quantiles = [float(np.quantile(feat_acts, q)) for q in quantiles]

        # Strip out zeros from beginning
        first_nonzero = next((i for i, x in enumerate(feat_quantiles) if abs(x) > 1e-6), len(feat_quantiles))
        feat_quantiles = feat_quantiles[first_nonzero:]

        # Round values for smaller JSON
        feat_quantiles = [round(q, 6) for q in feat_quantiles]
        feature_stats["quantile_data"].append(feat_quantiles)

    new_data.feature_stats = feature_stats
    
    # For each feature, create visualization data
    print("Generating visualization data for each feature")
    for i, feature_idx in enumerate(feature_indices):
        print(f"Processing feature idx {feature_idx} at column {i}")
        # Extract this feature's activations
        feature_acts = all_feature_acts[:, i]

        # Calculate max and threshold
        max_act_value = float(np.max(feature_acts))
        activation_threshold = max(max_act_value * 0.05, 0.000001)

        # Calculate nonzero percentage
        nonzero_mask = feature_acts > activation_threshold
        frac_nonzero = float(np.sum(nonzero_mask) / feature_acts.size)

        # Get non-zero activations for histogram
        if np.sum(nonzero_mask) > 0:
            nonzero_acts = feature_acts[nonzero_mask]
        else:
            # Generate synthetic data for empty histograms
            base_val = 1e-6
            nonzero_acts = np.ones(100) * base_val + np.linspace(0, base_val/10, 100)
            frac_nonzero = 0.001

        # Create feature tables data - correlations are not computed in this function
        # as they would require full activations
        feature_tables = {
            "neuron_alignment_indices": [],
            "neuron_alignment_values": [],
            "neuron_alignment_l1": [],
            "correlated_neurons_indices": [],
            "correlated_neurons_pearson": [],
            "correlated_neurons_cossim": [],
            "correlated_features_indices": [],
            "correlated_features_pearson": [],
            "correlated_features_cossim": [],
            "correlated_b_features_indices": [],
            "correlated_b_features_pearson": [],
            "correlated_b_features_cossim": []
        }

        # Create neuron alignment data (which we can do from feature directions)
        # Get L1 norms
        l1_norms = np.linalg.norm(feature_output_directions[i], ord=1)
        l1_fractions = feature_output_directions[i] / (l1_norms + 1e-6)

        # Find which dimensions have largest weights
        neuron_indices = np.argsort(-np.abs(feature_output_directions[i]))[:5]  # Top 5
        neuron_values = feature_output_directions[i][neuron_indices]
        neuron_l1 = l1_fractions[neuron_indices]

        # Update feature tables
        feature_tables["neuron_alignment_indices"] = neuron_indices.tolist()
        feature_tables["neuron_alignment_values"] = neuron_values.tolist()
        feature_tables["neuron_alignment_l1"] = neuron_l1.tolist()

        # Generate histogram data
        acts_histogram = compute_histogram_data(nonzero_acts, n_bins=40)
        density_str = f"{frac_nonzero:.3%}" if frac_nonzero >= 0.001 else f"{frac_nonzero:.2e}"
        max_str = f"{max_act_value:.3f}" if max_act_value >= 0.001 else f"{max_act_value:.2e}"
        acts_histogram["title"] = f"ACTIVATIONS (Feature {feature_idx})<br>DENSITY = {density_str}<br>MAX = {max_str}"

        # Generate logits histogram and table
        logits_histogram = compute_histogram_data(logit_effects[i], n_bins=40)
        logits_table = get_logits_table_data(logit_effects[i], n_rows=10)

        # Generate sequence data with top examples and quantiles
        sequence_data = generate_sequence_data_numpy(
            feature_acts, token_ids, token_strings,
            max_sequences=10, buffer_size=5, n_quantiles=5
        )

        # Assemble feature data
        feature_data = {
            "feature_tables_data": feature_tables,
            "acts_histogram_data": acts_histogram,
            "logits_histogram_data": logits_histogram,
            "logits_table_data": logits_table,
            "sequence_data": sequence_data
        }

        # Add to feature data dictionary
        new_data.feature_data_dict[feature_idx] = feature_data

    print(f"Successfully processed {total_tokens} tokens for {len(feature_indices)} features")
    return new_data


def generate_sequence_data_numpy(feature_acts, token_ids, token_strings, max_sequences=10, buffer_size=5, n_quantiles=5):
    """
    Creates sequence groups for visualization using numpy arrays

    Args:
        feature_acts: Feature activation values for all tokens
        token_ids: Array of token IDs corresponding to activations
        token_strings: Array of token strings corresponding to token_ids
        max_sequences: Maximum number of sequences to generate per group
        buffer_size: Size of context window around activating tokens
        n_quantiles: Number of quantile groups to generate

    Returns:
        dict: Sequence data for visualization with top activations and quantile groups
    """
    import numpy as np

    # Check if data is empty
    if feature_acts.size == 0:
        return {"seq_group_data": []}

    # Find max activation
    max_act = float(np.max(feature_acts))

    # If max is exactly 0, use tiny epsilon values
    if max_act == 0:
        return {"seq_group_data": [
            {
                "title": "SYNTHETIC DATA (NO ACTIVATIONS)",
                "seq_data": [{
                    "original_index": 0,
                    "qualifying_token_index": 0,
                    "token_ids": [0, 1, 2, 3, 4],
                    "feat_acts": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
                    "token_strings": ["[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"],
                    "loss_contribution": [],
                    "token_logits": [],
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                }]
            }
        ]}

    # Initialize list to hold all sequence groups
    sequence_groups = []

    # 1. Generate TOP ACTIVATIONS group
    # ---------------------------------
    # Use a threshold to find activating tokens
    threshold = max(max_act * 0.05, 0.000001)
    activating_tokens = np.where(feature_acts > threshold)[0]

    # If no tokens exceed threshold, return synthetic data
    if len(activating_tokens) == 0:
        return {"seq_group_data": [
            {
                "title": "SYNTHETIC DATA (NO ACTIVATIONS ABOVE THRESHOLD)",
                "seq_data": [{
                    "original_index": 0,
                    "qualifying_token_index": 0,
                    "token_ids": [0, 1, 2, 3, 4],
                    "feat_acts": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
                    "token_strings": ["[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"],
                    "loss_contribution": [],
                    "token_logits": [],
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                }]
            }
        ]}

    # Sort by activation value (descending)
    sorted_indices = np.argsort(-feature_acts[activating_tokens])
    top_tokens = activating_tokens[sorted_indices[:max_sequences]]

    # Create sequence data for top activating tokens
    top_sequence_data = []
    for token_idx in top_tokens:
        # Calculate start and end for context window
        start_idx = max(0, token_idx - buffer_size)
        end_idx = min(len(token_ids), token_idx + buffer_size + 1)

        # Extract token ids and activations
        seq_token_ids = token_ids[start_idx:end_idx]
        seq_feat_acts = feature_acts[start_idx:end_idx]
        seq_token_strings = token_strings[start_idx:end_idx]

        # Check token_strings type - convert if needed
        if isinstance(seq_token_strings, np.ndarray):
            seq_token_strings = seq_token_strings.tolist()

        # Convert to needed format
        seq_data = {
            "original_index": int(token_idx),
            "qualifying_token_index": int(token_idx - start_idx),  # Adjust for window
            "token_ids": seq_token_ids.tolist() if isinstance(seq_token_ids, np.ndarray) else list(seq_token_ids),
            "feat_acts": [round(float(act), 4) for act in seq_feat_acts],
            "token_strings": seq_token_strings,
            "loss_contribution": [],  # No loss contribution without ablation
            "token_logits": [],
            "top_token_ids": [],
            "top_logits": [],
            "bottom_token_ids": [],
            "bottom_logits": []
        }
        top_sequence_data.append(seq_data)

    # Create the TOP ACTIVATIONS group
    max_str = f"{max_act:.6f}" if max_act < 0.01 else f"{max_act:.3f}"
    top_group = {
        "title": f"TOP ACTIVATIONS <br>MAX = {max_str}",
        "seq_data": top_sequence_data
    }
    sequence_groups.append(top_group)

    # 2. Generate QUANTILE groups
    # --------------------------
    if n_quantiles > 0:
        # Generate quantile boundaries similar to get_sequence_groups
        quantiles = np.linspace(0, max_act, n_quantiles + 1)

        # For each quantile range (except the last), create a group
        for i in range(n_quantiles):
            lower, upper = float(quantiles[i]), float(quantiles[i + 1])

            # Find tokens within this range
            in_range_mask = (feature_acts >= lower) & (feature_acts <= upper)
            tokens_in_range = np.where(in_range_mask)[0]

            # Skip ranges with no tokens
            if len(tokens_in_range) == 0:
                print(f"Skipping empty quantile range: {lower:.6f} - {upper:.6f}")
                continue

            # Calculate percentage of activations in this range
            pct = float(len(tokens_in_range)) / feature_acts.size

            # Take a random sample of tokens in range (up to max_sequences)
            if len(tokens_in_range) > max_sequences:
                # Create a random sample without replacement
                np.random.seed(i)  # Use quantile index as seed for reproducibility
                sample_indices = np.random.choice(len(tokens_in_range), max_sequences, replace=False)
                tokens_in_range = tokens_in_range[sample_indices]

            # Create sequence data for this quantile group
            quantile_sequence_data = []
            for token_idx in tokens_in_range:
                # Calculate start and end for context window
                start_idx = max(0, token_idx - buffer_size)
                end_idx = min(len(token_ids), token_idx + buffer_size + 1)

                # Extract token ids and activations
                seq_token_ids = token_ids[start_idx:end_idx]
                seq_feat_acts = feature_acts[start_idx:end_idx]
                seq_token_strings = token_strings[start_idx:end_idx]

                # Check token_strings type - convert if needed
                if isinstance(seq_token_strings, np.ndarray):
                    seq_token_strings = seq_token_strings.tolist()

                # Convert to needed format
                seq_data = {
                    "original_index": int(token_idx),
                    "qualifying_token_index": int(token_idx - start_idx),  # Adjust for window
                    "token_ids": seq_token_ids.tolist() if isinstance(seq_token_ids, np.ndarray) else list(seq_token_ids),
                    "feat_acts": [round(float(act), 4) for act in seq_feat_acts],
                    "token_strings": seq_token_strings,
                    "loss_contribution": [],  # No loss contribution without ablation
                    "token_logits": [],
                    "top_token_ids": [],
                    "top_logits": [],
                    "bottom_token_ids": [],
                    "bottom_logits": []
                }
                quantile_sequence_data.append(seq_data)

            # Format the title similar to SAEDashboard's format
            # Use more precision for small values
            if lower < 0.01 or upper < 0.01:
                lower_str, upper_str = f"{lower:.6f}", f"{upper:.6f}"
            else:
                lower_str, upper_str = f"{lower:.3f}", f"{upper:.3f}"

            # Add the quantile group
            quantile_group = {
                "title": f"INTERVAL {lower_str} - {upper_str}<br>CONTAINS {pct:.3%}",
                "seq_data": quantile_sequence_data
            }
            sequence_groups.append(quantile_group)

    return {"seq_group_data": sequence_groups}
