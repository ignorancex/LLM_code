"""
SAEDashboard Adapter for JAX SAE Interface - Clean Version

This module provides adapter functions to convert JAX SAE visualization data
to SAEDashboard's format for visualization and saving.
This version doesn't include the NaN handling or other fixes from work.md.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Check if SAEDashboard is available
SAEDASHBOARD_AVAILABLE = importlib.util.find_spec("sae_dashboard") is not None

if not SAEDASHBOARD_AVAILABLE:
    print("Warning: SAEDashboard is not installed. Adapter functions will not work.")
    print("To install, add SAEDashboard to your Python path or install it.")

# Check if JAX is available for array type checking
JAX_AVAILABLE = importlib.util.find_spec("jax") is not None

# Import JAX only if available
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    JAX_ARRAY_TYPES = (jnp.ndarray, jax.Array)
else:
    JAX_ARRAY_TYPES = tuple()


def convert_to_native_python(obj: Any) -> Any:
    """
    Convert JAX arrays or NumPy arrays to native Python types recursively.
    Also handles NaN and inf values by replacing them with appropriate defaults.

    Args:
        obj: Any Python object, potentially containing JAX or NumPy arrays

    Returns:
        The same object with all JAX/NumPy arrays converted to native Python types
        and NaN/inf values replaced with appropriate defaults
    """
    import numpy as np
    import math

    # Helper function to clean NaN values
    def clean_numeric_value(x):
        """Replace NaN or inf values with appropriate defaults"""
        if x is None:
            return 0.0
        elif isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                return 0.0
        return x

    # Handle JAX and NumPy arrays
    if JAX_AVAILABLE and isinstance(obj, JAX_ARRAY_TYPES):
        # Convert JAX array to NumPy first
        return convert_to_native_python(np.array(obj))
    elif isinstance(obj, np.ndarray):
        # Check for NaN/inf values
        if np.issubdtype(obj.dtype, np.number):
            # Replace NaN/inf with zeros
            if np.isnan(obj).any() or np.isinf(obj).any():
                obj = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert NumPy array to native Python types
        if obj.ndim == 0:
            # Handle scalar arrays
            return clean_numeric_value(obj.item())
        elif obj.ndim == 1:
            # Handle 1D arrays
            return [convert_to_native_python(x) for x in obj]
        else:
            # Handle multi-dimensional arrays
            return [convert_to_native_python(x) for x in obj]

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: convert_to_native_python(v) for k, v in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_python(x) for x in obj]

    # Handle NaN/inf in raw numeric values
    elif isinstance(obj, (int, float)):
        return clean_numeric_value(obj)

    # Return other types unchanged
    else:
        return obj


def _import_saedashboard_classes():
    """Import SAEDashboard classes only when needed to avoid import errors."""
    if not SAEDASHBOARD_AVAILABLE:
        raise ImportError("SAEDashboard is not installed. Cannot import classes.")

    from sae_dashboard.sae_vis_data import SaeVisData, SaeVisConfig
    from sae_dashboard.feature_data import FeatureData
    from sae_dashboard.utils_fns import FeatureStatistics, get_decode_html_safe_fn
    from sae_dashboard.components import (
        FeatureTablesData,
        ActsHistogramData,
        LogitsHistogramData,
        LogitsTableData,
        SequenceData,
        SequenceGroupData,
        SequenceMultiGroupData
    )
    from sae_dashboard.data_writing_fns import save_feature_centric_vis as saed_save
    from sae_dashboard.components_config import PromptConfig, SequencesConfig

    return {
        'SaeVisData': SaeVisData,
        'SaeVisConfig': SaeVisConfig,
        'FeatureData': FeatureData,
        'FeatureStatistics': FeatureStatistics,
        'FeatureTablesData': FeatureTablesData,
        'ActsHistogramData': ActsHistogramData,
        'LogitsHistogramData': LogitsHistogramData,
        'LogitsTableData': LogitsTableData,
        'SequenceData': SequenceData,
        'SequenceGroupData': SequenceGroupData,
        'SequenceMultiGroupData': SequenceMultiGroupData,
        'save_feature_centric_vis': saed_save,
        'get_decode_html_safe_fn': get_decode_html_safe_fn,
        'PromptConfig': PromptConfig,
        'SequencesConfig': SequencesConfig
    }


def convert_feature_tables_data(jax_data):
    """Convert JAX feature tables data to SAEDashboard format."""
    if not SAEDASHBOARD_AVAILABLE:
        return None

    classes = _import_saedashboard_classes()

    # Initialize with default values
    saed_feature_tables = classes['FeatureTablesData']()

    # Ensure we have at least 3 entries (FeatureTablesConfig default n_rows)
    DEFAULT_MIN_ROWS = 3

    # Convert input data to native Python types first
    jax_data = convert_to_native_python(jax_data)

    # Map fields
    field_mappings = [
        ('neuron_alignment_indices', 'neuron_alignment_indices'),
        ('neuron_alignment_values', 'neuron_alignment_values'),
        ('neuron_alignment_l1', 'neuron_alignment_l1'),
        ('correlated_neurons_indices', 'correlated_neurons_indices'),
        ('correlated_neurons_pearson', 'correlated_neurons_pearson'),
        ('correlated_neurons_cossim', 'correlated_neurons_cossim'),
        ('correlated_features_indices', 'correlated_features_indices'),
        ('correlated_features_pearson', 'correlated_features_pearson'),
        ('correlated_features_cossim', 'correlated_features_cossim'),
        ('correlated_b_features_indices', 'correlated_b_features_indices'),
        ('correlated_b_features_pearson', 'correlated_b_features_pearson'),
        ('correlated_b_features_cossim', 'correlated_b_features_cossim')
    ]

    # Process each field
    for jax_field, saed_field in field_mappings:
        if jax_field in jax_data:
            data = jax_data[jax_field]

            # Pad to minimum rows if needed
            if isinstance(data, list) and len(data) > 0:
                # For indices fields, pad with 0, for values fields pad with 0.0
                is_indices = 'indices' in jax_field
                pad_value = 0 if is_indices else 0.0

                # Ensure we have the minimum required rows
                if len(data) < DEFAULT_MIN_ROWS:
                    data = data + [pad_value] * (DEFAULT_MIN_ROWS - len(data))
            elif not data:
                # Create default data if field is empty or missing
                is_indices = 'indices' in jax_field
                pad_value = 0 if is_indices else 0.0
                data = [pad_value] * DEFAULT_MIN_ROWS

            # Set the attribute
            setattr(saed_feature_tables, saed_field, data)
        else:
            # Create a default value for missing fields
            is_indices = 'indices' in saed_field
            pad_value = 0 if is_indices else 0.0
            setattr(saed_feature_tables, saed_field, [pad_value] * DEFAULT_MIN_ROWS)

    return saed_feature_tables


def convert_histogram_data(jax_data, title=None):
    """Convert JAX histogram data to SAEDashboard format."""
    if not SAEDASHBOARD_AVAILABLE:
        return None

    # Ensure histograms have at least 50 bins (default in ActsHistogramConfig/LogitsHistogramConfig)
    DEFAULT_MIN_BINS = 50

    # Convert input data to native Python types first
    jax_data = convert_to_native_python(jax_data)

    # Create histogram data
    histogram_data = {
        'bar_heights': jax_data.get('bar_heights', []),
        'bar_values': jax_data.get('bar_values', []),
        'tick_vals': jax_data.get('tick_vals', [])
    }

    # Ensure arrays are non-empty
    if not histogram_data['bar_heights']:
        histogram_data['bar_heights'] = [0.0] * DEFAULT_MIN_BINS
    if not histogram_data['bar_values']:
        histogram_data['bar_values'] = [0.1 * i for i in range(DEFAULT_MIN_BINS)]
    if not histogram_data['tick_vals']:
        histogram_data['tick_vals'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Pad each histogram array to ensure minimum bin count
    if len(histogram_data['bar_heights']) < DEFAULT_MIN_BINS:
        histogram_data['bar_heights'] = histogram_data['bar_heights'] + [0.0] * (DEFAULT_MIN_BINS - len(histogram_data['bar_heights']))
    if len(histogram_data['bar_values']) < DEFAULT_MIN_BINS:
        histogram_data['bar_values'] = histogram_data['bar_values'] + [0.0] * (DEFAULT_MIN_BINS - len(histogram_data['bar_values']))

    if title is not None:
        histogram_data['title'] = title
    elif 'title' in jax_data:
        histogram_data['title'] = jax_data['title']

    return histogram_data


def convert_logits_table_data(jax_data):
    """Convert JAX logits table data to SAEDashboard format."""
    if not SAEDASHBOARD_AVAILABLE:
        return None

    classes = _import_saedashboard_classes()

    # Ensure we have at least 10 entries in each list
    # (10 is the default n_rows value in LogitsTableConfig)
    DEFAULT_MIN_ENTRIES = 10

    # Convert input data to native Python types first
    jax_data = convert_to_native_python(jax_data)

    # Get values from JAX data with defaults
    top_token_ids = jax_data.get('top_token_ids', [])
    top_logits = jax_data.get('top_logits', [])
    bottom_token_ids = jax_data.get('bottom_token_ids', [])
    bottom_logits = jax_data.get('bottom_logits', [])

    # Ensure lists are non-empty
    if not top_token_ids:
        top_token_ids = list(range(DEFAULT_MIN_ENTRIES))
    if not top_logits:
        top_logits = [0.1] * DEFAULT_MIN_ENTRIES
    if not bottom_token_ids:
        bottom_token_ids = list(range(DEFAULT_MIN_ENTRIES))
    if not bottom_logits:
        bottom_logits = [-0.1] * DEFAULT_MIN_ENTRIES

    # Pad with defaults if needed
    if len(top_token_ids) < DEFAULT_MIN_ENTRIES:
        top_token_ids = top_token_ids + [0] * (DEFAULT_MIN_ENTRIES - len(top_token_ids))
    if len(top_logits) < DEFAULT_MIN_ENTRIES:
        top_logits = top_logits + [0.0] * (DEFAULT_MIN_ENTRIES - len(top_logits))
    if len(bottom_token_ids) < DEFAULT_MIN_ENTRIES:
        bottom_token_ids = bottom_token_ids + [0] * (DEFAULT_MIN_ENTRIES - len(bottom_token_ids))
    if len(bottom_logits) < DEFAULT_MIN_ENTRIES:
        bottom_logits = bottom_logits + [0.0] * (DEFAULT_MIN_ENTRIES - len(bottom_logits))

    # Create logits table data
    logits_table = classes['LogitsTableData'](
        top_token_ids=top_token_ids,
        top_logits=top_logits,
        bottom_token_ids=bottom_token_ids,
        bottom_logits=bottom_logits
    )

    return logits_table


def convert_sequence_data(jax_data):
    """Convert JAX sequence data to SAEDashboard format."""
    if not SAEDASHBOARD_AVAILABLE:
        return None

    classes = _import_saedashboard_classes()

    # Convert input data to native Python types first
    jax_data = convert_to_native_python(jax_data)

    # Let's add some debug info about what's in the sequence data
    sequence_groups = []

    # Print available group data
    seq_group_data = jax_data.get('seq_group_data', [])
    print(f"Number of sequence groups: {len(seq_group_data)}")
    for i, group_data in enumerate(seq_group_data):
        title = group_data.get('title', 'Sequence Group')
        seq_data = group_data.get('seq_data', [])
        print(f"Group {i}: '{title}' with {len(seq_data)} sequences")

    # Handle seq_group_data - we'll focus on ensuring proper data format
    # According to sequence_data_generator.py in SAEDashboard, we need at least:
    # 1. A TOP ACTIVATIONS group
    # 2. Optional quantile groups if n_quantiles > 0

    # First, check if we have any top activations group
    has_top_activations = False
    for group_data in seq_group_data:
        title = group_data.get('title', '')
        if "TOP ACTIVATIONS" in title:
            has_top_activations = True
            break

    # If we don't have a top activations group, create one
    if not has_top_activations:
        print("No TOP ACTIVATIONS group found, will create one")
        # Structure follows sequence_data_generator.py in SAEDashboard
        top_acts_group = {
            'title': f"TOP ACTIVATIONS<br>MAX = {0.0:.3f}",
            'seq_data': []
        }
        seq_group_data.insert(0, top_acts_group)

    # Process all sequence groups
    for group_idx, group_data in enumerate(seq_group_data):
        title = group_data.get('title', 'Sequence Group')
        seq_data_list = []

        # Print debug info about group and its title
        print(f"Processing sequence group {group_idx}: '{title}'")

        # Get sequence data
        seq_data = group_data.get('seq_data', [])

        # Check if this is the top activations group
        is_top_acts = "TOP ACTIVATIONS" in title

        # Check if this is a quantile group
        is_quantile = "INTERVAL" in title or "QUANTILE" in title

        # If no sequences in the top activations group, add at least one
        if len(seq_data) == 0 and is_top_acts:
            print(f"  Warning: No sequences in TOP ACTIVATIONS group. Adding placeholder.")
            seq_data.append({
                'original_index': 0,
                'qualifying_token_index': 0,
                'token_ids': [0, 0, 0],
                'feat_acts': [0.0, 0.1, 0.0],  # Some minimal activation
                'loss_contribution': [0.0, 0.0, 0.0],
                'token_logits': [0.0, 0.0, 0.0],
                'top_token_ids': [[0], [0], [0]],
                'top_logits': [[0.0], [0.0], [0.0]],
                'bottom_token_ids': [[0], [0], [0]],
                'bottom_logits': [[0.0], [0.0], [0.0]]
            })

        # Print info about sequences in this group
        if seq_data:
            print(f"  Found {len(seq_data)} sequences in '{title}'")
        else:
            # For non-top-activation groups, it's OK to have no sequences
            # SAEDashboard will just not show these groups if they're empty
            print(f"  No sequences in '{title}' group")

        # Process each sequence in the group
        for seq in seq_data:
            # Ensure minimum lengths for all sequence data
            token_ids = seq.get('token_ids', [])
            if len(token_ids) < 3:  # Minimum to avoid index errors
                token_ids = token_ids + [0] * (3 - len(token_ids))

            feat_acts = seq.get('feat_acts', [])
            if len(feat_acts) < len(token_ids):
                feat_acts = feat_acts + [0.0] * (len(token_ids) - len(feat_acts))

            loss_contribution = seq.get('loss_contribution', [])
            if len(loss_contribution) < len(token_ids):
                loss_contribution = loss_contribution + [0.0] * (len(token_ids) - len(loss_contribution))

            token_logits = seq.get('token_logits', [])
            if len(token_logits) < len(token_ids):
                token_logits = token_logits + [0.0] * (len(token_ids) - len(token_logits))

            # Handle nested lists for tokens and logits
            top_token_ids = seq.get('top_token_ids', [])
            if len(top_token_ids) < len(token_ids):
                top_token_ids = top_token_ids + [[0]] * (len(token_ids) - len(top_token_ids))

            top_logits = seq.get('top_logits', [])
            if len(top_logits) < len(token_ids):
                top_logits = top_logits + [[0.0]] * (len(token_ids) - len(top_logits))

            bottom_token_ids = seq.get('bottom_token_ids', [])
            if len(bottom_token_ids) < len(token_ids):
                bottom_token_ids = bottom_token_ids + [[0]] * (len(token_ids) - len(bottom_token_ids))

            bottom_logits = seq.get('bottom_logits', [])
            if len(bottom_logits) < len(token_ids):
                bottom_logits = bottom_logits + [[0.0]] * (len(token_ids) - len(bottom_logits))

            # Create the sequence object with validated data
            sequence = classes['SequenceData'](
                original_index=seq.get('original_index', 0),
                qualifying_token_index=seq.get('qualifying_token_index', 0),
                token_ids=token_ids,
                feat_acts=feat_acts,
                loss_contribution=loss_contribution,
                token_logits=token_logits,
                top_token_ids=top_token_ids,
                top_logits=top_logits,
                bottom_token_ids=bottom_token_ids,
                bottom_logits=bottom_logits
            )
            seq_data_list.append(sequence)

        # Create a SequenceGroupData for this group
        sequence_group = classes['SequenceGroupData'](title, seq_data_list)
        sequence_groups.append(sequence_group)

    # Create the SequenceMultiGroupData
    sequence_multi_group = classes['SequenceMultiGroupData'](sequence_groups)

    # Debug: Print the number of SequenceGroupData objects contained in this SequenceMultiGroupData
    print(f"Created SequenceMultiGroupData with {len(sequence_multi_group.seq_group_data)} groups")

    # Ensure the groups are properly structured in the object
    for i, group in enumerate(sequence_multi_group.seq_group_data):
        print(f"  Group {i}: '{group.title}' with {len(group.seq_data)} sequences")

    return sequence_multi_group


def convert_to_saedashboard_format(jax_data, config=None, tokenizer=None):
    """
    Convert JAX SaeVisData to SAEDashboard format.

    Args:
        jax_data: Our JAX SaeVisData object
        config: Optional SaeVisConfig to use
        tokenizer: Optional tokenizer to include with the data

    Returns:
        A SAEDashboard compatible SaeVisData object
    """
    if not SAEDASHBOARD_AVAILABLE:
        raise ImportError("SAEDashboard is not installed. Cannot convert data.")

    classes = _import_saedashboard_classes()

    # Convert the entire input data structure to native Python types
    jax_data = convert_to_native_python(jax_data)

    # Get the feature indices from the JAX data
    original_features = list(jax_data.feature_data_dict.keys())
    print(f"Original feature indices from jax_data: {original_features}")

    # Create a default config if none provided
    if config is None:
        features = original_features.copy()
        if not features:
            features = [0]

        config = classes['SaeVisConfig'](
            hook_point="residual_stream",
            features=features,
            use_dfa=True,
            device="cpu",
            dtype="float32"
        )

    # Create feature statistics - make sure it has all required attributes
    feature_stats = classes['FeatureStatistics']()
    if hasattr(jax_data, 'feature_stats') and jax_data.feature_stats:
        fs = jax_data.feature_stats
        feature_stats.max = fs.get('max', [0.1] * len(config.features))
        feature_stats.frac_nonzero = fs.get('frac_nonzero', [0.01] * len(config.features))
        feature_stats.quantile_data = fs.get('quantile_data', [[0.0, 0.0, 0.1]] * len(config.features))
        feature_stats.quantiles = fs.get('quantiles', [0.0, 0.5, 1.0])
    else:
        # Create reasonable defaults if no stats are available
        feature_stats.max = [0.1] * len(config.features)
        feature_stats.frac_nonzero = [0.01] * len(config.features)
        feature_stats.quantile_data = [[0.0, 0.0, 0.1]] * len(config.features)
        feature_stats.quantiles = [0.0, 0.5, 1.0]

    # Create a mock model with tokenizer for SAEDashboard compatibility
    class MockModel:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    # Initialize SAEDashboard's SaeVisData with all required components
    sae_data = classes['SaeVisData'](
        cfg=config,
        feature_stats=feature_stats,
        model=MockModel(tokenizer) if tokenizer is not None else None
    )

    # Let's do a detailed trace of the data we're working with
    print("\n=== DETAILED DATA TRACE ===")

    # Collect data for each feature to analyze
    feature_data_summary = {}
    for feature_idx, feature_data in jax_data.feature_data_dict.items():
        max_activation = "Unknown"
        activation_density = "Unknown"

        # Extract key metrics for debugging
        if 'acts_histogram_data' in feature_data:
            if 'title' in feature_data['acts_histogram_data']:
                title = feature_data['acts_histogram_data']['title']
                # Parse max value from title
                if "MAX =" in title:
                    max_str = title.split("MAX =")[1].strip()
                    max_activation = max_str
                # Parse density from title
                if "DENSITY =" in title:
                    density_str = title.split("DENSITY =")[1].split("<br>")[0].strip()
                    activation_density = density_str

        feature_data_summary[feature_idx] = {
            "max_activation": max_activation,
            "activation_density": activation_density,
        }

    # Print the data summary for each feature
    for feature_idx, summary in sorted(feature_data_summary.items()):
        print(f"Feature {feature_idx}: Max={summary['max_activation']}, Density={summary['activation_density']}")

    # Keep keys as the original feature indices but add extra metadata to track them
    print("\n=== PREPARING OUTPUT DATA ===")

    # Convert each feature data, keeping original keys but with traceable titles
    for feature_idx, feature_data in jax_data.feature_data_dict.items():
        # Create a new properly-structured FeatureData instance
        saed_feature = classes['FeatureData']()

        if 'acts_histogram_data' in feature_data:
            # Store original title for reference
            original_title = feature_data['acts_histogram_data'].get('title', '')
            print(f"Feature {feature_idx} original title: {original_title}")

            # Get max value for reference
            max_val = "unknown"
            if "MAX =" in original_title:
                max_str = original_title.split("MAX =")[1].strip()
                max_val = max_str
            print(f"Feature {feature_idx} max value: {max_val}")

        # Convert each component using our conversion functions
        if 'feature_tables_data' in feature_data:
            saed_feature.feature_tables_data = convert_feature_tables_data(feature_data['feature_tables_data'])

        if 'acts_histogram_data' in feature_data:
            acts_histogram = convert_histogram_data(feature_data['acts_histogram_data'])
            # Add a TRACE ID to help debug the issue
            saed_feature.acts_histogram_data = classes['ActsHistogramData'](**acts_histogram)

        if 'logits_histogram_data' in feature_data:
            logits_histogram = convert_histogram_data(feature_data['logits_histogram_data'])
            saed_feature.logits_histogram_data = classes['LogitsHistogramData'](**logits_histogram)

        if 'logits_table_data' in feature_data:
            saed_feature.logits_table_data = convert_logits_table_data(feature_data['logits_table_data'])

        if 'sequence_data' in feature_data:
            # Even more debug info for sequence data
            seq_data = feature_data['sequence_data']
            if 'seq_group_data' in seq_data:
                groups = seq_data['seq_group_data']
                print(f"Feature {feature_idx} sequence groups: {len(groups)}")
                for i, group in enumerate(groups):
                    if 'title' in group:
                        print(f"  Group {i} title: {group['title']}")

            saed_feature.sequence_data = convert_sequence_data(feature_data['sequence_data'])

        # Keep using original feature indices as keys to avoid any confusion
        # This gives us more information to diagnose the issue
        sae_data.feature_data_dict[feature_idx] = saed_feature
        print(f"Added data for feature {feature_idx} with its original index")

    return sae_data


def save_feature_centric_vis(jax_data, html_path, tokenizer=None, model_name="JAX SAE"):
    """
    Save feature-centric visualization using SAEDashboard

    Args:
        jax_data: Our JAX SaeVisData object
        html_path: Path to save the HTML file
        tokenizer: Optional tokenizer object or function:
                  - If a function, it should convert token IDs to strings
                  - If an object with id_to_piece method (like SentencePiece), it will be adapted
        model_name: Name of the model to display
    """
    if not SAEDASHBOARD_AVAILABLE:
        raise ImportError("SAEDashboard is not installed. Cannot save visualization.")

    classes = _import_saedashboard_classes()

    # Convert our JAX data to native Python types first
    jax_data = convert_to_native_python(jax_data)

    # Create a config with proper settings
    # Sort features to ensure consistent ordering
    sorted_features = sorted(list(jax_data.feature_data_dict.keys())) if hasattr(jax_data, 'feature_data_dict') else [0]

    print(f"Original feature order in data: {list(jax_data.feature_data_dict.keys()) if hasattr(jax_data, 'feature_data_dict') else [0]}")
    print(f"Using sorted feature order: {sorted_features}")

    config = classes['SaeVisConfig'](
        hook_point=f"{model_name} activations",
        features=sorted_features,  # Use consistently sorted features
        use_dfa=True,
        device="cpu",
        dtype="float32",
        minibatch_size_features=10,
        minibatch_size_tokens=50
    )

    # Set up feature-centric layout explicitly
    try:
        from sae_dashboard.components_config import (
            FeatureTablesConfig, ActsHistogramConfig,
            LogitsHistogramConfig, LogitsTableConfig,
            SequencesConfig, Column
        )
        from sae_dashboard.layout import SaeVisLayoutConfig

        # Create a layout with balanced column widths
        # The key is to use appropriate settings for the SequencesConfig
        config.feature_centric_layout = SaeVisLayoutConfig(
            columns=[
                Column(FeatureTablesConfig(), width=300),
                Column(
                    ActsHistogramConfig(),
                    LogitsHistogramConfig(),
                    LogitsTableConfig(n_rows=10),
                    width=300
                ),
                Column(SequencesConfig(
                    stack_mode="stack-quantiles",  # Use stack-all to ensure all groups are shown
                    n_quantiles=5,  # Generate 5 quantile groups
                    buffer=[5, 5],  # Default buffer size (5 tokens before and after)
                    compute_buffer=True,  # Compute data for all tokens in the buffer
                    top_acts_group_size=10,  # Show 10 examples in the top activations group
                    quantile_group_size=3,  # Show 3 examples in each quantile group
                    top_logits_hoverdata=5  # Number of top logits to display in hover data
                ), width=450)  # Standard width
            ],
            height=1000  # Standard height
        )

        # Print config settings to help with debugging
        print(f"SequencesConfig settings:")
        print(f"  stack_mode: {config.feature_centric_layout.columns[2][0].stack_mode}")
        print(f"  n_quantiles: {config.feature_centric_layout.columns[2][0].n_quantiles}")
        print(f"  top_acts_group_size: {config.feature_centric_layout.columns[2][0].top_acts_group_size}")
        print(f"  quantile_group_size: {config.feature_centric_layout.columns[2][0].quantile_group_size}")

        # Save the settings to the config.feature_centric_layout.seq_cfg object
        config.feature_centric_layout.seq_cfg = config.feature_centric_layout.columns[2][0]
        print(f"  group_sizes: {config.feature_centric_layout.seq_cfg.group_sizes}")
        print("Successfully created feature-centric layout")
    except Exception as e:
        print(f"Warning: Error creating layout: {e}")

    # Monkey patch the get_decode_html_safe_fn function to ensure it works with our tokenizer
    original_fn = classes['get_decode_html_safe_fn']

    def get_decode_html_safe_fn_patch(tokenizer, html=False):
        """Custom version of get_decode_html_safe_fn that exactly mimics the original function"""
        from sae_dashboard.utils_fns import process_str_tok
        import torch

        # Create a vocab_dict just like in the original function
        if hasattr(tokenizer, 'vocab') and tokenizer.vocab:
            vocab_dict = {v: k for k, v in tokenizer.vocab.items()}
        elif hasattr(tokenizer, 'id_to_piece'):
            # For SentencePiece tokenizers, build a vocab from id_to_piece
            vocab_dict = {}
            try:
                vocab_size = tokenizer.get_piece_size() if hasattr(tokenizer, 'get_piece_size') else 50000
                for i in range(vocab_size):
                    piece = tokenizer.id_to_piece(i)
                    if piece:
                        vocab_dict[i] = piece
            except:
                # Fall back to a minimal dictionary if needed
                vocab_dict = {}
        else:
            # Create a minimal default vocabulary
            vocab_dict = {}

        def decode(token_id):
            """Exact replica of the original decode function"""
            if isinstance(token_id, int):
                # This is the key line - exactly like the original
                str_tok = vocab_dict.get(token_id, "UNK")
                return process_str_tok(str_tok, html=html)
            else:
                # Handle lists of tokens, just like the original
                if isinstance(token_id, torch.Tensor):
                    token_id = token_id.tolist()
                return [decode(tid) for tid in token_id]

        return decode

    # Replace the function with our patch
    try:
        classes['get_decode_html_safe_fn'] = get_decode_html_safe_fn_patch
    except Exception as e:
        print(f"Warning: Error patching get_decode_html_safe_fn: {e}")

    # Adapt the tokenizer for SAEDashboard
    mock_tokenizer = None
    try:
        if tokenizer is not None:
            # Create a tokenizer object compatible with SAEDashboard
            if hasattr(tokenizer, 'id_to_piece'):
                # Adapt SentencePiece-like tokenizers
                class SPTokenizerAdapter:
                    def __init__(self, sp_tokenizer):
                        self.sp_tokenizer = sp_tokenizer

                        # Build a vocabulary dict identical to what SAEDashboard expects
                        self.vocab = {}
                        try:
                            # Get the total vocabulary size if available
                            vocab_size = sp_tokenizer.get_piece_size() if hasattr(sp_tokenizer, 'get_piece_size') else 32000

                            # Build mappings in both directions:
                            # 1. piece (string) -> id (int) - this is what .vocab is expected to be
                            # 2. id (int) -> piece (string) - this is needed internally
                            id_to_piece_dict = {}

                            print(f"Building vocabulary dictionary from SentencePiece tokenizer (size: {vocab_size})")
                            for i in range(vocab_size):
                                try:
                                    piece = sp_tokenizer.id_to_piece(i)
                                    if piece:  # Only add non-empty pieces
                                        self.vocab[piece] = i
                                        id_to_piece_dict[i] = piece
                                except:
                                    continue

                            # Store the reverse mapping for our internal use
                            self.id_to_piece_dict = id_to_piece_dict
                            print(f"Built vocabulary with {len(self.vocab)} entries")
                        except Exception as e:
                            print(f"Warning: Error building vocabulary: {e}")
                            # Create a minimal vocabulary as fallback
                            for i in range(1000):
                                piece = f"token_{i}"
                                self.vocab[piece] = i
                                self.id_to_piece_dict = {i: piece for i, piece in enumerate(self.vocab.keys())}

                    def decode(self, token_id):
                        """Mimics the exact behavior of the original get_decode_html_safe_fn"""
                        if isinstance(token_id, (list, tuple)):
                            return [self.decode(tid) for tid in token_id]
                        try:
                            tid = int(token_id)
                            # Use the same vocab_dict.get(token_id, "UNK") pattern as the original
                            if tid in self.id_to_piece_dict:
                                return self.id_to_piece_dict[tid]
                            else:
                                # Critical: Return EXACTLY "UNK" like the original function does
                                return "UNK"
                        except:
                            return "UNK"

                    def batch_decode(self, token_ids):
                        """Batch version that maintains the same behavior"""
                        try:
                            return [self.decode(tid) for tid in token_ids]
                        except Exception:
                            return ["UNK"] * (len(token_ids) if hasattr(token_ids, '__len__') else 1)

                mock_tokenizer = SPTokenizerAdapter(tokenizer)

            elif callable(tokenizer):
                # Adapt function-based tokenizers
                class FunctionTokenizerAdapter:
                    def __init__(self, decode_fn):
                        self.decode_fn = decode_fn
                        # Build a vocabulary dictionary that mimics the original SAEDashboard behavior
                        self.vocab = {}  # piece (string) -> id (int)
                        self.id_to_piece_dict = {}  # id (int) -> piece (string)

                        # Try to build a vocabulary
                        print("Building vocabulary from tokenizer function...")
                        vocab_size = 10000  # Try a large vocabulary to catch common tokens
                        for i in range(vocab_size):
                            try:
                                # Try to decode single tokens
                                token = str(self.decode_fn(i))
                                if token and token != "UNK" and not token.startswith("<unk"):
                                    self.vocab[token] = i
                                    self.id_to_piece_dict[i] = token
                            except:
                                continue

                        print(f"Built vocabulary with {len(self.vocab)} entries")

                    def decode(self, token_id):
                        """Mimics the exact behavior of the original get_decode_html_safe_fn"""
                        if isinstance(token_id, (list, tuple)):
                            return [self.decode(tid) for tid in token_id]
                        try:
                            tid = int(token_id)
                            # Use the same pattern as original: vocab_dict.get(token_id, "UNK")
                            if tid in self.id_to_piece_dict:
                                return self.id_to_piece_dict[tid]
                            else:
                                # Critical: Return EXACTLY "UNK" like the original function does
                                return "UNK"
                        except:
                            return "UNK"

                    def batch_decode(self, token_ids):
                        """Batch version that maintains the same behavior"""
                        try:
                            return [self.decode(tid) for tid in token_ids]
                        except Exception:
                            return ["UNK"] * (len(token_ids) if hasattr(token_ids, '__len__') else 1)

                mock_tokenizer = FunctionTokenizerAdapter(tokenizer)
    except Exception as e:
        print(f"Warning: Error creating tokenizer adapter: {e}")
        mock_tokenizer = None

    # If we don't have a valid tokenizer, create a dummy one
    if mock_tokenizer is None:
        try:
            # Create a simple default tokenizer for HTML display
            class SimpleTokenizer:
                def __init__(self):
                    # Create a vocabulary dictionary with the same format as the original
                    self.vocab = {}  # piece (string) -> id (int)

                    # Fill vocabulary with simple token mappings
                    for i in range(5000):  # Create a reasonably large vocab
                        token = f"token_{i}"
                        self.vocab[token] = i

                    # Create a reverse mapping (id -> token) for internal use
                    self.id_to_piece_dict = {i: token for token, i in self.vocab.items()}

                def decode(self, token_id):
                    """Follows exactly the same pattern as the original get_decode_html_safe_fn"""
                    if isinstance(token_id, (list, tuple)):
                        return [self.decode(tid) for tid in token_id]
                    try:
                        tid = int(token_id)
                        # Use the same pattern as original: vocab_dict.get(token_id, "UNK")
                        return self.id_to_piece_dict.get(tid, "UNK")
                    except:
                        return "UNK"

                def batch_decode(self, token_ids):
                    """Batch version that maintains the same behavior"""
                    try:
                        return [self.decode(tid) for tid in token_ids]
                    except Exception:
                        return ["UNK"] * (len(token_ids) if hasattr(token_ids, '__len__') else 1)

            # Use our simple tokenizer for the API
            mock_tokenizer = SimpleTokenizer()
        except Exception as e:
            print(f"Warning: Error creating simple tokenizer: {e}")
            # Will proceed without a tokenizer

    # Convert our data to SAEDashboard format
    saed_data = convert_to_saedashboard_format(jax_data, config=config, tokenizer=mock_tokenizer)

    # Debug info
    n_features = len(saed_data.feature_data_dict) if hasattr(saed_data, 'feature_data_dict') else 0
    print(f"Converting {n_features} features to visualization")

    # Print more detailed debug info about sequence data
    print("\nSequence data debug information:")
    for feature_idx, feature_data in saed_data.feature_data_dict.items():
        if hasattr(feature_data, 'sequence_data') and feature_data.sequence_data:
            seq_data = feature_data.sequence_data
            if hasattr(seq_data, 'sequence_groups') and seq_data.sequence_groups:
                print(f"Feature {feature_idx}: {len(seq_data.sequence_groups)} sequence groups")
                for i, group in enumerate(seq_data.sequence_groups):
                    print(f"  Group {i}: '{group.title}' with {len(group.seq_data) if hasattr(group, 'seq_data') else 0} sequences")

    # Print some diagnostic information about sequence groups
    print("\nSequence groups diagnostic information:")
    for feature_idx, feature_data in saed_data.feature_data_dict.items():
        if hasattr(feature_data, 'sequence_data') and feature_data.sequence_data:
            seq_data = feature_data.sequence_data
            if hasattr(seq_data, 'sequence_groups') and seq_data.sequence_groups:
                print(f"Feature {feature_idx}: {len(seq_data.sequence_groups)} sequence groups")
                for i, group in enumerate(seq_data.sequence_groups):
                    print(f"  Group {i}: '{group.title}' with {len(group.seq_data) if hasattr(group, 'seq_data') else 0} sequences")

    # Call SAEDashboard's save function
    try:
        # Print final data summary for each feature right before HTML generation
        print("\n=== FINAL FEATURE DATA SUMMARY ===")
        for feature_idx, feature_data in saed_data.feature_data_dict.items():
            title = "Unknown"
            if hasattr(feature_data, 'acts_histogram_data'):
                if hasattr(feature_data.acts_histogram_data, 'title'):
                    title = feature_data.acts_histogram_data.title

            print(f"Feature {feature_idx} histogram title: {title}")

        # Use the first feature index as starting point
        all_features = sorted(list(saed_data.feature_data_dict.keys()))
        starting_feature = all_features[0] if all_features else 0

        print(f"\nFinal feature indices: {all_features}")
        print(f"Using feature {starting_feature} as starting point")

        # Now, save the actual file
        classes['save_feature_centric_vis'](
            sae_vis_data=saed_data,
            filename=html_path,
            feature_idx=starting_feature,
            include_only=None,
            separate_files=False
        )
        print(f"Successfully saved visualization to {html_path}")
    except Exception as e:
        print(f"Error saving feature-centric visualization: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always restore the original function
        try:
            classes['get_decode_html_safe_fn'] = original_fn
        except Exception as e:
            print(f"Warning: Error restoring get_decode_html_safe_fn: {e}")


def create_tokenizer_from_list(token_strings_list):
    """
    Create a tokenizer function from a list of token strings.

    Args:
        token_strings_list: A list of token strings where index corresponds to token ID

    Returns:
        A function that maps token IDs to strings
    """
    def tokenizer(token_ids):
        try:
            if isinstance(token_ids, (list, tuple)):
                result = []
                for tid in token_ids:
                    try:
                        tid_int = int(tid)
                        if 0 <= tid_int < len(token_strings_list):
                            result.append(token_strings_list[tid_int])
                        else:
                            result.append("UNK")  # Use exact "UNK" for better SAEDashboard compatibility
                    except (ValueError, TypeError):
                        result.append("UNK")
                return result

            # Single token ID
            try:
                tid_int = int(token_ids)
                if 0 <= tid_int < len(token_strings_list):
                    return token_strings_list[tid_int]
                return "UNK"
            except (ValueError, TypeError):
                return "UNK"
        except Exception as e:
            print(f"Error in tokenizer: {e}")
            if isinstance(token_ids, (list, tuple)):
                return ["UNK"] * len(token_ids)
            return "UNK"

    return tokenizer


def get_unique_token_strings(batch):
    """
    Extract unique token strings from a batch to create a tokenizer.

    Args:
        batch: A tuple of (token_ids, token_strings, intermediate_activations)

    Returns:
        A function that maps token IDs to strings
    """
    token_ids, token_strings, _ = batch[:3]

    # Create a mapping of token ID to token string
    token_map = {}

    # Convert input data to native Python types
    token_ids = convert_to_native_python(token_ids)
    token_strings = convert_to_native_python(token_strings)

    # Handle different formats of token_strings
    if isinstance(token_strings, list) and all(isinstance(s, str) for s in token_strings):
        # If token_strings is already a list of strings, use it directly
        # Create a basic mapping assuming token IDs are sequential
        max_id = max(100, len(token_strings))  # Ensure reasonable size
        tokens_list = ["UNK"] * max_id

        for i, token in enumerate(token_strings):
            if i < max_id:  # Stay within bounds
                tokens_list[i] = token

        return create_tokenizer_from_list(tokens_list)

    # Flatten arrays if needed - for token_ids in matrix form and token_strings as array
    elif hasattr(token_ids, 'shape') and len(token_ids.shape) > 1:
        # Check if token_strings is also a matrix
        if hasattr(token_strings, 'shape') and len(token_strings.shape) > 1:
            for i in range(token_ids.shape[0]):
                for j in range(token_ids.shape[1]):
                    tid = int(token_ids[i][j])
                    tstr = str(token_strings[i][j])
                    token_map[tid] = tstr
        else:
            # If token_strings is flat or different format, create dummy tokens
            for i in range(token_ids.shape[0]):
                for j in range(token_ids.shape[1]):
                    tid = int(token_ids[i][j])
                    token_map[tid] = f"token_{tid}"

    # Convert to a list where index = token ID
    max_token_id = max(token_map.keys()) if token_map else 100  # Default to reasonable size
    tokens_list = ["UNK"] * (max_token_id + 1)

    for tid, tstr in token_map.items():
        tokens_list[tid] = tstr

    return create_tokenizer_from_list(tokens_list)
