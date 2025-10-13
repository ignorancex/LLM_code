# src/__init__.py

"""SAERepGeo: Sparsification and Reconstruction from Representation Geometry

This package contains the implementation of experiments for analyzing the representation geometry
of Sparse Autoencoders (SAEs) in language models, as described in the paper submitted to NeurIPS 2025.
"""

__version__ = "1.0.0"

# Optional: Export key modules for easier access
from .sae_manifold_analyzer import SAEManifoldAnalyzer
from .case1_stratified_manifold import generate_case1_caches
from .case2_representation_structure import generate_case2_cache, analyze_representation_structure
from .case3_intervention_analysis import run_intervention_analysis, CustomSAE
from .utils import setup_logging, setup_device, create_output_dirs