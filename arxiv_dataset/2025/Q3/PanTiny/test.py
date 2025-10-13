#!/usr/bin/env python
# coding=utf-8
"""
Test Script for Trained Pan-sharpening Models
@Description: Test trained models on multiple datasets with different modes

Usage Examples:
    # Test all modes (test128 + full resolution) on all datasets
    python test.py ../Out/expxxxxxx_1234567890

    # Test only reduced resolution (test128)
    python test.py ../Out/expxxxxxx_1234567890 --mode test128

    # Test only full resolution
    python test.py ../Out/expxxxxxx_1234567890 --mode full

    # Test specific datasets
    python test.py ../Out/expxxxxxx_1234567890 --datasets WV2,GF2

Features:
    - Automatically finds the latest experiment directory with UTC timestamp
    - Tests all available model checkpoints (latest.pth, bestPSNR.pth, etc.)
    - Supports both reduced resolution (test128) and full resolution testing
    - Calculates comprehensive metrics (PSNR, SSIM, SAM, QNR, etc.)
    - Output format matches runner.py for consistency
"""
import os
import sys
import argparse
import yaml
import torch
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.metrics import ComprehensiveMetrics
from utils.model_analysis import print_model_analysis


def find_experiment_dirs(base_dir):
    """Find all experiment directories in the base directory"""
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")

    experiment_dirs = []

    # Check if this is a single experiment directory (has checkpoints directly)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        experiment_dirs.append(base_dir)
        print(f"Found single experiment directory: {base_dir}")
        return experiment_dirs

    # Check if this is a batch experiment directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Look for nested experiment structure: item/item/timestamp/
            nested_path = os.path.join(item_path, item)
            if os.path.exists(nested_path):
                # Find timestamp directories
                for subitem in os.listdir(nested_path):
                    subitem_path = os.path.join(nested_path, subitem)
                    if os.path.isdir(subitem_path) and subitem.isdigit():
                        # Check if it has checkpoints
                        checkpoint_path = os.path.join(subitem_path, 'checkpoints')
                        if os.path.exists(checkpoint_path):
                            experiment_dirs.append(subitem_path)
                            print(f"Found experiment: {item} -> {subitem_path}")

    if not experiment_dirs:
        raise ValueError(f"No valid experiment directories found in {base_dir}")

    return experiment_dirs


def find_available_models(checkpoint_dir):
    """Find available model checkpoints"""
    available_models = {}
    
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory does not exist: {checkpoint_dir}")
        return available_models
    
    # Look for common model checkpoint names
    model_patterns = {
        'latest': 'latest.pth',
        'bestPSNR': 'bestPSNR.pth',
        'bestSSIM': 'bestSSIM.pth',
        'bestQNR': 'bestQNR.pth'
    }
    
    for model_name, pattern in model_patterns.items():
        model_path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(model_path):
            available_models[model_name] = model_path
            print(f"Found {model_name} model: {model_path}")
    
    return available_models


def load_experiment_config(experiment_dir):
    """Load experiment configuration from results directory"""
    config_path = os.path.join(experiment_dir, 'results', 'config.yml')
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded experiment config from: {config_path}")
    return config


def create_test_config(original_config, data_dirs, test_mode, datasets=None):
    """Create a minimal test configuration from original config"""
    # Default datasets if not specified
    if datasets is None:
        datasets = ['WV2', 'WV3', 'GF2']

    # Create a minimal config for testing
    test_config = {
        'algorithm': original_config['algorithm'],
        'gpu_mode': original_config.get('gpu_mode', True),
        'data': {
            'n_colors': 4,
            'batch_size': 1,  # Use batch size 1 for testing
            'patch_size': original_config.get('data', {}).get('patch_size', 128),
            'normalize': original_config.get('data', {}).get('normalize', True),
            'upsacle': original_config.get('data', {}).get('upsacle', 4),  # Note: original typo
            'rgb_range': original_config.get('data', {}).get('rgb_range', 255)
        },
        'model': original_config.get('model', {}),
        'evaluation': {
            'save_all_test_images': True,
            'test_models': {
                'latest': True,
                'bestPSNR': True,
                'bestSSIM': False,
                'bestQNR': False
            },
            'full_resolution_in_val': False,
            'full_resolution_in_test': True,
            'eval_interval': 1,
            'save_interval': 1,
            'use_YCbCr': False
        }
    }

    # Set up data directories based on test mode and selected datasets
    if test_mode in ['test128', 'all']:
        eval_dirs = {}
        for dataset in datasets:
            if dataset in data_dirs and 'test128' in data_dirs[dataset]:
                eval_dirs[dataset] = data_dirs[dataset]['test128']
        test_config['data_dirs'] = {'eval': eval_dirs}

    if test_mode in ['full', 'all']:
        full_dirs = {}
        for dataset in datasets:
            if dataset in data_dirs and 'full' in data_dirs[dataset]:
                full_dirs[dataset] = {
                    'enabled': True,
                    'path': data_dirs[dataset]['full']
                }
        test_config['full_data_dirs'] = full_dirs

    # Add required schedule fields with defaults
    test_config['schedule'] = {
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'weight_decay': 0.0,
        'lr_scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'eval_interval': 1,
        'save_interval': 1
    }

    # Add data usage config
    test_config['data_usage'] = {
        'datasets': datasets,
        'usage_percent': 1.0,
        'data_seed': 42,
        'balance_datasets': False
    }

    # Add loss config (minimal, will use default L1)
    test_config['loss'] = original_config.get('loss', {
        'L1': {
            'enabled': True,
            'weight': 1.0
        }
    })

    # Add dummy training config (not used in testing)
    test_config['nEpochs'] = 1
    test_config['name'] = 'test'
    test_config['output_dir'] = '/tmp'

    # Add data source configuration (required by data loader)
    test_config['source_ms'] = original_config.get('source_ms', 'ms')
    test_config['source_pan'] = original_config.get('source_pan', 'pan')
    test_config['threads'] = original_config.get('threads', 4)
    test_config['seed'] = original_config.get('seed', 123)
    test_config['gpus'] = original_config.get('gpus', [0])

    return test_config


class TestOnlySolver:
    """Simplified solver for testing only"""

    def __init__(self, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('gpu_mode', True) else 'cpu')
        self.few_shot = config.get('few_shot', 0)

        # Get datasets from config
        self.datasets = config.get('data_usage', {}).get('datasets', ['WV2', 'WV3', 'GF2'])

        # Initialize model
        self._initialize_model()

        # Setup datasets
        self._setup_datasets()

        # Initialize metrics
        self.metrics_calculator = ComprehensiveMetrics(self.datasets)

        # Get model stats
        self.model_stats = self._analyze_model()

    def _initialize_model(self):
        """Initialize the model"""
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        # Get model configuration from config
        model_config = self.cfg.get('model', {})
        base_filter = model_config.get('base_filter', 64)

        self.model = net(
            num_channels=self.cfg['data']['n_colors'],
            base_filter=base_filter,
            args=self.cfg
        )

        self.model.to(self.device)
        self.model.eval()

    def _analyze_model(self):
        """Analyze model parameters and FLOPs"""
        try:
            from utils.model_analysis import InferenceProfiler

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Calculate model size
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            model_size_mb = (param_size + buffer_size) / 1024 / 1024

            # Calculate FLOPs (using 128x128 input)
            try:
                profiler = InferenceProfiler(self.model, input_shape=(1, 4, 128, 128), device=self.device)
                flops = profiler.get_flops()
            except:
                flops = 0

            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'total_flops': flops
            }
        except Exception as e:
            print(f"Warning: Could not analyze model: {e}")
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'model_size_mb': 0.0,
                'total_flops': 0
            }

    def _setup_datasets(self):
        """Setup datasets for testing"""
        from data.data import MultiDatasetLoader

        self.data_loader_manager = MultiDatasetLoader(self.cfg)

        # Validation data (test128)
        self.val_loaders = {}
        if 'data_dirs' in self.cfg and 'eval' in self.cfg['data_dirs']:
            self.val_loaders = self.data_loader_manager.get_val_loaders(self.datasets)

        # Full resolution test data
        self.full_test_loaders = {}
        if 'full_data_dirs' in self.cfg:
            self.full_test_loaders = self.data_loader_manager.get_full_test_loaders(self.datasets)

    def test_with_model(self, model_path, model_name):
        """Test with specific model"""
        print(f"\n=== Testing with {model_name} model ===")

        # Load the model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['net'])
            print(f"Loaded model from: {model_path}")
            if 'epoch' in checkpoint:
                print(f"Model was trained for {checkpoint['epoch']} epochs")
        else:
            print(f"Warning: Model not found at {model_path}")
            return {}

        self.model.eval()

        # Clear previous results
        self.metrics_calculator = ComprehensiveMetrics(self.datasets)

        # Test on validation datasets (with ground truth)
        for dataset_name, data_loader in self.val_loaders.items():
            if len(data_loader) == 0:
                continue

            print(f"Testing {dataset_name} (with GT, {len(data_loader)} samples)...")

            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    # Few-shot mode: break after N samples
                    if self.few_shot > 0 and i >= self.few_shot:
                        print(f"Few-shot mode: stopping after {self.few_shot} samples")
                        break

                    try:
                        # Handle batch format
                        ms_image, lms_image, pan_image, bms_image, _ = batch

                        # Move to device
                        ms_image = ms_image.to(self.device)
                        lms_image = lms_image.to(self.device)
                        pan_image = pan_image.to(self.device)
                        bms_image = bms_image.to(self.device)

                        # Forward pass
                        prediction = self.model(lms_image, bms_image, pan_image)

                        # Denormalize if needed
                        if self.cfg['data'].get('normalize', True):
                            ms_image = (ms_image + 1) / 2
                            prediction = (prediction + 1) / 2

                        # Calculate metrics
                        self.metrics_calculator.add_reference_result(
                            dataset_name, prediction, ms_image
                        )

                    except Exception as e:
                        print(f"Warning: Error processing sample {i}: {e}")
                        continue

        # Test on full resolution datasets (no ground truth)
        for dataset_name, data_loader in self.full_test_loaders.items():
            if len(data_loader) == 0:
                continue

            print(f"Testing {dataset_name} (full resolution, {len(data_loader)} samples)...")

            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    # Few-shot mode: break after N samples
                    if self.few_shot > 0 and i >= self.few_shot:
                        print(f"Few-shot mode: stopping after {self.few_shot} samples")
                        break

                    try:
                        # Handle batch format
                        ms_image, lms_image, pan_image, bms_image, _ = batch

                        # Move to device
                        lms_image = lms_image.to(self.device)
                        pan_image = pan_image.to(self.device)
                        bms_image = bms_image.to(self.device)

                        # Forward pass
                        prediction = self.model(lms_image, bms_image, pan_image)

                        # Denormalize if needed
                        if self.cfg['data'].get('normalize', True):
                            lms_image = (lms_image + 1) / 2
                            pan_image = (pan_image + 1) / 2
                            prediction = (prediction + 1) / 2

                        # Calculate no-reference metrics
                        self.metrics_calculator.add_no_reference_result(
                            dataset_name, prediction, pan_image, lms_image
                        )

                    except Exception as e:
                        print(f"Warning: Error processing full resolution sample {i}: {e}")
                        continue

        # Print results
        print(f"\n{model_name} Test Results:")
        self.metrics_calculator.print_results()

        # Get final results from metrics calculator
        stats = self.metrics_calculator.get_statistics()
        return stats


def test_model_with_solver(solver, model_path, model_name):
    """Test a specific model using the solver's test method"""
    print(f"\n{'='*80}")
    print(f"Testing {model_name} model")
    print(f"Model path: {model_path}")
    print(f"{'='*80}")

    # Run testing
    results = solver.test_with_model(model_path, model_name)
    return results


def print_comprehensive_summary(all_experiment_results):
    """Print comprehensive summary similar to runner.py"""

    if not all_experiment_results:
        print("No results to display")
        return

    # Determine all datasets used across experiments
    all_datasets = set()
    for exp_data in all_experiment_results.values():
        for model_results in exp_data['results'].values():
            if isinstance(model_results, dict):  # Skip error strings
                all_datasets.update(model_results.keys())

    # Sort datasets for consistent ordering (put common ones first)
    dataset_order = ['WV2', 'WV3', 'GF2', 'jilin']
    sorted_datasets = []
    for ds in dataset_order:
        if ds in all_datasets:
            sorted_datasets.append(ds)
    # Add any other datasets not in the predefined order
    for ds in sorted(all_datasets):
        if ds not in sorted_datasets:
            sorted_datasets.append(ds)

    # Calculate table width dynamically
    base_width = 25 + 12 + 12 + 12 + 10  # Model + Algorithm + Params + FLOPs + Size
    metrics_width = len(sorted_datasets) * 30  # 3 metrics * 10 chars each per dataset
    total_width = base_width + metrics_width

    print("\n" + "="*total_width)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*total_width)

    # Print table header
    header = f"{'Model':<25} {'Algorithm':<12} {'Params(K)':<12} {'FLOPs(M)':<12} {'Size(MB)':<10} "
    for dataset in sorted_datasets:
        header += f"{dataset+'_PSNR':<10} {dataset+'_SSIM':<10} {dataset+'_SAM':<10} "
    print(header)
    print("-" * total_width)

    for exp_name, exp_data in all_experiment_results.items():
        config = exp_data['config']
        results = exp_data['results']
        model_stats = exp_data.get('model_stats', {})

        algorithm = config.get('algorithm', 'unknown')
        params_k = model_stats.get('total_parameters', 0) / 1000
        flops_m = model_stats.get('total_flops', 0) / 1000000
        size_mb = model_stats.get('model_size_mb', 0)

        for model_name, model_results in results.items():
            if isinstance(model_results, str):
                # Error case
                model_display_name = f"{exp_name}_{model_name}"
                row = f"{model_display_name:<25} {algorithm:<12} {params_k:<12.1f} {flops_m:<12.2f} {size_mb:<10.2f} "
                for dataset in sorted_datasets:
                    row += f"{'ERROR':<10} {'ERROR':<10} {'ERROR':<10} "
                print(row)
                continue

            # Extract metrics for each dataset
            model_display_name = f"{exp_name}_{model_name}"
            row = f"{model_display_name:<25} {algorithm:<12} {params_k:<12.1f} {flops_m:<12.2f} {size_mb:<10.2f} "

            for dataset in sorted_datasets:
                psnr = model_results.get(dataset, {}).get('PSNR', {}).get('mean', 0.0)
                ssim = model_results.get(dataset, {}).get('SSIM', {}).get('mean', 0.0)
                sam = model_results.get(dataset, {}).get('SAM', {}).get('mean', 0.0)
                row += f"{psnr:<10.4f} {ssim:<10.4f} {sam:<10.4f} "

            print(row)

    print("="*total_width)

    # Print individual experiment details
    print("\nINDIVIDUAL EXPERIMENT DETAILS:")
    print("-" * 50)

    for exp_name, exp_data in all_experiment_results.items():
        print(f"\nExperiment: {exp_name}")
        print(f"Algorithm: {exp_data['config'].get('algorithm', 'unknown')}")

        model_stats = exp_data.get('model_stats', {})
        if model_stats:
            print(f"Parameters: {model_stats.get('total_parameters', 'N/A'):,}")
            print(f"FLOPs: {model_stats.get('total_flops', 'N/A'):,}")
            print(f"Model Size: {model_stats.get('model_size_mb', 'N/A'):.2f} MB")

        results = exp_data['results']
        for model_name, model_results in results.items():
            print(f"  {model_name}: {'SUCCESS' if not isinstance(model_results, str) else model_results}")

        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description='Test trained pan-sharpening models')
    parser.add_argument('experiment_dir', help='Path to experiment directory (e.g., ../Out/expxxxxxx_{utc})')
    parser.add_argument('--mode', choices=['test128', 'full', 'all'], default='all',
                       help='Test mode: test128 (reduced resolution), full (full resolution), all (both)')
    parser.add_argument('--datasets', default='WV2,WV3,GF2',
                       help='Datasets to test on (comma-separated, available: WV2,WV3,GF2,jilin)')
    parser.add_argument('--few-shot', type=int, default=0,
                       help='Test only first N samples per dataset (0 = test all)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: test only 2 samples per dataset')

    args = parser.parse_args()

    # Set few-shot mode for debug
    if args.debug:
        args.few_shot = 2

    # Parse datasets
    requested_datasets = [ds.strip() for ds in args.datasets.split(',')]
    available_datasets = ['WV2', 'WV3', 'GF2', 'jilin']

    # Validate requested datasets
    invalid_datasets = [ds for ds in requested_datasets if ds not in available_datasets]
    if invalid_datasets:
        print(f"Warning: Invalid datasets specified: {invalid_datasets}")
        print(f"Available datasets: {available_datasets}")
        requested_datasets = [ds for ds in requested_datasets if ds in available_datasets]

    if not requested_datasets:
        print("No valid datasets specified, using default: WV2,WV3,GF2")
        requested_datasets = ['WV2', 'WV3', 'GF2']

    # Default data directories
    default_data_dirs = {
        'WV2': {
            'test128': '../data/WV2_data/test128',
            'full': '../data/fullWV2_dataset'
        },
        'WV3': {
            'test128': '../data/WV3_data/test128',
            'full': '../data/fullWV3_dataset'
        },
        'GF2': {
            'test128': '../data/GF2_data/test128',
            'full': '../data/fullGF2_data/train512'
        },
        'jilin': {
            'test128': '../data/jilin_data/test200',
            'full': '../data/jilin_data/full'
        }
    }

    print("="*80)
    print("PAN-SHARPENING MODEL TESTING")
    print("="*80)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Test mode: {args.mode}")
    print(f"Datasets: {requested_datasets}")
    if args.few_shot > 0:
        print(f"Few-shot mode: {args.few_shot} samples per dataset")
    print("="*80)

    try:
        # Find all experiment directories
        experiment_dirs = find_experiment_dirs(args.experiment_dir)

        all_experiment_results = {}

        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            if exp_name.isdigit():
                # Extract experiment name from parent directories
                parent_path = os.path.dirname(exp_dir)
                exp_name = os.path.basename(parent_path)

            print(f"\n{'='*100}")
            print(f"TESTING EXPERIMENT: {exp_name}")
            print(f"Directory: {exp_dir}")
            print(f"{'='*100}")

            try:
                # Load experiment configuration
                original_config = load_experiment_config(exp_dir)

                # Find available models
                checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
                available_models = find_available_models(checkpoint_dir)

                if not available_models:
                    print(f"No model checkpoints found for {exp_name}!")
                    continue

                print(f"Found {len(available_models)} model(s): {list(available_models.keys())}")

                # Create test configuration
                test_config = create_test_config(original_config, default_data_dirs, args.mode, requested_datasets)
                test_config['few_shot'] = args.few_shot  # Add few-shot config

                # Initialize solver for testing
                print("Initializing solver for testing...")
                solver = TestOnlySolver(test_config)

                # Print model analysis (params, FLOPs)
                print_model_analysis(solver.model, solver.cfg)

                # Test each available model
                exp_results = {}
                for model_name, model_path in available_models.items():
                    try:
                        print(f"\n{'-'*60}")
                        print(f"Testing {model_name} model...")
                        print(f"{'-'*60}")

                        results = test_model_with_solver(solver, model_path, model_name)
                        exp_results[model_name] = results

                    except torch.cuda.OutOfMemoryError as e:
                        print(f"CUDA OOM Error testing {model_name}: {e}")
                        print("Clearing CUDA cache and continuing...")
                        torch.cuda.empty_cache()
                        exp_results[model_name] = "OOM_ERROR"
                        continue
                    except Exception as e:
                        print(f"Error testing {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        exp_results[model_name] = f"ERROR: {str(e)}"
                        continue

                all_experiment_results[exp_name] = {
                    'config': original_config,
                    'results': exp_results,
                    'model_stats': getattr(solver, 'model_stats', {})
                }

            except Exception as e:
                print(f"Error processing experiment {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print comprehensive summary
        print_comprehensive_summary(all_experiment_results)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
