#!/usr/bin/env python
# coding=utf-8
"""
Universal Solver for Pan-sharpening Experiments
@Description: Unified solver supporting comprehensive loss functions and flexible model training
"""
import os
import sys
import yaml
import time
import copy
import math
import shutil
import numpy as np
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from datetime import datetime
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tensorboardX import SummaryWriter
from PIL import Image

from solver.basesolver import BaseSolver
from utils.model_analysis import print_model_analysis, InferenceProfiler
from utils.metrics import ComprehensiveMetrics
from data.data import MultiDatasetLoader, get_train_data, get_val_data, get_full_test_data
from utils.utils import save_config, save_net_config
from utils.config import save_yml
from utils.loss import make_loss, get_available_losses, FrequencyLoss, GradientLoss, EdgeLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def make_comprehensive_loss(loss_config):
    """Create comprehensive loss function prioritizing utils.utils verified losses"""
    loss_functions = {}
    loss_weights = {}
    
    # Create loss name mapping for better compatibility
    loss_mapping = {
        'L1': 'L1',
        'MSE': 'MSE', 
        'SSIM': 'MEF_SSIM',
        'Perceptual': 'VGG22',
        'VGG22': 'VGG22',
        'VGG54': 'VGG54',
        'NewLoss': 'newloss',  # Map NewLoss to newloss (verified)
        'newloss': 'newloss',
        'GAN': 'GAN',
        'CE': 'CE',
        # Custom loss functions (handled separately)
        'Frequency': 'Frequency',
        'Gradient': 'Gradient', 
        'Edge': 'Edge'
    }
    
    # Verified loss types from utils.utils.py
    verified_losses = ['L1', 'MSE', 'MEF_SSIM', 'VGG22', 'VGG54', 'newloss', 'GAN', 'CE']
    
    # Use enhanced loss system from utils.loss
    for loss_name, loss_params in loss_config.items():
        if loss_params.get('enabled', False):
            try:
                # Map loss names using the mapping dictionary
                mapped_name = loss_mapping.get(loss_name, loss_name)
                
                # Priority 1: Use verified losses from utils.utils.py
                if mapped_name in verified_losses:
                    loss_functions[loss_name] = make_loss(mapped_name)
                    print(f"Created VERIFIED loss function: {loss_name} -> {mapped_name} (from utils.utils)")
                    
                # Priority 2: Handle custom implementations for advanced losses
                elif mapped_name == 'Frequency':
                    loss_functions[loss_name] = FrequencyLoss()
                    print(f"Created custom loss function: {loss_name} -> FrequencyLoss")
                elif mapped_name == 'Gradient':
                    loss_functions[loss_name] = GradientLoss()
                    print(f"Created custom loss function: {loss_name} -> GradientLoss")
                elif mapped_name == 'Edge':
                    loss_functions[loss_name] = EdgeLoss()
                    print(f"Created custom loss function: {loss_name} -> EdgeLoss")
                else:
                    print(f"Warning: Unknown loss type {loss_name}, using verified L1 as fallback")
                    loss_functions[loss_name] = make_loss('L1')
                
                loss_weights[loss_name] = loss_params.get('weight', 1.0)
                
            except Exception as e:
                print(f"Error creating loss {loss_name}: {e}")
                print(f"Available verified loss types: {verified_losses}")
                print(f"Available custom loss types: Frequency, Gradient, Edge")
                print(f"Using verified L1 as fallback for {loss_name}")
                loss_functions[loss_name] = make_loss('L1')
                loss_weights[loss_name] = loss_params.get('weight', 1.0)
    
    # Default to L1 if no loss specified
    if not loss_functions:
        loss_functions['L1'] = make_loss('L1')
        loss_weights['L1'] = 1.0
    
    return loss_functions, loss_weights


class UniSolver(BaseSolver):
    """Universal solver for comprehensive pan-sharpening experiments"""
    
    def __init__(self, cfg):
        # Validate configuration first
        self._validate_config(cfg)
        
        # Store configuration
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.timestamp = int(time.time())
        self.start_time = time.time()
        self.epoch = 1
        
        # Create log name for compatibility
        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg.get('data', {}).get('upscale', self.cfg.get('data', {}).get('upsacle', 4))) + '_' + str(self.timestamp)
        
        # Setup directory structure
        self._setup_directories()
        
        # Initialize model
        self._initialize_model()
        
        # Check GPU and move model to GPU first
        self.check_gpu()
        
        # Setup datasets
        self._setup_datasets()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize comprehensive loss function (after GPU setup)
        self._setup_loss()
        
        # Setup comprehensive metrics
        self.metrics_calculator = ComprehensiveMetrics(['WV2', 'WV3', 'GF2'])
        
        # Setup logging
        self._setup_logging()
        
        # Initialize records
        self.records = self._initialize_records()
        
        # Model analysis
        self.model_stats = self._analyze_model()
    
    def _validate_config(self, cfg):
        """Validate configuration file for required fields"""
        required_fields = ['algorithm', 'nEpochs', 'data', 'schedule']
        for field in required_fields:
            if field not in cfg:
                raise ValueError(f"Missing required config field: {field}")
        
        # Validate data configuration
        if 'n_colors' not in cfg['data']:
            raise ValueError("Missing 'n_colors' in data configuration")
        
        if cfg['data']['n_colors'] != 4:
            raise ValueError("Currently only supports n_colors=4 for pan-sharpening")
        
        # Validate schedule configuration
        required_schedule_fields = ['lr', 'beta1', 'beta2', 'epsilon', 'weight_decay']
        for field in required_schedule_fields:
            if field not in cfg['schedule']:
                raise ValueError(f"Missing required schedule field: {field}")
        
        print("Configuration validation passed")
    
    def _setup_directories(self):
        """Setup directory structure"""
        experiment_name = self.cfg.get('name', 'experiment')
        output_dir = self.cfg.get('output_dir', '../Out')
        
        # Main experiment directory
        self.experiment_dir = os.path.join(output_dir, experiment_name, str(self.timestamp))
        
        # Sub-directories
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.results_dir = os.path.join(self.experiment_dir, 'results')
        self.test_results_dir = os.path.join(self.results_dir, 'test_results')
        
        # Create all directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Save config copy
        config_copy_path = os.path.join(self.results_dir, 'config.yml')
        with open(config_copy_path, 'w') as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def _initialize_model(self):
        """Initialize the model"""
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        
        assert (self.cfg['data']['n_colors'] == 4)
        
        # Get model configuration from config
        model_config = self.cfg.get('model', {})
        base_filter = model_config.get('base_filter', 64)
        
        self.model = net(
            num_channels=self.cfg['data']['n_colors'],
            base_filter=base_filter,
            args=self.cfg
        )
    
    def _setup_datasets(self):
        """Setup datasets"""
        self.data_loader_manager = MultiDatasetLoader(self.cfg)
        
        # Training data
        train_datasets = self.cfg['data_usage']['datasets']
        train_loaders = self.data_loader_manager.get_train_loaders(train_datasets)
        
        if len(train_loaders) == 1:
            self.train_loader = list(train_loaders.values())[0]
        else:
            # Combine multiple datasets
            combined_datasets = []
            for loader in train_loaders.values():
                combined_datasets.append(loader.dataset)
            combined_dataset = ConcatDataset(combined_datasets)
            self.train_loader = DataLoader(combined_dataset, batch_size=self.cfg['data']['batch_size'], shuffle=True)
        
        # Validation data
        self.val_loaders = self.data_loader_manager.get_val_loaders(['WV2', 'WV3', 'GF2'])
        
        # Full resolution test data
        self.full_test_loaders = self.data_loader_manager.get_full_test_loaders(['WV2', 'WV3', 'GF2'])
        
        print(f"Training on: {train_datasets}")
        print(f"Validation datasets: {list(self.val_loaders.keys())}")
        print(f"Full resolution test datasets: {list(self.full_test_loaders.keys())}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        lr = float(self.cfg['schedule']['lr'])
        weight_decay = float(self.cfg['schedule']['weight_decay'])
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(float(self.cfg['schedule']['beta1']), float(self.cfg['schedule']['beta2'])),
            eps=float(self.cfg['schedule']['epsilon']),
            weight_decay=weight_decay
        )
        
        # Scheduler
        lr_config = self.cfg['schedule']['lr_scheduler']
        if lr_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.nEpochs, eta_min=lr_config.get('eta_min', 1e-6)
            )
        elif lr_config['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_config.get('step_size', 50), gamma=lr_config.get('gamma', 0.5)
            )
        else:
            self.scheduler = None
    
    def _setup_loss(self):
        """Setup comprehensive loss function"""
        loss_config = self.cfg.get('loss', {})
        self.loss_functions, self.loss_weights = make_comprehensive_loss(loss_config)
        
        # Move loss functions to GPU if model is on GPU
        if next(self.model.parameters()).is_cuda:
            for name, loss_fn in self.loss_functions.items():
                self.loss_functions[name] = loss_fn.cuda()
        
        print(f"Loss functions: {list(self.loss_functions.keys())}")
        print(f"Loss weights: {self.loss_weights}")
    
    def _setup_logging(self):
        """Setup logging"""
        self.writer = SummaryWriter(self.log_dir)
        
        # Save configurations
        net_config_path = os.path.join(self.results_dir, 'net.txt')
        with open(net_config_path, 'w') as f:
            f.write(str(self.model))
        
        # Log dataset information
        dataset_info_path = os.path.join(self.results_dir, 'dataset_info.txt')
        with open(dataset_info_path, 'w') as f:
            f.write(f"Training datasets: {self.cfg['data_usage']['datasets']}\n")
            f.write(f"Validation datasets: {list(self.val_loaders.keys())}\n")
            f.write(f"Full test datasets: {list(self.full_test_loaders.keys())}\n")
    
    def _initialize_records(self):
        """Initialize comprehensive records - Match ResearchSolver logic"""
        records = {
            'Epoch': [],
            'Loss': [],
            'LR': []
        }
        
        # Add records for each dataset and metric
        datasets = ['WV2', 'WV3', 'GF2']
        ref_metrics = ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']
        no_ref_metrics = ['D_lambda', 'D_s', 'QNR']
        
        for dataset in datasets:
            for metric in ref_metrics:
                records[f'{dataset}_{metric}'] = []
            for metric in no_ref_metrics:
                records[f'{dataset}_{metric}_full'] = []
        
        # Average metrics across datasets
        for metric in ref_metrics:
            records[f'Avg_{metric}'] = []
        for metric in no_ref_metrics:
            records[f'Avg_{metric}_full'] = []
        
        return records
    
    def _analyze_model(self):
        """Analyze model parameters and FLOPs"""
        try:
            # Ensure model is on the correct device
            device = next(self.model.parameters()).device
            
            # Basic statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            
            # Try to calculate FLOPs - but skip if likely to cause device issues
            try:
                # Check if we're on GPU and thop might cause issues
                if device.type == 'cuda':
                    print("Skipping THOP FLOPs calculation on GPU to avoid device conflicts")
                    # Use parameter-based estimation for GPU
                    flops = total_params * 2 * 128 * 128
                else:
                    flops = self._calculate_flops_backup()
            except Exception as e:
                print(f"FLOPs calculation failed: {e}, using estimation")
                # Rough estimation: assume 2 FLOPs per parameter for typical CNN
                flops = total_params * 2 * 128 * 128
            
            return {
                'total_parameters': total_params,
                'total_flops': flops,
                'model_size_mb': model_size_mb
            }
        except Exception as e:
            print(f"Model analysis failed: {e}, using backup calculation")
            return self._get_model_stats_backup()
    
    def _calculate_flops_backup(self):
        """Backup method to calculate FLOPs (CPU only to avoid device conflicts)"""
        try:
            from thop import profile
            
            # Create a CPU copy of the model for FLOPs calculation
            model_cpu = type(self.model)(
                num_channels=self.cfg['data']['n_colors'],
                base_filter=self.cfg.get('model', {}).get('base_filter', 64),
                args=self.cfg
            )
            model_cpu.load_state_dict(self.model.state_dict())
            model_cpu.eval()
            
            # Create sample inputs on CPU
            upscale = self.cfg['data'].get('upscale', self.cfg['data'].get('upsacle', 4))
            patch_size = self.cfg['data'].get('patch_size', 32)
            hr_size = patch_size * upscale
            
            l_ms = torch.randn(1, 4, patch_size, patch_size)
            b_ms = torch.randn(1, 4, hr_size, hr_size)
            x_pan = torch.randn(1, 1, hr_size, hr_size)
            
            with torch.no_grad():
                flops, params = profile(model_cpu, inputs=(l_ms, b_ms, x_pan), verbose=False)
            
            # Clean up
            del model_cpu
            
            return flops
        except Exception as e:
            print(f"THOP FLOPs calculation failed: {e}, using parameter-based estimation")
            # If thop fails, use rough estimation
            total_params = sum(p.numel() for p in self.model.parameters())
            # Rough estimation: assume 2 FLOPs per parameter for typical CNN
            return total_params * 2 * 128 * 128
    
    def _get_model_stats_backup(self):
        """Backup method to get model statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'total_flops': self._calculate_flops_backup(),
            'model_size_mb': model_size_mb
        }
    
    def compute_loss(self, pred, target):
        """Compute comprehensive loss"""
        total_loss = 0
        loss_components = {}
        
        for name, loss_fn in self.loss_functions.items():
            component_loss = loss_fn(pred, target)
            weighted_loss = component_loss * self.loss_weights[name]
            total_loss += weighted_loss
            loss_components[name] = component_loss.item()
        
        return total_loss, loss_components
    
    def train(self):
        """Training loop"""
        self.model.train()
        epoch_loss = 0
        loss_components_sum = {}
        
        # Reset GPU memory stats for monitoring
        # if torch.cuda.is_available():
        #     torch.cuda.reset_peak_memory_stats()
        # disable for speed
        
        with tqdm(total=len(self.train_loader), desc=f'Epoch {self.epoch}/{self.nEpochs}') as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Prepare data - handle both 4 and 5 element batches
                if len(batch) == 5:
                    ms_image, lms_image, pan_image, bms_image, filename = batch
                else:
                    ms_image, lms_image, pan_image, bms_image = batch
                
                if self.cfg.get('gpu_mode', True) and torch.cuda.is_available():
                    ms_image = ms_image.cuda()
                    lms_image = lms_image.cuda()
                    pan_image = pan_image.cuda()
                    bms_image = bms_image.cuda()
                
                # Forward pass
                self.optimizer.zero_grad()
                sr_image = self.model(lms_image, bms_image, pan_image)
                
                # Compute loss
                loss, loss_components = self.compute_loss(sr_image, ms_image)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                for name, value in loss_components.items():
                    if name not in loss_components_sum:
                        loss_components_sum[name] = 0
                    loss_components_sum[name] += value
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                pbar.update(1)
                
                # Memory monitoring - moved to epoch level for better performance
        
        # Average loss over epoch
        avg_epoch_loss = epoch_loss / len(self.train_loader)
        
        # Report peak memory usage and current memory status
        # if torch.cuda.is_available():
        #     peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        #     current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        #     print(f"Epoch {self.epoch} GPU memory - Peak: {peak_memory:.2f} GB, Current: {current_memory:.2f} GB")
        #     if peak_memory > 8.0:  # Warning if over 8GB
        #         print(f"Warning: High GPU memory usage detected: {peak_memory:.2f} GB")
        # disable
        
        # Log loss components
        for name, total_loss in loss_components_sum.items():
            avg_component_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar(f'Loss/{name}', avg_component_loss, self.epoch)
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        return avg_epoch_loss
    
    def eval(self):
        """Comprehensive evaluation on all datasets - Match ResearchSolver logic"""
        self.model.eval()
        
        print(f"\nEvaluating epoch {self.epoch}...")
        
        # Reset metrics calculator for this epoch
        self.metrics_calculator = ComprehensiveMetrics(['WV2', 'WV3', 'GF2'])
        
        # Evaluate on validation datasets (with ground truth)
        for dataset_name, data_loader in self.val_loaders.items():
            print(f"Evaluating {dataset_name} (with GT, {len(data_loader)} samples)...")
            
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f'{dataset_name} Val'):
                    # Handle batch format like ResearchSolver
                    ms_image, lms_image, pan_image, bms_image, _ = (
                        Variable(batch[0]), Variable(batch[1]), 
                        Variable(batch[2]), Variable(batch[3]), batch[4]
                    )
                    
                    if self.cfg.get('gpu_mode', True) and torch.cuda.is_available():
                        ms_image = ms_image.cuda()
                        lms_image = lms_image.cuda()
                        pan_image = pan_image.cuda()
                        bms_image = bms_image.cuda()
                    
                    prediction = self.model(lms_image, bms_image, pan_image)
                    
                    # Add results to metrics calculator
                    self.metrics_calculator.add_reference_result(
                        dataset_name, prediction, ms_image
                    )
        
        # Evaluate on full resolution datasets (no ground truth) - only if enabled
        eval_config = self.cfg.get('evaluation', {})
        run_full_resolution = eval_config.get('full_resolution_in_val', False)
        
        if run_full_resolution:
            for dataset_name, data_loader in self.full_test_loaders.items():
                print(f"Evaluating {dataset_name} (full resolution, {len(data_loader)} samples)...")
                
                with torch.no_grad():
                    for batch in tqdm(data_loader, desc=f'{dataset_name} Full'):
                        try:
                            # Handle full resolution data format like ResearchSolver
                            ms_image, lms_image, pan_image, bms_image, _ = (
                                Variable(batch[0]), Variable(batch[1]), 
                                Variable(batch[2]), Variable(batch[3]), batch[4]
                            )
                            
                            if self.cfg.get('gpu_mode', True) and torch.cuda.is_available():
                                lms_image = lms_image.cuda()
                                pan_image = pan_image.cuda()
                                bms_image = bms_image.cuda()
                            
                            prediction = self.model(lms_image, bms_image, pan_image)
                            
                            # Calculate no-reference metrics like ResearchSolver
                            self.metrics_calculator.add_no_reference_result(
                                dataset_name, prediction, pan_image, lms_image
                            )
                        except Exception as e:
                            print(f"Warning: Error processing full resolution sample: {e}")
                            continue
        else:
            print("Full resolution evaluation disabled during validation (full_resolution_in_val=false)")
        
        # Get statistics and update records
        stats = self.metrics_calculator.get_statistics()
        averages = self.metrics_calculator.get_average_metrics()
        
        # Update records - only if not already updated in this epoch
        if len(self.records['Epoch']) == 0 or self.records['Epoch'][-1] != self.epoch:
            self.records['Epoch'].append(self.epoch)
        
        # Update individual dataset records
        for dataset_name in ['WV2', 'WV3', 'GF2']:
            if dataset_name in stats:
                for metric in ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']:
                    key = f'{dataset_name}_{metric}'
                    if metric in stats[dataset_name] and key in self.records:
                        self.records[key].append(stats[dataset_name][metric]['mean'])
                        # Log to TensorBoard
                        self.writer.add_scalar(f'Val_{dataset_name}/{metric}', 
                                             stats[dataset_name][metric]['mean'], self.epoch)
                
                for metric in ['D_lambda', 'D_s', 'QNR']:
                    key = f'{dataset_name}_{metric}_full'
                    if metric in stats[dataset_name] and key in self.records:
                        self.records[key].append(stats[dataset_name][metric]['mean'])
                        # Log to TensorBoard
                        self.writer.add_scalar(f'Full_{dataset_name}/{metric}', 
                                             stats[dataset_name][metric]['mean'], self.epoch)
        
        # Update average records
        for metric in ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']:
            key = f'Avg_{metric}'
            if key in self.records:
                self.records[key].append(averages[metric])
                self.writer.add_scalar(f'Avg_Val/{metric}', averages[metric], self.epoch)
        
        for metric in ['D_lambda', 'D_s', 'QNR']:
            key = f'Avg_{metric}_full'
            if key in self.records:
                self.records[key].append(averages[metric])
                self.writer.add_scalar(f'Avg_Full/{metric}', averages[metric], self.epoch)
        
        # Print results
        self.metrics_calculator.print_results()
        
        # Log summary like ResearchSolver
        log_str = f'Universal Eval Epoch {self.epoch}: '
        for metric in ['PSNR', 'SSIM', 'SAM', 'ERGAS']:
            log_str += f'Avg_{metric}={averages[metric]:.4f}, '
        for metric in ['D_lambda', 'D_s', 'QNR']:
            log_str += f'Avg_{metric}={averages[metric]:.4f}, '
        
        print(log_str.rstrip(', '))
        
        # Log summary to file
        eval_log_file = os.path.join(self.results_dir, 'evaluation_log.txt')
        with open(eval_log_file, 'a') as f:
            f.write(log_str.rstrip(', ') + '\n')
        
        # Return simple results for compatibility
        simple_results = {}
        for dataset_name in ['WV2', 'WV3', 'GF2']:
            if dataset_name in stats:
                simple_results[dataset_name] = {}
                for metric in ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']:
                    if metric in stats[dataset_name]:
                        simple_results[dataset_name][metric] = stats[dataset_name][metric]['mean']
        
        return simple_results
    
    def check_gpu(self):
        """Check GPU availability"""
        if torch.cuda.is_available() and self.cfg.get('gpu_mode', True):
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save comprehensive checkpoint with best model tracking"""
        ckp = {
            'epoch': epoch,
            'records': self.records,
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
            'model_stats': self.model_stats,
            'loss_functions': {name: type(loss_fn).__name__ for name, loss_fn in self.loss_functions.items()},
            'loss_weights': self.loss_weights
        }
        
        # Add scheduler state if available
        if self.scheduler:
            ckp['scheduler'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(ckp, latest_path)
        
        # Save best models based on different metrics
        if self.cfg.get('save_best', True) and len(self.records['Epoch']) > 0:
            import shutil
            
            # Best average PSNR
            if 'WV2_PSNR' in self.records and self.records['WV2_PSNR']:
                current_avg_psnr = (
                    self.records['WV2_PSNR'][-1] + 
                    self.records['WV3_PSNR'][-1] + 
                    self.records['GF2_PSNR'][-1]
                ) / 3
                
                # Check if this is the best average PSNR so far
                best_avg_psnr = float('-inf')
                for i in range(len(self.records['Epoch'])):
                    if (len(self.records['WV2_PSNR']) > i and 
                        len(self.records['WV3_PSNR']) > i and 
                        len(self.records['GF2_PSNR']) > i):
                        avg_psnr = (self.records['WV2_PSNR'][i] + 
                                   self.records['WV3_PSNR'][i] + 
                                   self.records['GF2_PSNR'][i]) / 3
                        if avg_psnr > best_avg_psnr:
                            best_avg_psnr = avg_psnr
                
                if current_avg_psnr >= best_avg_psnr:
                    shutil.copy(latest_path, os.path.join(self.checkpoint_dir, 'bestPSNR.pth'))
            
            # Best average SSIM
            if 'WV2_SSIM' in self.records and self.records['WV2_SSIM']:
                current_avg_ssim = (
                    self.records['WV2_SSIM'][-1] + 
                    self.records['WV3_SSIM'][-1] + 
                    self.records['GF2_SSIM'][-1]
                ) / 3
                
                best_avg_ssim = float('-inf')
                for i in range(len(self.records['Epoch'])):
                    if (len(self.records['WV2_SSIM']) > i and 
                        len(self.records['WV3_SSIM']) > i and 
                        len(self.records['GF2_SSIM']) > i):
                        avg_ssim = (self.records['WV2_SSIM'][i] + 
                                   self.records['WV3_SSIM'][i] + 
                                   self.records['GF2_SSIM'][i]) / 3
                        if avg_ssim > best_avg_ssim:
                            best_avg_ssim = avg_ssim
                
                if current_avg_ssim >= best_avg_ssim:
                    shutil.copy(latest_path, os.path.join(self.checkpoint_dir, 'bestSSIM.pth'))
            
            # Best average QNR (from full resolution if available)
            qnr_keys = ['WV2_QNR', 'WV3_QNR', 'GF2_QNR']
            if any(key in self.records and self.records[key] for key in qnr_keys):
                qnr_values = []
                for key in qnr_keys:
                    if key in self.records and self.records[key]:
                        qnr_values.append(self.records[key][-1])
                
                if qnr_values:
                    current_avg_qnr = sum(qnr_values) / len(qnr_values)
                    
                    best_avg_qnr = float('-inf')
                    for i in range(len(self.records['Epoch'])):
                        epoch_qnr_values = []
                        for key in qnr_keys:
                            if key in self.records and len(self.records[key]) > i:
                                epoch_qnr_values.append(self.records[key][i])
                        
                        if epoch_qnr_values:
                            avg_qnr = sum(epoch_qnr_values) / len(epoch_qnr_values)
                            if avg_qnr > best_avg_qnr:
                                best_avg_qnr = avg_qnr
                    
                    if current_avg_qnr >= best_avg_qnr:
                        shutil.copy(latest_path, os.path.join(self.checkpoint_dir, 'bestQNR.pth'))
        
        print(f"Checkpoint saved: epoch {epoch}")
        if is_best:
            print("  -> New best model saved!")
    
    def generate_final_report(self):
        """Generate comprehensive final experiment report with testing"""
        report_file = os.path.join(self.results_dir, 'experiment_report.txt')
        
        total_time = time.time() - self.start_time
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNIVERSAL SOLVER FINAL REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.cfg.get('name', 'experiment')}\n")
            f.write(f"Total Training Time: {total_time/3600:.2f} hours\n\n")
            
            # Experiment Configuration
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model: {self.cfg['algorithm']}\n")
            f.write(f"Training Dataset: {self.cfg['data_usage']['datasets']}\n")
            f.write(f"Evaluation Datasets: WV2, WV3, GF2\n")
            f.write(f"Epochs: {self.cfg['nEpochs']}\n")
            f.write(f"Batch Size: {self.cfg['data']['batch_size']}\n")
            f.write(f"Learning Rate: {self.cfg['schedule']['lr']}\n")
            f.write(f"Data Augmentation: {self.cfg['data'].get('augmentation', {}).get('enabled', False)}\n\n")
            
            # Model Information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Parameters: {self.model_stats['total_parameters']:,}\n")
            flops = self.model_stats['total_flops']
            if flops > 1e9:
                f.write(f"Model FLOPs: {flops/1e9:.2f} G\n")
            elif flops > 1e6:
                f.write(f"Model FLOPs: {flops/1e6:.2f} M\n")
            else:
                f.write(f"Model FLOPs: {flops:,}\n")
            f.write(f"Model Size: {self.model_stats['model_size_mb']:.2f} MB\n\n")
            
            # Loss Configuration
            f.write("LOSS CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            for name, weight in self.loss_weights.items():
                f.write(f"  {name}: weight={weight}\n")
            f.write("\n")
            
            # Best Results Summary
            f.write("BEST RESULTS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            # Reference metrics (with ground truth)
            ref_metrics = ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']
            f.write("Reference Metrics (with ground truth):\n")
            for metric in ref_metrics:
                avg_values = []
                for dataset in ['WV2', 'WV3', 'GF2']:
                    key = f'{dataset}_{metric}'
                    if key in self.records and self.records[key]:
                        if metric in ['PSNR', 'SSIM', 'CC']:
                            best_val = max(self.records[key])
                        else:  # SAM, ERGAS - lower is better
                            best_val = min(self.records[key])
                        avg_values.append(best_val)
                        best_epoch = self.records['Epoch'][self.records[key].index(best_val)]
                        f.write(f"  Best {dataset} {metric}: {best_val:.4f} (Epoch {best_epoch})\n")
                
                if avg_values:
                    avg_best = sum(avg_values) / len(avg_values)
                    f.write(f"  Average Best {metric}: {avg_best:.4f}\n")
            
            f.write("\nNo-Reference Metrics (full resolution):\n")
            no_ref_metrics = ['D_lambda', 'D_s', 'QNR']
            for metric in no_ref_metrics:
                avg_values = []
                for dataset in ['WV2', 'WV3', 'GF2']:
                    key = f'{dataset}_{metric}'
                    if key in self.records and self.records[key]:
                        if metric == 'QNR':
                            best_val = max(self.records[key])
                        else:  # D_lambda, D_s - lower is better
                            best_val = min(self.records[key])
                        avg_values.append(best_val)
                        best_epoch = self.records['Epoch'][self.records[key].index(best_val)]
                        f.write(f"  Best {dataset} {metric}: {best_val:.4f} (Epoch {best_epoch})\n")
                
                if avg_values:
                    avg_best = sum(avg_values) / len(avg_values)
                    f.write(f"  Average Best {metric}: {avg_best:.4f}\n")
            
            # Saved Models
            f.write("\nSAVED MODELS:\n")
            f.write("-" * 40 + "\n")
            saved_models = []
            if os.path.exists(os.path.join(self.checkpoint_dir, 'latest.pth')):
                saved_models.append('latest.pth (Final model)')
            if os.path.exists(os.path.join(self.checkpoint_dir, 'bestPSNR.pth')):
                saved_models.append('bestPSNR.pth (Best Average PSNR)')
            if os.path.exists(os.path.join(self.checkpoint_dir, 'bestSSIM.pth')):
                saved_models.append('bestSSIM.pth (Best Average SSIM)')
            if os.path.exists(os.path.join(self.checkpoint_dir, 'bestQNR.pth')):
                saved_models.append('bestQNR.pth (Best Average QNR)')
            
            for model in saved_models:
                f.write(f"  {model}\n")
            
            f.write("\nExperiment Directory: " + self.experiment_dir + "\n")
            f.write("Checkpoints: " + self.checkpoint_dir + "\n")
            f.write("TensorBoard Logs: " + self.log_dir + "\n")
            f.write("Results: " + self.results_dir + "\n")
            
            # Perform final testing with saved models
            f.write("\n" + "="*80 + "\n")
            f.write("FINAL TEST RESULTS WITH SAVED MODELS\n")
            f.write("="*80 + "\n")
            
            # Configure which models to test based on config
            test_config = self.cfg.get('evaluation', {}).get('test_models', {
                'latest': True,
                'bestPSNR': True,
                'bestSSIM': False,
                'bestQNR': False
            })
            
            test_models = []
            if test_config.get('latest', True) and os.path.exists(os.path.join(self.checkpoint_dir, 'latest.pth')):
                test_models.append(('latest.pth', 'Latest'))
            if test_config.get('bestPSNR', True) and os.path.exists(os.path.join(self.checkpoint_dir, 'bestPSNR.pth')):
                test_models.append(('bestPSNR.pth', 'BestPSNR'))
            if test_config.get('bestSSIM', False) and os.path.exists(os.path.join(self.checkpoint_dir, 'bestSSIM.pth')):
                test_models.append(('bestSSIM.pth', 'BestSSIM'))
            if test_config.get('bestQNR', False) and os.path.exists(os.path.join(self.checkpoint_dir, 'bestQNR.pth')):
                test_models.append(('bestQNR.pth', 'BestQNR'))
            
            all_test_results = {}
            for model_file, model_name in test_models:
                model_path = os.path.join(self.checkpoint_dir, model_file)
                test_results = self.test_with_model(model_path, model_name)
                all_test_results[model_name] = test_results
                
                # Write test results to report
                f.write(f"\n{model_name} Model Test Results:\n")
                f.write("-" * 50 + "\n")
                
                # Reference metrics table
                f.write("Reference Metrics (with ground truth):\n")
                f.write(f"{'Dataset':<8} {'PSNR':<8} {'SSIM':<8} {'CC':<8} {'SAM':<8} {'ERGAS':<8}\n")
                f.write("-" * 56 + "\n")
                
                for dataset in ['WV2', 'WV3', 'GF2']:
                    if dataset in test_results:
                        stats = test_results[dataset]
                        psnr = stats.get('PSNR', {}).get('mean', 0.0)
                        ssim = stats.get('SSIM', {}).get('mean', 0.0)
                        cc = stats.get('CC', {}).get('mean', 0.0)
                        sam = stats.get('SAM', {}).get('mean', 0.0)
                        ergas = stats.get('ERGAS', {}).get('mean', 0.0)
                        f.write(f"{dataset:<8} {psnr:<8.4f} {ssim:<8.4f} {cc:<8.4f} {sam:<8.4f} {ergas:<8.4f}\n")
                
                # No-reference metrics table
                f.write("\nNo-Reference Metrics (full resolution):\n")
                f.write(f"{'Dataset':<12} {'D_lambda':<10} {'D_s':<10} {'QNR':<10}\n")
                f.write("-" * 45 + "\n")
                
                for dataset in ['WV2', 'WV3', 'GF2']:
                    if dataset in test_results:
                        stats = test_results[dataset]
                        d_lambda = stats.get('D_lambda', {}).get('mean', 0.0)
                        d_s = stats.get('D_s', {}).get('mean', 0.0)
                        qnr = stats.get('QNR', {}).get('mean', 0.0)
                        f.write(f"{dataset:<12} {d_lambda:<10.4f} {d_s:<10.4f} {qnr:<10.4f}\n")
            
            f.write("="*80 + "\n")
        
        print(f"Final report saved to: {report_file}")
        return all_test_results
    
    def print_model_info(self):
        """Print comprehensive model information"""
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)
        
        # Basic model information
        print(f"Algorithm: {self.cfg['algorithm']}")
        print(f"Model Configuration:")
        model_config = self.cfg.get('model', {})
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        # Model statistics
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {self.model_stats['total_parameters']:,}")
        print(f"  Model Size: {self.model_stats['model_size_mb']:.2f} MB")
        
        # Handle FLOPs display with proper formatting
        flops = self.model_stats['total_flops']
        if flops > 1e9:
            print(f"  FLOPs: {flops/1e9:.2f} G")
        elif flops > 1e6:
            print(f"  FLOPs: {flops/1e6:.2f} M")
        elif flops > 1e3:
            print(f"  FLOPs: {flops/1e3:.2f} K")
        else:
            print(f"  FLOPs: {flops:.0f}")
        
        # Training configuration
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.nEpochs}")
        print(f"  Learning Rate: {self.cfg['schedule']['lr']}")
        print(f"  Optimizer: Adam")
        print(f"  Weight Decay: {self.cfg['schedule']['weight_decay']}")
        
        # Loss configuration
        print(f"\nLoss Configuration:")
        for loss_name, weight in self.loss_weights.items():
            print(f"  {loss_name}: weight={weight}")
        
        # Data configuration
        print(f"\nData Configuration:")
        print(f"  Training Datasets: {self.cfg['data_usage']['datasets']}")
        print(f"  Batch Size: {self.cfg['data']['batch_size']}")
        print(f"  Patch Size: {self.cfg['data']['patch_size']}")
        print(f"  Upscale Factor: {self.cfg['data'].get('upscale', self.cfg['data'].get('upsacle', 4))}")
        
        # GPU information
        if torch.cuda.is_available() and self.cfg.get('gpu_mode', True):
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\nGPU Information:")
            print(f"  Device: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
        
        print("="*80)
        print()

    def run(self):
        """Main training loop with enhanced error handling"""
        print(f"Starting training for {self.nEpochs} epochs...")
        self.check_gpu()
        
        # Print comprehensive model information
        self.print_model_info()
        
        # Save experiment start indicator
        save_config_path = os.path.join(self.results_dir, 'experiment_started.txt')
        with open(save_config_path, 'w') as f:
            f.write('Universal experiment started.\n')
            f.write(f'Algorithm: {self.cfg["algorithm"]}\n')
            f.write(f'Training on: {self.cfg["data_usage"]["datasets"]}\n')
            f.write(f'Evaluating on: WV2, WV3, GF2\n')
            f.write(f'Loss functions: {list(self.loss_functions.keys())}\n')
        
        eval_interval = self.cfg.get('schedule', {}).get('eval_interval', 1)
        save_interval = self.cfg.get('schedule', {}).get('save_interval', 10)
        
        try:
            while self.epoch <= self.nEpochs:
                # Training
                epoch_loss = self.train()
                
                # Validation (every eval_interval epochs)
                if self.epoch % eval_interval == 0:
                    val_results = self.eval()  # eval() now handles records update internally
                    
                    # Calculate average PSNR for display
                    avg_psnr = 0
                    if len(self.records.get('Avg_PSNR', [])) > 0:
                        avg_psnr = self.records['Avg_PSNR'][-1]
                    
                    # Print progress
                    print(f"Epoch {self.epoch}/{self.nEpochs} - Loss: {epoch_loss:.6f}, Avg PSNR: {avg_psnr:.4f}")
                else:
                    # Update basic records even when not evaluating
                    self.records['Epoch'].append(self.epoch)
                    self.records['Loss'].append(epoch_loss)
                    self.records['LR'].append(self.optimizer.param_groups[0]['lr'])
                    
                    # Log to tensorboard
                    self.writer.add_scalar('Loss/Train', epoch_loss, self.epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.epoch)
                    
                    print(f"Epoch {self.epoch}/{self.nEpochs} - Loss: {epoch_loss:.6f}")
                
                # Save checkpoint (every save_interval epochs)
                if self.epoch % save_interval == 0:
                    self.save_checkpoint(epoch=self.epoch)
                
                self.epoch += 1
                
        except KeyboardInterrupt:
            interrupt_log = os.path.join(self.results_dir, 'experiment_interrupted.txt')
            with open(interrupt_log, 'w') as f:
                f.write(f'Universal experiment interrupted at epoch {self.epoch}.\n')
            self.save_checkpoint(epoch=self.epoch)
            print(f"\nExperiment interrupted at epoch {self.epoch}")
        
        except Exception as e:
            error_log = os.path.join(self.results_dir, 'experiment_error.txt')
            with open(error_log, 'w') as f:
                f.write(f'Universal experiment failed at epoch {self.epoch}.\n')
                f.write(f'Error: {str(e)}\n')
            self.save_checkpoint(epoch=self.epoch)
            print(f"\nExperiment failed at epoch {self.epoch}: {e}")
            raise
        
        # Final evaluation and save - Match ResearchSolver logic
        if self.epoch > self.nEpochs:
            print("Training completed successfully. Performing final evaluation...")
            self.eval()
        self.save_checkpoint(epoch=self.epoch)
        
        # Generate comprehensive final report with testing
        print("Generating final report and testing models...")
        final_results = self.generate_final_report()
        
        # Log completion
        completion_log = os.path.join(self.results_dir, 'experiment_completed.txt')
        with open(completion_log, 'w') as f:
            f.write('Universal experiment completed successfully.\n')
            f.write(f'Total epochs: {self.epoch - 1}\n')
            f.write(f'Total time: {(time.time() - self.start_time) / 3600:.2f} hours\n')
        
        # Cleanup
        self.writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\nUniversal experiment completed!")
        print(f"Results saved in: {self.experiment_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"Experiment report: {os.path.join(self.results_dir, 'experiment_report.txt')}")
        
        return {
            'model_stats': self.model_stats,
            'latest_metrics': final_results,
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'experiment_dir': self.experiment_dir,
            'success': True
        }
    
    def test_with_model(self, model_path, model_name):
        """Test with specific model and save comparison images - Match ResearchSolver logic"""
        print(f"\n=== Testing with {model_name} model ===")
        
        # Load the model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
            self.model.load_state_dict(checkpoint['net'])
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
            return {}
        
        self.model.eval()
        
        # Clear previous results
        self.metrics_calculator = ComprehensiveMetrics(['WV2', 'WV3', 'GF2'])
        
        # Test on validation datasets (with ground truth)
        for dataset_name, data_loader in self.val_loaders.items():
            print(f"Testing {dataset_name} (with GT, {len(data_loader)} samples)...")
            
            save_dir = os.path.join(self.test_results_dir, f"{model_name}/{dataset_name}")
            os.makedirs(save_dir, exist_ok=True)
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, desc=f'{dataset_name} Test')):
                    # Handle batch format like ResearchSolver
                    ms_image, lms_image, pan_image, bms_image, name = (
                        Variable(batch[0]), Variable(batch[1]), 
                        Variable(batch[2]), Variable(batch[3]), batch[4]
                    )
                    
                    if self.cfg.get('gpu_mode', True) and torch.cuda.is_available():
                        ms_image = ms_image.cuda()
                        lms_image = lms_image.cuda()
                        pan_image = pan_image.cuda()
                        bms_image = bms_image.cuda()
                    
                    prediction = self.model(lms_image, bms_image, pan_image)
                    
                    if self.cfg['data']['normalize']:
                        ms_image = (ms_image + 1) / 2
                        lms_image = (lms_image + 1) / 2
                        pan_image = (pan_image + 1) / 2
                        bms_image = (bms_image + 1) / 2
                        prediction = (prediction + 1) / 2
                    
                    # Calculate metrics
                    self.metrics_calculator.add_reference_result(
                        dataset_name, prediction, ms_image
                    )
                    
                    # Save images - check config for save policy
                    eval_config = self.cfg.get('evaluation', {})
                    save_all = eval_config.get('save_all_test_images', True)
                    
                    if save_all or i < 10:  # Save all if enabled, otherwise only first 10
                        filename = name[0] if isinstance(name, (list, tuple)) else str(name)
                        filename = filename.replace('.tif', '')
                        
                        self.save_test_img(ms_image.cpu().data, 
                                         os.path.join(save_dir, f'{filename}_gt.tif'))
                        self.save_test_img(bms_image.cpu().data, 
                                         os.path.join(save_dir, f'{filename}_bic.tif'))
                        self.save_test_img(prediction.cpu().data, 
                                         os.path.join(save_dir, f'{filename}_pred.tif'))
        
        # Test on full resolution datasets (no ground truth) 
        for dataset_name, data_loader in self.full_test_loaders.items():
            if len(data_loader) == 0:
                continue
                
            print(f"Testing {dataset_name} (full resolution, {len(data_loader)} samples)...")
            
            save_dir = os.path.join(self.test_results_dir, f"{model_name}/{dataset_name}_full")
            os.makedirs(save_dir, exist_ok=True)
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, desc=f'{dataset_name} Full Test')):
                    try:
                        # Handle batch format like ResearchSolver
                        ms_image, lms_image, pan_image, bms_image, name = (
                            Variable(batch[0]), Variable(batch[1]), 
                            Variable(batch[2]), Variable(batch[3]), batch[4]
                        )
                        
                        if self.cfg.get('gpu_mode', True) and torch.cuda.is_available():
                            lms_image = lms_image.cuda()
                            pan_image = pan_image.cuda()
                            bms_image = bms_image.cuda()
                        
                        prediction = self.model(lms_image, bms_image, pan_image)
                        
                        if self.cfg['data']['normalize']:
                            lms_image = (lms_image + 1) / 2
                            pan_image = (pan_image + 1) / 2
                            bms_image = (bms_image + 1) / 2
                            prediction = (prediction + 1) / 2
                        
                        # Calculate no-reference metrics
                        self.metrics_calculator.add_no_reference_result(
                            dataset_name, prediction, pan_image, lms_image
                        )
                        
                        # Save images (save all full resolution results)
                        filename = name[0] if isinstance(name, (list, tuple)) else str(name)
                        filename = filename.replace('.tif', '')
                        
                        self.save_test_img(bms_image.cpu().data, 
                                         os.path.join(save_dir, f'{filename}_bic.tif'))
                        self.save_test_img(prediction.cpu().data, 
                                         os.path.join(save_dir, f'{filename}_pred.tif'))
                        
                    except Exception as e:
                        print(f"Warning: Error processing full resolution sample: {e}")
                        continue
        
        # Print results like ResearchSolver
        print(f"\n{model_name} Test Results:")
        self.metrics_calculator.print_results()
        
        # Get final results from metrics calculator
        stats = self.metrics_calculator.get_statistics()
        return stats
    
    def save_test_img(self, img, img_path):
        """Save test image in the same format as ResearchSolver"""
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        # Save image
        save_img = np.uint8(save_img * 255).astype('uint8')
        save_img = Image.fromarray(save_img, mode='CMYK')
        save_img.save(img_path)


if __name__ == "__main__":
    # Example usage
    with open('../configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    solver = UniSolver(config)
    solver.run()
