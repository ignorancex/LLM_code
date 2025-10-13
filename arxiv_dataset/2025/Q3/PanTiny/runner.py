#!/usr/bin/env python
# coding=utf-8
"""
Universal Experiment Runner
@Description: Generic runner for pan-sharpening experiments supporting flexible configurations
"""
import os
import sys
import yaml
import time
import copy
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from solver.unisolver import UniSolver


def convert_config_types(config):
    """Convert string values to appropriate numeric types in config"""
    
    def convert_value(value):
        """Convert a single value if it's a numeric string"""
        if isinstance(value, str):
            # Skip conversion for algorithm names and other string configs
            if value in ['pnn', 'pnnv2', 'pannet', 'cosine', 'step', 'large', 'small', 'original', 'huge', 'ADAM', 'SGD', 'RMSprop']:
                return value
                
            # Try to convert boolean strings first
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            
            # Try to convert to float (handles both regular and scientific notation)
            try:
                # Check if it's a numeric string (including scientific notation)
                if any(char in value.lower() for char in '0123456789.-+e'):
                    float_val = float(value)
                    # Convert to int if it's a whole number without decimal point in original string
                    if '.' not in value and 'e' not in value.lower() and float_val.is_integer():
                        return int(float_val)
                    return float_val
            except (ValueError, TypeError):
                pass
            
            # Return original string if no conversion possible
            return value
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(item) for item in value]
        else:
            return value
    
    return convert_value(config)


def merge_configs(base_config, experiment_config):
    """Merge experiment specific config with base config with enhanced logging"""
    merged = copy.deepcopy(base_config)
    
    # Track what gets overridden for better debugging
    overridden_keys = []
    
    def deep_merge(dict1, dict2, path=""):
        for key, value in dict2.items():
            current_path = f"{path}.{key}" if path else key
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                deep_merge(dict1[key], value, current_path)
            else:
                if key in dict1 and dict1[key] != value:
                    overridden_keys.append(f"{current_path}: {dict1[key]} -> {value}")
                dict1[key] = value
    
    deep_merge(merged, experiment_config)
    
    # Log overridden configurations for debugging
    if overridden_keys:
        print(f"Configuration overrides applied:")
        for override in overridden_keys[:5]:  # Show first 5 to avoid spam
            print(f"  {override}")
        if len(overridden_keys) > 5:
            print(f"  ... and {len(overridden_keys) - 5} more overrides")
    
    return merged


def run_single_experiment(base_config, exp_config, exp_dir, bug_report_file):
    """Run a single experiment and return results"""
    
    exp_name = exp_config['name']
    exp_description = exp_config.get('description', 'No description')
    algorithm = exp_config.get('algorithm', base_config.get('algorithm', 'pnn'))
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Description: {exp_description}")
    print(f"Algorithm: {algorithm}")
    print(f"{'='*80}")
    
    try:
        # Merge configurations
        full_config = merge_configs(base_config, exp_config)
        full_config['name'] = f"{exp_name}"
        full_config['algorithm'] = algorithm
        
        # Convert types
        full_config = convert_config_types(full_config)
        
        # Create experiment subdirectory
        exp_subdir = os.path.join(exp_dir, exp_name)
        os.makedirs(exp_subdir, exist_ok=True)
        
        # Override output directory to point to our experiment structure
        full_config['output_dir'] = exp_subdir
        
        print(f"Configuration:")
        print(f"  Algorithm: {algorithm}")
        if 'model' in exp_config:
            model_info = exp_config['model']
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    print(f"  Model {key}: {value}")
            else:
                print(f"  Model: {model_info}")
        
        if 'loss' in exp_config:
            loss_info = exp_config['loss']
            print(f"  Loss configuration:")
            for loss_name, loss_config in loss_info.items():
                if isinstance(loss_config, dict) and loss_config.get('enabled', False):
                    weight = loss_config.get('weight', 1.0)
                    print(f"    {loss_name}: enabled (weight={weight})")
        
        print(f"  Epochs: {full_config['nEpochs']}")
        print(f"  Learning rate: {full_config['schedule']['lr']}")
        print(f"  Output directory: {exp_subdir}")
        
        # Record start time
        start_time = time.time()
        
        # Initialize and run solver
        solver = UniSolver(full_config)
        result = solver.run()
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Collect results from solver
        try:
            results = {
                'name': exp_name,
                'description': exp_description,
                'algorithm': algorithm,
                'training_time_hours': training_time / 3600,
                'success': True,
                'error': None,
                'exp_dir': result['experiment_dir'],
                'model_stats': result['model_stats'],
                'latest_metrics': result['latest_metrics'],
                'loss_config': exp_config.get('loss', {}),
                'model_config': exp_config.get('model', {})
            }
        except Exception as e:
            print(f"Warning: Could not extract all results from solver: {e}")
            results = {
                'name': exp_name,
                'description': exp_description,
                'algorithm': algorithm,
                'training_time_hours': training_time / 3600,
                'success': True,
                'error': None,
                'exp_dir': exp_subdir,
                'model_stats': getattr(solver, 'model_stats', {}),
                'latest_metrics': {},
                'loss_config': exp_config.get('loss', {}),
                'model_config': exp_config.get('model', {})
            }
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED: {exp_name}")
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"Results saved in: {results['exp_dir']}")
        print(f"{'='*80}")
        
        return results
        
    except KeyboardInterrupt:
        error_msg = f"Experiment {exp_name} interrupted by user (KeyboardInterrupt)"
        print(f"\nERROR: {error_msg}")
        
        with open(bug_report_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] {error_msg}\n")
        
        raise  # Re-raise to stop all experiments
        
    except Exception as e:
        error_msg = f"Experiment {exp_name} failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        print("Traceback:")
        traceback.print_exc()
        
        # Log error to bug report
        with open(bug_report_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] {error_msg}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        # Return failed result
        return {
            'name': exp_name,
            'description': exp_description,
            'algorithm': algorithm,
            'training_time_hours': -1,
            'success': False,
            'error': str(e),
            'exp_dir': None,
            'model_stats': {},
            'latest_metrics': {},
            'loss_config': exp_config.get('loss', {}),
            'model_config': exp_config.get('model', {})
        }


def generate_comprehensive_report(results, total_time):
    """Generate comprehensive experiment report with all required metrics"""
    
    print("\n" + "="*150)
    print("COMPREHENSIVE EXPERIMENT REPORT")
    print("="*150)
    
    # Separate successful and failed experiments
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    # Summary
    print(f"\nEXPERIMENT SUMMARY:")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    
    if successful_results:
        # Main comparison table
        print(f"\nMAIN COMPARISON TABLE:")
        print("="*180)
        
        # Table header
        header = f"{'Model':<25} {'Algorithm':<12} {'Params(K)':<12} {'FLOPs(M)':<12} {'Size(MB)':<10} {'Time(h)':<8} "
        header += f"{'WV2_PSNR':<10} {'WV2_SSIM':<10} {'WV2_SAM':<10} "
        header += f"{'WV3_PSNR':<10} {'WV3_SSIM':<10} {'WV3_SAM':<10} "
        header += f"{'GF2_PSNR':<10} {'GF2_SSIM':<10} {'GF2_SAM':<10}"
        print(header)
        print("-" * 180)
        
        table_data = []
        
        for result in successful_results:
            if not result['success']:
                continue
            
            # Model name with loss info
            loss_names = []
            if result['loss_config']:
                for loss_name, loss_cfg in result['loss_config'].items():
                    if isinstance(loss_cfg, dict) and loss_cfg.get('enabled', False):
                        weight = loss_cfg.get('weight', 1.0)
                        loss_names.append(f"{loss_name}({weight})")
            
            model_name = result['name']
            if loss_names:
                model_name += f"_{'+'.join(loss_names)}"
            
            # Extract metrics
            model_stats = result['model_stats']
            params_k = model_stats.get('total_parameters', 0) / 1000
            flops_m = model_stats.get('total_flops', 0) / 1000000
            size_mb = model_stats.get('model_size_mb', 0)
            time_h = result['training_time_hours']
            
            # Extract latest metrics for each dataset
            metrics = result['latest_metrics']
            
            # Handle UniSolver format: latest_metrics = {'Latest': {dataset: {metric: {mean: value}}}}
            if isinstance(metrics, dict) and 'Latest' in metrics:
                # Extract from Latest model results
                latest_stats = metrics['Latest']
                wv2_psnr = latest_stats.get('WV2', {}).get('PSNR', {}).get('mean', 0.0)
                wv2_ssim = latest_stats.get('WV2', {}).get('SSIM', {}).get('mean', 0.0)
                wv2_sam = latest_stats.get('WV2', {}).get('SAM', {}).get('mean', 0.0)
                
                wv3_psnr = latest_stats.get('WV3', {}).get('PSNR', {}).get('mean', 0.0)
                wv3_ssim = latest_stats.get('WV3', {}).get('SSIM', {}).get('mean', 0.0)
                wv3_sam = latest_stats.get('WV3', {}).get('SAM', {}).get('mean', 0.0)
                
                gf2_psnr = latest_stats.get('GF2', {}).get('PSNR', {}).get('mean', 0.0)
                gf2_ssim = latest_stats.get('GF2', {}).get('SSIM', {}).get('mean', 0.0)
                gf2_sam = latest_stats.get('GF2', {}).get('SAM', {}).get('mean', 0.0)
            else:
                # Handle direct format: metrics = {'WV2': {'PSNR': value}}
                wv2_psnr = metrics.get('WV2', {}).get('PSNR', 0.0)
                wv2_ssim = metrics.get('WV2', {}).get('SSIM', 0.0)
                wv2_sam = metrics.get('WV2', {}).get('SAM', 0.0)
                
                wv3_psnr = metrics.get('WV3', {}).get('PSNR', 0.0)
                wv3_ssim = metrics.get('WV3', {}).get('SSIM', 0.0)
                wv3_sam = metrics.get('WV3', {}).get('SAM', 0.0)
                
                gf2_psnr = metrics.get('GF2', {}).get('PSNR', 0.0)
                gf2_ssim = metrics.get('GF2', {}).get('SSIM', 0.0)
                gf2_sam = metrics.get('GF2', {}).get('SAM', 0.0)
            
            # Format row
            row = f"{model_name:<25} {result['algorithm']:<12} {params_k:<12.1f} {flops_m:<12.2f} {size_mb:<10.2f} {time_h:<8.2f} "
            row += f"{wv2_psnr:<10.4f} {wv2_ssim:<10.4f} {wv2_sam:<10.4f} "
            row += f"{wv3_psnr:<10.4f} {wv3_ssim:<10.4f} {wv3_sam:<10.4f} "
            row += f"{gf2_psnr:<10.4f} {gf2_ssim:<10.4f} {gf2_sam:<10.4f}"
            
            print(row)
            
            # Store for analysis
            table_data.append({
                'name': model_name,
                'algorithm': result['algorithm'],
                'params_k': params_k,
                'flops_m': flops_m,
                'size_mb': size_mb,
                'time_h': time_h,
                'wv2_psnr': wv2_psnr,
                'wv2_ssim': wv2_ssim,
                'wv2_sam': wv2_sam,
                'wv3_psnr': wv3_psnr,
                'wv3_ssim': wv3_ssim,
                'wv3_sam': wv3_sam,
                'gf2_psnr': gf2_psnr,
                'gf2_ssim': gf2_ssim,
                'gf2_sam': gf2_sam,
                'avg_psnr': (wv2_psnr + wv3_psnr + gf2_psnr) / 3
            })
        
        print("="*180)
        
        # Best performing configurations
        if table_data:
            best_psnr = max(table_data, key=lambda x: x['avg_psnr'])
            best_efficiency = min([x for x in table_data if x['params_k'] > 0], 
                                key=lambda x: x['params_k'], default=None)
            
            print(f"\nBEST PERFORMING CONFIGURATIONS:")
            print("-" * 50)
            print(f"Best PSNR: {best_psnr['name']} (Avg PSNR: {best_psnr['avg_psnr']:.4f})")
            if best_efficiency:
                print(f"Most efficient: {best_efficiency['name']} ({best_efficiency['params_k']:.1f}K params)")
    
    # Failed experiments
    if failed_results:
        print(f"\nFAILED EXPERIMENTS:")
        print("-" * 50)
        for result in failed_results:
            print(f"{result['name']}: {result['error']}")
    
    print("="*150)
    return successful_results


def save_comprehensive_report(exp_dir, results, total_time):
    """Save comprehensive experiment report to file"""
    
    report_file = os.path.join(exp_dir, 'comprehensive_report.txt')
    
    with open(report_file, 'w') as f:
        f.write(f"COMPREHENSIVE EXPERIMENT REPORT\n")
        f.write(f"{'='*80}\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"Total experiment time: {total_time/3600:.2f} hours\n\n")
        
        # Successful experiments summary
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        f.write(f"EXPERIMENT SUMMARY:\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful: {len(successful_results)}\n")
        f.write(f"Failed: {len(failed_results)}\n\n")
        
        if successful_results:
            # Main table
            f.write(f"MAIN COMPARISON TABLE:\n")
            f.write(f"{'='*180}\n")
            
            # Header
            header = f"{'Model':<25} {'Algorithm':<12} {'Params(K)':<12} {'FLOPs(M)':<12} {'Size(MB)':<10} {'Time(h)':<8} "
            header += f"{'WV2_PSNR':<10} {'WV2_SSIM':<10} {'WV2_SAM':<10} "
            header += f"{'WV3_PSNR':<10} {'WV3_SSIM':<10} {'WV3_SAM':<10} "
            header += f"{'GF2_PSNR':<10} {'GF2_SSIM':<10} {'GF2_SAM':<10}\n"
            f.write(header)
            f.write(f"{'-'*180}\n")
            
            # Data rows
            for result in successful_results:
                if not result['success']:
                    continue
                
                # Model name with loss info
                loss_names = []
                if result['loss_config']:
                    for loss_name, loss_cfg in result['loss_config'].items():
                        if isinstance(loss_cfg, dict) and loss_cfg.get('enabled', False):
                            weight = loss_cfg.get('weight', 1.0)
                            loss_names.append(f"{loss_name}({weight})")
                
                model_name = result['name']
                if loss_names:
                    model_name += f"_{'+'.join(loss_names)}"
                
                # Extract metrics
                model_stats = result['model_stats']
                params_k = model_stats.get('total_parameters', 0) / 1000
                flops_m = model_stats.get('total_flops', 0) / 1000000
                size_mb = model_stats.get('model_size_mb', 0)
                time_h = result['training_time_hours']
                
                # Extract latest metrics with proper format handling
                metrics = result['latest_metrics']
                
                # Handle UniSolver format: latest_metrics = {'Latest': {dataset: {metric: {mean: value}}}}
                if isinstance(metrics, dict) and 'Latest' in metrics:
                    # Extract from Latest model results
                    latest_stats = metrics['Latest']
                    wv2_psnr = latest_stats.get('WV2', {}).get('PSNR', {}).get('mean', 0.0)
                    wv2_ssim = latest_stats.get('WV2', {}).get('SSIM', {}).get('mean', 0.0)
                    wv2_sam = latest_stats.get('WV2', {}).get('SAM', {}).get('mean', 0.0)
                    
                    wv3_psnr = latest_stats.get('WV3', {}).get('PSNR', {}).get('mean', 0.0)
                    wv3_ssim = latest_stats.get('WV3', {}).get('SSIM', {}).get('mean', 0.0)
                    wv3_sam = latest_stats.get('WV3', {}).get('SAM', {}).get('mean', 0.0)
                    
                    gf2_psnr = latest_stats.get('GF2', {}).get('PSNR', {}).get('mean', 0.0)
                    gf2_ssim = latest_stats.get('GF2', {}).get('SSIM', {}).get('mean', 0.0)
                    gf2_sam = latest_stats.get('GF2', {}).get('SAM', {}).get('mean', 0.0)
                else:
                    # Handle direct format: metrics = {'WV2': {'PSNR': value}}
                    wv2_psnr = metrics.get('WV2', {}).get('PSNR', 0.0)
                    wv2_ssim = metrics.get('WV2', {}).get('SSIM', 0.0)
                    wv2_sam = metrics.get('WV2', {}).get('SAM', 0.0)
                    
                    wv3_psnr = metrics.get('WV3', {}).get('PSNR', 0.0)
                    wv3_ssim = metrics.get('WV3', {}).get('SSIM', 0.0)
                    wv3_sam = metrics.get('WV3', {}).get('SAM', 0.0)
                    
                    gf2_psnr = metrics.get('GF2', {}).get('PSNR', 0.0)
                    gf2_ssim = metrics.get('GF2', {}).get('SSIM', 0.0)
                    gf2_sam = metrics.get('GF2', {}).get('SAM', 0.0)
                
                # Format row
                row = f"{model_name:<25} {result['algorithm']:<12} {params_k:<12.1f} {flops_m:<12.2f} {size_mb:<10.2f} {time_h:<8.2f} "
                row += f"{wv2_psnr:<10.4f} {wv2_ssim:<10.4f} {wv2_sam:<10.4f} "
                row += f"{wv3_psnr:<10.4f} {wv3_ssim:<10.4f} {wv3_sam:<10.4f} "
                row += f"{gf2_psnr:<10.4f} {gf2_ssim:<10.4f} {gf2_sam:<10.4f}\n"
                
                f.write(row)
            
            f.write(f"{'='*180}\n\n")
        
        # Failed experiments
        if failed_results:
            f.write(f"FAILED EXPERIMENTS:\n")
            f.write(f"{'-'*50}\n")
            for result in failed_results:
                f.write(f"{result['name']}: {result['error']}\n")
            f.write(f"\n")
        
        # Individual experiment details
        f.write(f"INDIVIDUAL EXPERIMENT DETAILS:\n")
        f.write(f"{'-'*50}\n")
        for result in results:
            f.write(f"\nExperiment: {result['name']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Algorithm: {result['algorithm']}\n")
            f.write(f"Success: {result['success']}\n")
            if result['success']:
                f.write(f"Training time: {result['training_time_hours']:.2f} hours\n")
                f.write(f"Results directory: {result['exp_dir']}\n")
                if result['model_stats']:
                    f.write(f"Parameters: {result['model_stats'].get('total_parameters', 'N/A')}\n")
                    f.write(f"FLOPs: {result['model_stats'].get('total_flops', 'N/A')}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write(f"{'-'*30}\n")
    
    print(f"\nComprehensive report saved to: {report_file}")


def main():
    """Main function to run experiments"""
    
    # Get config file from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/experiment4.yml'
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print(f"UNIVERSAL EXPERIMENT RUNNER")
    print("="*80)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Description: {config['experiment_description']}")
    print(f"Total experiments: {len(config['experiments'])}")
    print("="*80)
    
    # Create experiment directory
    timestamp = int(time.time())
    exp_name = f"{config['experiment_name']}_{timestamp}"
    output_dir = config['base_config']['output_dir']
    exp_dir = os.path.join(output_dir, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize bug report
    bug_report_file = os.path.join(exp_dir, 'bug_report.txt')
    with open(bug_report_file, 'w') as f:
        f.write(f"Bug Report for {config['experiment_name']}\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write("="*50 + "\n\n")
    
    print(f"Experiment directory: {exp_dir}")
    
    experiment_start_time = time.time()
    results = []
    
    try:
        # Run all experiments
        for i, experiment in enumerate(config['experiments'], 1):
            print(f"\n[{i}/{len(config['experiments'])}] Starting experiment: {experiment['name']}")
            
            result = run_single_experiment(
                config['base_config'], 
                experiment, 
                exp_dir, 
                bug_report_file
            )
            results.append(result)
            
            print(f"Experiment {experiment['name']} completed: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        total_time = time.time() - experiment_start_time
        
        # Generate and save comprehensive report
        print(f"\nGenerating comprehensive report...")
        generate_comprehensive_report(results, total_time)
        save_comprehensive_report(exp_dir, results, total_time)
        
        print(f"\nAll experiments completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Results saved in: {exp_dir}")
        
    except KeyboardInterrupt:
        print(f"\nExperiments interrupted by user!")
        total_time = time.time() - experiment_start_time
        
        # Still generate report for completed experiments
        if results:
            print(f"Generating partial report for completed experiments...")
            generate_comprehensive_report(results, total_time)
            save_comprehensive_report(exp_dir, results, total_time)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        
        # Log to bug report
        with open(bug_report_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] Unexpected error: {e}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")


if __name__ == "__main__":
    main()
