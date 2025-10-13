#!/usr/bin/env python3
"""
Cognac Best Runner - Runs both Cognac variants and reports the best result.

This script automates the process of:
1. Running hyperparameter tuning for both cognac and cognac-descent
2. Running main experiments for both variants
3. Comparing results and reporting the best performing variant

Usage:
    python run_cognac_best.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label

For full argument list, run: python run_cognac_best.py --help
"""

import subprocess
import sys
import json
import os
from pathlib import Path
import time
from framework.training_args import parse_args

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def parse_results_from_logs(dataset, attack_type, df_size, random_seed, variant, args=None):
    """Parse results from log files to compare performance."""
    log_dir = Path("logs/default") / dataset
    
    # Load class information for the dataset
    try:
        with open("classes_to_poison.json", "r") as f:
            class_dataset_dict = json.load(f)
        
        class1 = class_dataset_dict[dataset]["class1"]
        class2 = class_dataset_dict[dataset]["class2"]
        
        # Build the log file pattern matching the Logger's naming convention
        pattern = f"run_logs_{attack_type}_{df_size}_{class1}_{class2}"
        
        # Add corrective fraction suffix if applicable
        if args and hasattr(args, 'corrective_frac') and args.corrective_frac < 1.0:
            pattern += f"_cf_{args.corrective_frac}"
        
        pattern += ".json"
        
        # Look for log files matching the pattern
        log_files = list(log_dir.glob(pattern))
        
        if not log_files:
            print(f"⚠️  No log files found for {variant} with pattern {pattern}")
            return None
        
        # Take the most recent log file
        log_file = max(log_files, key=os.path.getmtime)
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # The logger structure is: results -> seed -> method -> metrics
            # We need to find the results for our variant
            results_data = data.get('results', {})
            
            # Look for results across all seeds for this variant
            variant_results = None
            for seed, methods in results_data.items():
                if variant in methods:
                    variant_results = methods[variant]
                    break
            
            if variant_results:
                # Extract key metrics based on the actual log structure
                # The log file uses "forget" and "utility" as the main metrics
                forget_acc = variant_results.get('forget', 0.0)
                utility_acc = variant_results.get('utility', 0.0)
                forget_f1 = variant_results.get('forget_f1', 0.0)
                utility_f1 = variant_results.get('utility_f1', 0.0)
                time_taken = variant_results.get('time_taken', 0.0)
                
                return {
                    'forget_acc': forget_acc,
                    'utility_acc': utility_acc,
                    'forget_f1': forget_f1,
                    'utility_f1': utility_f1,
                    'time_taken': time_taken,
                    'variant': variant,
                    'log_file': str(log_file)
                }
            else:
                print(f"⚠️  No results found for variant {variant} in log file {log_file}")
                
        except Exception as e:
            print(f"⚠️  Error parsing log file {log_file}: {e}")
    
    except Exception as e:
        print(f"⚠️  Error loading classes_to_poison.json: {e}")
    
    return None

def compare_results(cognac_results, cognac_descent_results):
    """Compare results and determine the best variant."""
    if not cognac_results and not cognac_descent_results:
        print("❌ No results available for comparison")
        return None
    
    if not cognac_results:
        return cognac_descent_results
    
    if not cognac_descent_results:
        return cognac_results
    
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    print(f"Cognac (full method):")
    print(f"  Forget Accuracy: {cognac_results['forget_acc']:.4f}")
    print(f"  Utility Accuracy: {cognac_results['utility_acc']:.4f}")
    print(f"  Forget F1: {cognac_results['forget_f1']:.4f}")
    print(f"  Utility F1: {cognac_results['utility_f1']:.4f}")
    print(f"  Time Taken: {cognac_results['time_taken']:.2f}s")
    
    print(f"\nCognac-Descent (descent only):")
    print(f"  Forget Accuracy: {cognac_descent_results['forget_acc']:.4f}")
    print(f"  Utility Accuracy: {cognac_descent_results['utility_acc']:.4f}")
    print(f"  Forget F1: {cognac_descent_results['forget_f1']:.4f}")
    print(f"  Utility F1: {cognac_descent_results['utility_f1']:.4f}")
    print(f"  Time Taken: {cognac_descent_results['time_taken']:.2f}s")
    
    # Select based on best forget score (higher is better for unlearning)
    print(f"\n📊 Forget Scores (higher is better for unlearning):")
    print(f"  Cognac: {cognac_results['forget_acc']:.4f}")
    print(f"  Cognac-Descent: {cognac_descent_results['forget_acc']:.4f}")
    
    if cognac_results['forget_acc'] >= cognac_descent_results['forget_acc']:
        best = cognac_results
        print(f"\n🏆 Best variant: Cognac (full method)")
        print(f"   Best Forget Score: {best['forget_acc']:.4f}")
    else:
        best = cognac_descent_results
        print(f"\n🏆 Best variant: Cognac-Descent (descent only)")
        print(f"   Best Forget Score: {best['forget_acc']:.4f}")
    
    return best

def main():
    # Parse arguments using the existing framework
    args = parse_args()
    
    # Add control options specific to this script
    import argparse
    parser = argparse.ArgumentParser(parents=[argparse.ArgumentParser()], add_help=False)
    parser.add_argument("--skip-hp-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--only-compare", action="store_true", help="Only compare existing results")
    control_args, _ = parser.parse_known_args()
    
    # Convert args to command line format for subprocess calls
    def args_to_cmd_list(args, unlearning_model=None):
        cmd_args = []
        for key, value in vars(args).items():
            if key == 'unlearning_model' and unlearning_model:
                cmd_args.extend([f"--{key}", unlearning_model])
            elif key not in ['experiment_name'] and value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                else:
                    cmd_args.extend([f"--{key}", str(value)])
        return cmd_args
    
    variants = ["cognac", "cognac-descent"]
    results = {}
    
    if not control_args.only_compare:
        print(f"🚀 Starting Cognac Best Runner")
        print(f"Dataset: {args.dataset}, GNN: {args.gnn}, Attack: {args.attack_type}")
        print(f"Forgetting fraction: {args.df_size}, Seed: {args.random_seed}")
        
        for variant in variants:
            print(f"\n{'#'*60}")
            print(f"PROCESSING VARIANT: {variant.upper()}")
            print(f"{'#'*60}")
            
            # Step 1: Hyperparameter tuning
            if not control_args.skip_hp_tune:
                hp_cmd = ["python", "hp_tune.py"] + args_to_cmd_list(args, variant)
                success = run_command(hp_cmd, f"Hyperparameter tuning for {variant}")
                if not success:
                    print(f"❌ Hyperparameter tuning failed for {variant}, skipping...")
                    continue
            
            # Step 2: Main experiment
            main_cmd = ["python", "main.py"] + args_to_cmd_list(args, variant)
            success = run_command(main_cmd, f"Main experiment for {variant}")
            if not success:
                print(f"❌ Main experiment failed for {variant}, skipping...")
                continue
            
            # Step 3: Parse results
            time.sleep(1)  # Give time for logs to be written
            result = parse_results_from_logs(args.dataset, args.attack_type, args.df_size, args.random_seed, variant, args)
            if result:
                results[variant] = result
                print(f"✅ Results captured for {variant}")
            else:
                print(f"⚠️  Could not parse results for {variant}")
    
    # Compare results
    cognac_results = results.get("cognac")
    cognac_descent_results = results.get("cognac-descent")
    
    best_result = compare_results(cognac_results, cognac_descent_results)
    
    if best_result:
        print(f"\n{'='*60}")
        print("FINAL RECOMMENDATION")
        print(f"{'='*60}")
        print(f"✨ For your configuration, use: --unlearning_model {best_result['variant']}")
        print(f"📊 Forget Accuracy: {best_result['forget_acc']:.4f} (higher is better)")
        print(f"📊 Utility Accuracy: {best_result['utility_acc']:.4f} (higher is better)")
        print(f"⏱️  Time Taken: {best_result['time_taken']:.2f}s")
        print(f"📁 Full results available in: {best_result['log_file']}")
    else:
        print(f"\n❌ Could not determine the best variant. Please check the logs manually.")
        print(f"📁 Log directory: logs/default/{args.dataset}/")
        sys.exit(1)

if __name__ == "__main__":
    main()
