import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_evaluation_results(evaluation_dir, datasets, eval_models, target_model):
    results = {dataset: {model: {} for model in eval_models} for dataset in datasets}
    for dataset in datasets:
        for model in eval_models:
            original_path = os.path.join(evaluation_dir, dataset, f'{dataset}_{model}_original.json')
            enhanced_path = os.path.join(evaluation_dir, dataset, target_model, model, f'{dataset}_{model}_enhanced.json')
            
            if os.path.exists(original_path):
                with open(original_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    correct = 0
                    for item in data:
                        if item['answer'] in item['extracted_response'] or item['extracted_response'] in item['answer']:
                            correct += 1
                    results[dataset][model]['original'] = correct / len(data) if data else 0
            else:
                results[dataset][model]['original'] = 0
            
            if os.path.exists(enhanced_path):
                with open(enhanced_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    correct = sum(1 for item in data if item['result'])
                    results[dataset][model]['enhanced'] = correct / len(data) if data else 0
            else:
                results[dataset][model]['enhanced'] = 0
    return results

def plot_results(results, datasets, eval_models, output_dir, target_model):
    os.makedirs(output_dir, exist_ok=True)
    
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(16, 6), sharey=True)
    
    colors = ['#A1D99B', '#FDBB84', '#B2ABD2', '#FFFFB2', '#B3CDE3', '#CCEBC5', '#DECBE4']
    hatch_styles = ['', '//']

    for idx, dataset in enumerate(datasets):
        ax = axes[idx] if num_datasets > 1 else axes
        models = eval_models
        original = [results[dataset][model]['original'] * 100 for model in models]
        enhanced = [results[dataset][model]['enhanced'] * 100 for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, (orig, enh) in enumerate(zip(original, enhanced)):
            ax.bar(x[i] - width/2, orig, width, label='Original' if i == 0 else "", 
                   color=colors[i], edgecolor='black', alpha=0.7)
            ax.bar(x[i] + width/2, enh, width, label='Enhanced' if i == 0 else "", 
                   color=colors[i], edgecolor='black', hatch=hatch_styles[1], alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, fontsize=9)
        ax.set_title(dataset, fontsize=14, fontweight='bold')  
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.text(0, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels() if num_datasets > 1 else axes.get_legend_handles_labels()
    fig.legend(handles[:2], ['Original', 'Enhanced'], loc='upper center', 
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.95))
    
    plt.savefig(os.path.join(output_dir, f'evaluation_results_{target_model}.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()

def csv_results(results, datasets, eval_models, output_dir, target_model):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'evaluation_results_{target_model}.csv'), 'w', encoding='utf-8') as f:
        f.write('Dataset,Model,Original Accuracy,Enhanced Accuracy\n')
        for dataset in datasets:
            for model in eval_models:
                original = results[dataset][model]['original']
                enhanced = results[dataset][model]['enhanced']
                f.write(f'{dataset},{model},{original:.4f},{enhanced:.4f}\n')

def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help='Dataset names to process (e.g. CommonsenseQA MMLU)')
    parser.add_argument('--target_model', type=str, required=True,
                       help='Target model used for enhancement (e.g. gpt-4o-mini)')
    parser.add_argument('--eval_models', nargs='+', required=True,
                       help='Models to evaluate (e.g. gpt-4o gemma-2-27B)')
    parser.add_argument('--evaluation_dir', type=str, default='evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    results = load_evaluation_results(
        args.evaluation_dir,
        args.datasets,
        args.eval_models,
        args.target_model
    )
    
    plot_results(
        results,
        args.datasets,
        args.eval_models,
        args.output_dir,
        args.target_model
    )
    
    csv_results(
        results,
        args.datasets,
        args.eval_models,
        args.output_dir,
        args.target_model
    )

if __name__ == '__main__':
    main()