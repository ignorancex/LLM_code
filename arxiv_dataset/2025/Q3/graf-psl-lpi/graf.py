"""
GRAF: Gradient-based Radar Ambiguity Functions

PyTorch implementation of GRAF from the paper:
"Differentiable Radar Ambiguity Functions: Mathematical Formulation and Computational Implementation"

GRAF enables gradient-based optimization of radar waveforms through differentiable ambiguity functions,
providing a general approach to radar waveform design problems. This implementation demonstrates
the framework through Low Probability of Intercept (LPI) radar optimization.

Key Features:
- Differentiable ambiguity function computation using PyTorch
- Two-objective optimization: PSL (radar performance) vs LPI (stealth capability)  
- Configurable framework with YAML-based parameter management
- Comparative analysis against genetic algorithm baselines

Author: Marc Bara Iniesta
Patent Status: Patent Pending
License: Academic and Research Use Only - See LICENSE
Repository: https://github.com/marcbarainiesta/graf-psl-lpi

NOTICE: This software is protected by provisional patent application. 
Commercial use requires licensing agreement. Contact authors for commercial licensing inquiries.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from typing import List, Tuple, Dict
from config_loader import Config, load_config

# Set up Seaborn styling for publication-quality plots
sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.5})
sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# ========== Core Implementation (unchanged) ==========
class DifferentiableAmbiguity(nn.Module):
    """Differentiable ambiguity function implementation"""
    
    def __init__(self, normalization_method: str = "max"):
        super().__init__()
        self.normalization_method = normalization_method
    
    def forward(self, s):
        N = s.shape[-1]
        
        # Create shift matrix efficiently
        idx = torch.arange(N, device=s.device)
        shift_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)) % N
        S_shifted = s[..., shift_idx]
        
        # Compute correlations for all delays
        R = s.unsqueeze(-2) * S_shifted.conj()
        
        # FFT over time dimension
        X = torch.fft.fft(R, dim=-1)
        
        # Magnitude squared
        chi = (X * X.conj()).real
        
        # Center the ambiguity function
        chi = torch.fft.fftshift(chi, dim=(-2, -1))
        
        # Normalize for stability
        if self.normalization_method == "max":
            chi = chi / chi.amax(dim=(-2, -1), keepdim=True)
        elif self.normalization_method == "sum":
            chi = chi / chi.sum(dim=(-2, -1), keepdim=True)
        
        return chi

# ========== PSL Computation for Two-Objective Optimization ==========
def compute_psl(chi, mainlobe_width: int = 3):
    """
    Compute Peak Sidelobe Level with improved robustness
    
    Args:
        chi: Ambiguity function (NxN tensor)
        mainlobe_width: Half-width of mainlobe exclusion region
    
    Returns:
        PSL: Peak sidelobe level (ratio)
    """
    N = chi.shape[-1]
    center = N // 2
    
    # Find actual peak location (may not be exactly at center after optimization)
    peak_value = chi.max()
    flat_idx = chi.argmax()
    # Manual unravel_index for compatibility with older PyTorch
    peak_location = (flat_idx // N, flat_idx % N)
    
    # Create mainlobe mask - exclude region around actual peak
    mask = torch.ones_like(chi)
    
    # Use diamond-shaped mask which is more appropriate for ambiguity functions
    # Create coordinate grids
    i_coords, j_coords = torch.meshgrid(
        torch.arange(N, device=chi.device), 
        torch.arange(N, device=chi.device), 
        indexing='ij'
    )
    
    # Manhattan distance from actual peak location
    distance_from_peak = (
        torch.abs(i_coords - peak_location[0]) + 
        torch.abs(j_coords - peak_location[1])
    )
    
    # Zero out mainlobe region
    mask[distance_from_peak <= mainlobe_width] = 0
    
    # Extract sidelobes
    sidelobes = chi * mask
    
    # Safety check to prevent division by zero
    if peak_value < 1e-10:
        return torch.tensor(1.0, device=chi.device)
    
    # PSL = max(sidelobes) / max(mainlobe)
    psl = sidelobes.max() / peak_value
    
    return psl

# ========== LPI Metrics ==========
def compute_spectral_uniformity(s, epsilon: float = 1e-10):
    """
    Example LPI metric: spectral peakiness  
    Lower values = more uniform spectrum = harder to detect
    Note: GRAF supports any differentiable radar performance metric
    """
    S = torch.fft.fft(s)
    power_spectrum = torch.abs(S)**2
    
    # Normalize to probability distribution
    power_spectrum = power_spectrum / power_spectrum.sum()
    
    # Use variance (lower = more uniform)
    spectral_variance = torch.var(power_spectrum)
    
    return spectral_variance

def combined_loss(s, chi, config: Config):
    """Example multi-objective loss: PSL + LPI (easily extensible to other metrics)"""
    psl = compute_psl(chi, config.signal.mainlobe_width)
    lpi = compute_spectral_uniformity(s, config.loss.numerical_epsilon)
    
    # Configurable weighting and scaling
    return psl + config.loss.lpi_weight * lpi * config.loss.lpi_scale_factor

# ========== GRAF Optimization ==========
def gradient_optimize(config: Config, lambda_lpi: float = None):
    """GRAF: Gradient-based radar ambiguity function optimization"""
    
    # Use override lambda or config default
    effective_lambda = lambda_lpi if lambda_lpi is not None else config.loss.lpi_weight
    
    # Get parameters from config
    N = config.signal.sequence_length
    lr = config.optimization.gradient.learning_rate
    max_iterations = config.optimization.gradient.max_iterations
    convergence_threshold = config.optimization.gradient.convergence_threshold
    log_interval = config.logging.log_interval
    
    # Initialize with random phases
    phases = torch.rand(N, device=config.device) * 2 * np.pi
    phases.requires_grad_(True)
    
    # Setup
    amb_func = DifferentiableAmbiguity(config.advanced.normalization_method)
    optimizer = torch.optim.Adam([phases], lr=lr)
    
    # Optional learning rate scheduler
    scheduler = None
    if config.advanced.scheduler.enabled:
        if config.advanced.scheduler.type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=config.advanced.scheduler.step_size,
                gamma=config.advanced.scheduler.gamma
            )
    
    history = {'loss': [], 'psl': [], 'lpi': [], 'time': []}
    start_time = time.time()
    prev_loss = float('inf')
    
    # Create temporary config with override lambda
    temp_config = Config(config.config_path)
    temp_config.update_from_dict({'loss': {'lpi_weight': effective_lambda}})
    
    for i in range(max_iterations):
        optimizer.zero_grad()
        
        # Generate waveform with optional phase constraint
        if config.advanced.phase_constraint:
            phases.data = phases.data % (2 * np.pi)
        
        s = torch.exp(1j * phases)
        
        # Compute metrics
        chi = amb_func(s)
        loss = combined_loss(s, chi, temp_config)
        
        # Record metrics
        with torch.no_grad():
            current_loss = loss.item()
            history['loss'].append(current_loss)
            history['psl'].append(compute_psl(chi, config.signal.mainlobe_width).item())
            history['lpi'].append(compute_spectral_uniformity(s).item())
            history['time'].append(time.time() - start_time)
        
        # Optimize
        loss.backward()
        
        # Optional gradient clipping
        if hasattr(config.advanced, 'gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_([phases], config.advanced.gradient_clip_norm)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Logging
        if i % log_interval == 0:
            logging.info(f"GRAF iter {i}: Loss={current_loss:.4f}, "
                         f"PSL={history['psl'][-1]:.4f}, LPI={history['lpi'][-1]:.6f}")
        
        # Early stopping (if enabled)
        if (hasattr(config.optimization.gradient, 'enable_early_stopping') and 
            config.optimization.gradient.enable_early_stopping and
            abs(prev_loss - current_loss) < convergence_threshold):
            logging.info(f"Converged at iteration {i}")
            break
        prev_loss = current_loss
    
    # Return final waveform and history
    final_s = torch.exp(1j * phases).detach()
    return final_s, history

# ========== Genetic Algorithm implementation ==========
def genetic_algorithm(config: Config, lambda_lpi: float = None):
    """Genetic Algorithm for waveform optimization using config parameters"""
    
    # Use override lambda or config default
    effective_lambda = lambda_lpi if lambda_lpi is not None else config.loss.lpi_weight
    
    # Get parameters from config
    N = config.signal.sequence_length
    pop_size = config.optimization.genetic_algorithm.population_size
    max_generations = config.optimization.genetic_algorithm.max_generations
    mutation_rate = config.optimization.genetic_algorithm.mutation_rate
    crossover_rate = config.optimization.genetic_algorithm.crossover_rate
    tournament_size = config.optimization.genetic_algorithm.tournament_size
    mutation_magnitude = config.optimization.genetic_algorithm.mutation_magnitude
    elitism_ratio = config.optimization.genetic_algorithm.elitism_ratio
    log_interval = config.logging.log_interval
    
    # Setup
    amb_func = DifferentiableAmbiguity(config.advanced.normalization_method)
    
    history = {'loss': [], 'psl': [], 'lpi': [], 'time': [], 'evaluations': []}
    start_time = time.time()
    evaluations = 0
    
    # Initialize population
    population = torch.rand(pop_size, N, device=config.device) * 2 * np.pi
    
    # Create temporary config with override lambda
    temp_config = Config(config.config_path)
    temp_config.update_from_dict({'loss': {'lpi_weight': effective_lambda}})
    
    # Calculate elitism count
    elite_count = int(pop_size * elitism_ratio)
    
    for gen in range(max_generations):
        # Evaluate fitness
        fitness = torch.zeros(pop_size, device=config.device)
        
        for i in range(pop_size):
            s = torch.exp(1j * population[i])
            chi = amb_func(s)
            with torch.no_grad():
                fitness[i] = -combined_loss(s, chi, temp_config)  # Negative for maximization
            evaluations += 1
        
        # Record best individual
        best_idx = torch.argmax(fitness)
        best_s = torch.exp(1j * population[best_idx])
        best_chi = amb_func(best_s)
        
        with torch.no_grad():
            history['loss'].append(-fitness[best_idx].item())
            history['psl'].append(compute_psl(best_chi, config.signal.mainlobe_width).item())
            history['lpi'].append(compute_spectral_uniformity(best_s).item())
            history['time'].append(time.time() - start_time)
            history['evaluations'].append(evaluations)
        
        # Logging
        if gen % log_interval == 0:
            logging.info(f"GA generation {gen}: Loss={history['loss'][-1]:.4f}, "
                        f"PSL={history['psl'][-1]:.4f}, LPI={history['lpi'][-1]:.6f}")
        
        # Selection with elitism
        new_population = torch.zeros_like(population)
        
        # Keep elite individuals
        if elite_count > 0:
            elite_indices = torch.topk(fitness, elite_count).indices
            new_population[:elite_count] = population[elite_indices]
        
        # Tournament selection for remaining individuals
        for i in range(elite_count, pop_size):
            tournament = torch.randint(0, pop_size, (tournament_size,))
            winner = tournament[torch.argmax(fitness[tournament])]
            new_population[i] = population[winner]
        
        # Crossover
        for i in range(elite_count, pop_size-1, 2):
            if torch.rand(1) < crossover_rate:
                # Single-point crossover
                point = torch.randint(1, N-1, (1,)).item()
                temp = new_population[i, point:].clone()
                new_population[i, point:] = new_population[i+1, point:]
                new_population[i+1, point:] = temp
        
        # Mutation (skip elites)
        mutation_mask = torch.rand(pop_size-elite_count, N, device=config.device) < mutation_rate
        mutations = (torch.rand(pop_size-elite_count, N, device=config.device) - 0.5) * mutation_magnitude
        new_population[elite_count:] = torch.where(
            mutation_mask, 
            (new_population[elite_count:] + mutations) % (2*np.pi), 
            new_population[elite_count:]
        )
        
        population = new_population
    
    # Return best individual
    best_idx = torch.argmax(fitness)
    final_s = torch.exp(1j * population[best_idx]).detach()
    return final_s, history

# ========== Experiment Management ==========
def run_pareto_experiment(config: Config):
    """Generate Pareto frontier by varying lambda"""
    lambda_values = config.get_lambda_values()
    
    gradient_results = []
    ga_results = []
    
    for lambda_lpi in lambda_values:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running with lambda_lpi = {lambda_lpi:.2f}")
        logging.info(f"{'='*50}")
        
        # GRAF method
        logging.info("GRAF optimization:")
        grad_s, grad_hist = gradient_optimize(config, lambda_lpi)
        gradient_results.append({
            'lambda': lambda_lpi,
            'waveform': grad_s,
            'psl': grad_hist['psl'][-1],
            'lpi': grad_hist['lpi'][-1],
            'time': grad_hist['time'][-1],
            'history': grad_hist
        })
        
        # GA method
        logging.info("Genetic Algorithm:")
        ga_s, ga_hist = genetic_algorithm(config, lambda_lpi)
        ga_results.append({
            'lambda': lambda_lpi,
            'waveform': ga_s,
            'psl': ga_hist['psl'][-1],
            'lpi': ga_hist['lpi'][-1],
            'time': ga_hist['time'][-1],
            'history': ga_hist
        })
    
    return gradient_results, ga_results

# ========== Plotting Functions ==========
def plot_main_results(gradient_results, ga_results, config: Config):
    """Main comparison figure"""
    fig_size = config.get_figure_size('main_comparison')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    # 1. Convergence comparison
    mid_idx = len(gradient_results) // 2
    grad_hist = gradient_results[mid_idx]['history']
    ga_hist = ga_results[mid_idx]['history']
    
    # Use Seaborn color palette for better styling
    colors = sns.color_palette("Greys", n_colors=6)
    ax1.plot(grad_hist['time'], grad_hist['loss'], color=colors[5], linestyle='-', 
             label='GRAF', linewidth=2.5)
    ax1.plot(ga_hist['time'], ga_hist['loss'], color=colors[2], linestyle='-', 
             label='GA', linewidth=2.5)
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Combined Loss', fontweight='bold')
    ax1.set_title(f'Convergence Comparison (λ={gradient_results[mid_idx]["lambda"]:.1f})', 
                  fontweight='bold', pad=20)
    ax1.legend(frameon=True, shadow=True)
    
    # 2. Pareto frontier
    grad_psl = [r['psl'] for r in gradient_results]
    grad_lpi = [r['lpi'] * config.output.lpi_display_scale for r in gradient_results]
    ga_psl = [r['psl'] for r in ga_results]
    ga_lpi = [r['lpi'] * config.output.lpi_display_scale for r in ga_results]
    
    ax2.plot(grad_psl, grad_lpi, color=colors[5], linestyle='-', marker='o', 
             label='GRAF', markersize=8, linewidth=2.5, markerfacecolor=colors[5])
    ax2.plot(ga_psl, ga_lpi, color=colors[2], linestyle='-', marker='s', 
             label='GA', markersize=8, linewidth=2.5, 
             markerfacecolor='white', markeredgecolor=colors[2], markeredgewidth=2)
    ax2.set_xlabel('Peak Sidelobe Level (PSL)', fontweight='bold')
    ax2.set_ylabel(f'Spectral Variance (×{config.output.lpi_display_scale})', fontweight='bold')
    ax2.set_title('Pareto Frontier: Detection vs. LPI', fontweight='bold', pad=20)
    ax2.legend(frameon=True, shadow=True)
    
    # Add lambda annotations
    for i, (psl, lpi, lam) in enumerate(zip(grad_psl, grad_lpi, 
                                            [r['lambda'] for r in gradient_results])):
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            ax2.annotate(f'λ={lam:.1f}', (psl, lpi), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Apply Seaborn styling and save
    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()
    save_path = config.save_path(f'fig1_main_comparison.{config.output.plot_format}')
    plt.savefig(save_path, dpi=config.output.plot_dpi, bbox_inches='tight', facecolor='white')
    if not config.output.save_plots:
        plt.show()
    plt.close()

def plot_performance_summary(gradient_results, ga_results, config: Config):
    """Performance summary with grouped bars"""
    fig_size = config.get_figure_size('performance_summary')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    lambda_vals = [r['lambda'] for r in gradient_results]
    x = np.arange(len(lambda_vals))
    width = config.output.bar_width
    
    # Define professional grayscale palette  
    colors = sns.color_palette("Greys", n_colors=6)
    graf_color, ga_color = colors[5], colors[2]  # Dark and medium gray
    
    # Time comparison
    grad_times = [r['time'] for r in gradient_results]
    ga_times = [r['time'] for r in ga_results]
    
    bars1 = ax1.bar(x - width/2, grad_times, width, label='GRAF', 
                   color=graf_color, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, ga_times, width, label='GA', 
                   color=ga_color, edgecolor='black', linewidth=0.8)
    
    ax1.set_xlabel('λ (LPI weight)', fontweight='bold')
    ax1.set_ylabel('Time to convergence (s)', fontweight='bold')
    ax1.set_title('Computational Time Comparison', fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{l:.1f}' for l in lambda_vals])
    
    # Add speedup annotations above GA bars first, then adjust y-axis limits
    max_ga_time = max(ga_times)
    for i, (ga_time, grad_time) in enumerate(zip(ga_times, grad_times)):
        speedup = ga_time / grad_time
        ax1.text(i + width/2, ga_time + max_ga_time * 0.05, f'{speedup:.1f}×', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set y-axis limit to accommodate annotations and position legend outside plot
    ax1.set_ylim(0, max_ga_time * 1.15)  # Space for annotations
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    # PSL comparison
    grad_psl_db = [20*np.log10(r['psl']) for r in gradient_results]
    ga_psl_db = [20*np.log10(r['psl']) for r in ga_results]
    
    bars3 = ax2.bar(x - width/2, grad_psl_db, width, label='GRAF', 
                   color=graf_color, edgecolor='black', linewidth=0.8)
    bars4 = ax2.bar(x + width/2, ga_psl_db, width, label='GA', 
                   color=ga_color, edgecolor='black', linewidth=0.8)
    
    ax2.set_xlabel('λ (LPI weight)', fontweight='bold')
    ax2.set_ylabel('PSL (dB)', fontweight='bold')
    ax2.set_title('Peak Sidelobe Level Achieved', fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{l:.1f}' for l in lambda_vals])
    
    # Add clean difference annotations at consistent height above all bars
    max_psl = max(max(grad_psl_db), max(ga_psl_db))
    min_psl = min(min(grad_psl_db), min(ga_psl_db))
    psl_range = max_psl - min_psl
    
    # Position all difference labels at the same height above the highest bars
    label_height = max_psl + psl_range * 0.08
    
    for i, (grad_psl, ga_psl) in enumerate(zip(grad_psl_db, ga_psl_db)):
        diff = grad_psl - ga_psl  # GRAF improvement (should be negative = better)
        ax2.text(i, label_height, f'{diff:.1f} dB', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Set y-axis limits to accommodate annotations and position legend outside
    ax2.set_ylim(min_psl - psl_range * 0.05, max_psl + psl_range * 0.20)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    # Apply Seaborn styling and tight layout
    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()
    save_path = config.save_path(f'fig2_performance_summary.{config.output.plot_format}')
    plt.savefig(save_path, dpi=config.output.plot_dpi, bbox_inches='tight', facecolor='white')
    if not config.output.save_plots:
        plt.show()
    plt.close()

def plot_waveform_analysis(gradient_results, config: Config):
    """Spectral analysis of GRAF-optimized waveforms"""
    fig_size = config.get_figure_size('spectrum_analysis')
    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    
    examples = [0, len(gradient_results)//2, -1]
    example_names = ['GRAF: PSL Optimized', 'GRAF: Balanced', 'GRAF: LPI Optimized']
    
    # Find maximum power for consistent scaling
    max_power = 0
    for idx in examples:
        s = gradient_results[idx]['waveform']
        S = torch.fft.fft(s)
        power_spectrum = torch.abs(S)**2
        max_power = max(max_power, power_spectrum.max().item())
    
    # Plot with consistent scale
    for i, (idx, name) in enumerate(zip(examples, example_names)):
        ax = axes[i]
        s = gradient_results[idx]['waveform']
        
        # Compute spectrum
        S = torch.fft.fft(s)
        power_spectrum = torch.abs(S)**2
        
        ax.plot(power_spectrum.cpu().numpy(), color='black', linewidth=2)
        ax.set_xlabel('Frequency bin', fontweight='bold')
        ax.set_ylabel('Power', fontweight='bold')
        ax.set_title(f'{name} (λ={gradient_results[idx]["lambda"]:.1f})', 
                    fontweight='bold', pad=15)
        ax.set_ylim([0, max_power * 1.1])
        
        # Add statistics
        spectral_var = gradient_results[idx]['lpi']
        ax.text(0.95, 0.95, f'Var: {spectral_var:.2e}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    # Apply Seaborn styling and save
    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()
    save_path = config.save_path(f'fig3_spectrum_comparison.{config.output.plot_format}')
    plt.savefig(save_path, dpi=config.output.plot_dpi, bbox_inches='tight', facecolor='white')
    if not config.output.save_plots:
        plt.show()
    plt.close()

# ========== Main Execution ==========
def main():
    """Main execution with configuration"""
    # Load configuration
    config = load_config()
    config.print_summary()
    
    logging.info("GRAF: Gradient-based Radar Ambiguity Functions")
    logging.info("="*50)
    
    # Pareto frontier experiment
    logging.info("\n" + "="*50)
    logging.info("Running full Pareto frontier experiment...")
    logging.info("="*50)
    
    gradient_results, ga_results = run_pareto_experiment(config)
    
    # Generate plots
    if config.output.save_plots:
        logging.info("\nGenerating plots...")
        plot_main_results(gradient_results, ga_results, config)
        logging.info(f"✓ Saved: {config.save_path('fig1_main_comparison.' + config.output.plot_format)}")
        
        plot_performance_summary(gradient_results, ga_results, config)
        logging.info(f"✓ Saved: {config.save_path('fig2_performance_summary.' + config.output.plot_format)}")
        
        plot_waveform_analysis(gradient_results, config)
        logging.info(f"✓ Saved: {config.save_path('fig3_spectrum_comparison.' + config.output.plot_format)}")
    
    logging.info("\nExperiment complete! Check output files for results.")

if __name__ == "__main__":
    main()

