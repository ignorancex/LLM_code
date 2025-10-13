"""
Configuration loader for LPI Radar Waveform Design
Provides easy access to configuration parameters from YAML file with execution modes
"""

import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

class Config:
    """Configuration manager for radar waveform experiments with execution modes"""
    
    def __init__(self, config_path: str = "config.yaml", execution_mode: str = None):
        """Load configuration from YAML file and apply execution mode"""
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
            
        # Determine execution mode
        self.execution_mode = execution_mode or self._raw_config.get('execution_mode', 'custom')
        
        # Merge mode-specific config with defaults
        self._config = self._merge_mode_config()
        
        # Setup logging
        self._setup_logging()
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup random seed
        self._setup_random_seed()
        
        # Create results directory
        self._setup_results_dir()
        
    def _merge_mode_config(self):
        """Merge mode-specific configuration with defaults"""
        # Start with base configuration (everything except 'modes')
        merged_config = {}
        for key, value in self._raw_config.items():
            if key not in ['modes', 'execution_mode']:
                merged_config[key] = value
                
        # If we have a specific mode, overlay its settings
        if self.execution_mode in ['quick', 'full'] and 'modes' in self._raw_config:
            mode_config = self._raw_config['modes'].get(self.execution_mode, {})
            merged_config = self._deep_merge(merged_config, mode_config)
            
        return merged_config
        
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def _setup_logging(self):
        """Configure logging based on config"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        log_level = level_map.get(self.logging.level, logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
    def _setup_device(self):
        """Setup compute device"""
        device_config = self.compute.device
        
        if device_config == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
            
        logging.info(f"Using device: {device}")
        return device
        
    def _setup_random_seed(self):
        """Set random seeds for reproducibility"""
        seed = self.compute.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logging.info(f"Random seed set to: {seed}")
        
    def _setup_results_dir(self):
        """Create results directory if it doesn't exist"""
        results_dir = Path(self.output.results_dir)
        results_dir.mkdir(exist_ok=True)
        
    def __getattr__(self, name):
        """Enable dot notation access to config sections"""
        if name in self._config:
            return ConfigSection(self._config[name])
        raise AttributeError(f"Configuration section '{name}' not found")
        
    def get_lambda_values(self) -> List[float]:
        """Get lambda values for Pareto frontier experiment"""
        return self._config['experiment']['lambda_values']
            
    def get_figure_size(self, plot_name: str) -> List[int]:
        """Get figure size for specific plot"""
        return self.output.figure_sizes.get(plot_name, [10, 6])
        
    def save_path(self, filename: str) -> str:
        """Get full path for saving files"""
        return str(Path(self.output.results_dir) / filename)
        
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_update(self._config, updates)
        logging.info("Configuration updated with new values")
        
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()
        
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        
        print(f"Execution Mode: {self.execution_mode}")
        print(f"Signal Length: {self.signal.sequence_length}")
        print(f"Device: {self.device}")
        print(f"Random Seed: {self.compute.random_seed}")
        
        print(f"\nOptimization:")
        print(f"  Gradient - LR: {self.optimization.gradient.learning_rate}, "
              f"Iterations: {self.optimization.gradient.max_iterations}")
        print(f"  GA - Pop: {self.optimization.genetic_algorithm.population_size}, "
              f"Generations: {self.optimization.genetic_algorithm.max_generations}")
              
        print(f"\nLoss Function:")
        print(f"  LPI Weight: {self.loss.lpi_weight}")
        print(f"  Scale Factor: {self.loss.lpi_scale_factor}")
        
        lambda_vals = self.get_lambda_values()
        print(f"\nExperiment: {len(lambda_vals)} lambda values from "
              f"{min(lambda_vals):.2f} to {max(lambda_vals):.2f}")
              
        print(f"\nOutput: {self.output.results_dir}/ ({self.output.plot_format})")
        print("="*50)

    def switch_mode(self, new_mode: str):
        """Switch to a different execution mode"""
        if new_mode not in ['quick', 'full', 'custom']:
            raise ValueError(f"Invalid execution mode: {new_mode}")
            
        self.execution_mode = new_mode
        self._config = self._merge_mode_config()
        
        # Re-setup components that depend on config
        self._setup_logging()
        self.device = self._setup_device()
        self._setup_random_seed()
        self._setup_results_dir()
        
        logging.info(f"Switched to execution mode: {new_mode}")


class ConfigSection:
    """Helper class for dot notation access to config sections"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Configuration parameter '{name}' not found")
        
    def __contains__(self, key):
        return key in self._config
        
    def get(self, key, default=None):
        """Get config value with default"""
        return self._config.get(key, default)
        
    def keys(self):
        """Get all keys in this section"""
        return self._config.keys()
        
    def items(self):
        """Get all key-value pairs"""
        return self._config.items()


# Convenience functions
def load_config(config_path: str = "config.yaml", execution_mode: str = None) -> Config:
    """Load configuration from file with optional execution mode override"""
    return Config(config_path, execution_mode)

def create_mode_config(base_config_path: str = "config.yaml", 
                      mode: str = "quick", 
                      output_path: str = None):
    """Create a standalone config file from a specific mode (for backwards compatibility)"""
    config = load_config(base_config_path, mode)
    
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        print(f"Created standalone config for '{mode}' mode: {output_path}")

# Example usage and validation
if __name__ == "__main__":
    # Test the configuration loader
    try:
        print("Testing execution modes:")
        
        # Test full mode
        config_full = load_config(execution_mode="full")
        print(f"\nFull mode - N={config_full.signal.sequence_length}, "
              f"iterations={config_full.optimization.gradient.max_iterations}")
        
        # Test quick mode
        config_quick = load_config(execution_mode="quick")
        print(f"Quick mode - N={config_quick.signal.sequence_length}, "
              f"iterations={config_quick.optimization.gradient.max_iterations}")
        
        # Test mode switching
        config = load_config()
        config.print_summary()
        
        print(f"\nSwitching from {config.execution_mode} to quick mode...")
        config.switch_mode("quick")
        print(f"New signal length: {config.signal.sequence_length}")
        
    except Exception as e:
        print(f"Configuration test failed: {e}") 