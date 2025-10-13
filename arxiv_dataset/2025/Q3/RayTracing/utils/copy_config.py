import wandb
import yaml
import sys

def download_wandb_config(run_path, output_file=None):
    """
    Downloads the config dictionary from a wandb run and writes it to a YAML file.
    
    Args:
        run_path (str): The path of the wandb run (e.g. 'myentity/MyProject/abcd1234').
        output_file (str): Path to the output yaml file (default: 'config.yaml').
    """
    try:
        api = wandb.Api()
        run = api.run(run_path)
        config = run.config
        if output_file is None:
            output_file = f"configs/{config['dset']}.yaml"
        # Remove WandB internal keys if desired
        config_cleaned = {k: v for k, v in config.items() if not k.startswith('_')}

        with open(output_file, 'w') as f:
            yaml.dump(config_cleaned, f, sort_keys=False)

        print(f"Config saved to {output_file}")

    except wandb.errors.CommError:
        print(f"Could not fetch run {run_path}. Check your internet connection or the run path.")
    except wandb.errors.Error as e:
        print(f"WandB Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_config.py <wandb_run_path> [output_file]")
    else:
        run_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        download_wandb_config(run_path, output_file)
