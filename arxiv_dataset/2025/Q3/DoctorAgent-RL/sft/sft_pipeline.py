#!/usr/bin/env python3
import os
import argparse
import yaml
import subprocess
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTPipeline:
    """
    The pipeline for SFT:
        1. Generate SFT data
        2. Finetune the model using LoRA
        3. Merge the base model with LoRA weights
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_type = config['sft']['env_type']
        self.base_model = config['model']['base_model']
        self.output_dir = config['sft']['output_dir']
        
        # Create output directory if it doesn't exist
        # assert not os.path.exists(self.output_dir), f"Output directory {self.output_dir} already exists"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Environment-specific configurations
        self.env_configs = {
            'sokoban': {
                'DIM_X': config['sft']['sokoban']['dim_x'],
                'DIM_Y': config['sft']['sokoban']['dim_y'],
                'NUM_BOXES': config['sft']['sokoban']['num_boxes'],
                'MAX_STEPS': config['sft']['sokoban']['max_steps'],
                'SEARCH_DEPTH': config['sft']['sokoban']['search_depth']
            },
            'frozenlake': {
                'SIZE': config['sft']['frozenlake']['size'],
                'P': config['sft']['frozenlake']['p'],
                'IS_SLIPPERY': config['sft']['frozenlake']['is_slippery']
            },
            'medical_consultation': {
                'USE_ENV_LLM': config['sft'].get('medical_consultation', {}).get('use_env_llm', False),
                'ENV_LLM_MODEL': config['sft'].get('medical_consultation', {}).get('env_llm_model', None),
                'ENV_LLM_MAX_LENGTH': config['sft'].get('medical_consultation', {}).get('env_llm_max_length', 512),
                'ENV_LLM_TEMPERATURE': config['sft'].get('medical_consultation', {}).get('env_llm_temperature', 0.7)
            }
        }

    def generate_data(self) -> None:
        """Generate SFT data for the specified environment."""
        logger.info(f"Generating SFT data for {self.env_type}")
        
        # Set environment variables based on env type
        if self.env_type in self.env_configs:
            for key, value in self.env_configs[self.env_type].items():
                os.environ[key] = str(value)
        
        data_gen_config = self.config['sft']['data_generation']
        output_dir = os.path.join(data_gen_config['data_dir'], self.env_type)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python -m",
            f"sft.utils.generate_sft_verl_{self.env_type}",
            f"--env {self.env_type}",
            f"--algo {data_gen_config['algo']}",
            f"--seed {data_gen_config['seed']}",
            f"--output {output_dir}",
            f"--train_size {data_gen_config['train_size']}",
            f"--test_size {data_gen_config['test_size']}",
            f"--bfs_max_depths {data_gen_config['bfs_max_depths']}",
            f"--prefix {data_gen_config['prefix']}",
            f"--num_processes {data_gen_config['num_processes']}"
        ]
        
        subprocess.run(" ".join(cmd), shell=True, check=True)
        logger.info("Data generation completed")

    def finetune_model(self) -> str:
        """Finetune the model using LoRA."""
        logger.info("Starting model finetuning")
        
        training_config = self.config['sft']['training']
        lora_output_dir = os.path.join(self.output_dir, "lora_weights")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={training_config['num_gpus']}",
            "-m ragen.trainer.fsdp_sft_trainer",
            f"data.train_files=data/sft/{self.env_type}/train.parquet",
            f"data.val_files=data/sft/{self.env_type}/test.parquet",
            "data.prompt_key=prompt",
            "data.response_key=response",
            f"data.max_length={training_config['max_length']}",
            f"optim.lr={training_config['learning_rate']}",
            f"data.train_batch_size={training_config['train_batch_size']}",
            f"data.micro_batch_size={training_config['micro_batch_size']}",
            f"trainer.default_local_dir={lora_output_dir}",
            f"trainer.experiment_name={training_config['experiment_name']}",
            f"trainer.project_name={training_config['project_name']}",
            f"trainer.logger={training_config['logger']}",
            f"trainer.total_epochs={training_config['epochs']}",
            f"trainer.default_hdfs_dir=null", # NOTE hard code here
            f"trainer.validate_before_training={str(training_config.get('validate_before_training', True)).lower()}",
            f"model.lora_rank={training_config['lora_rank']}",
            f"model.lora_alpha={training_config['lora_alpha']}",
            f"model.target_modules={training_config['target_modules']}",
            f"model.enable_gradient_checkpointing={str(training_config.get('enable_gradient_checkpointing', False)).lower()}",
            f"model.partial_pretrain={training_config['base_model']}",
            
        ]

        log_file = os.path.join(lora_output_dir, "train.log")
        cmd.extend([f"2>&1 | tee {log_file}"])
        
        subprocess.run(" ".join(cmd), shell=True, check=True)
        logger.info("Model finetuning completed")
        return lora_output_dir

    def merge_model(self, lora_path: str) -> None:
        """Merge the base model with LoRA weights."""
        logger.info("Merging base model with LoRA weights")
        
        merged_model_path = os.path.join(self.output_dir, "merged_model")
        assert not os.path.exists(merged_model_path), f"Merged model path {merged_model_path} already exists"

        # read information from log file to find checkpoint with lowest validation loss
        log_file = os.path.join(lora_path, "train.log")
        min_val_loss = float('inf')
        best_step = 0
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "val/loss:" in line:
                    step = int(line.split("step:")[1].split(" -")[0])
                    val_loss = float(line.split("val/loss:")[1].strip())
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_step = step
        
        if best_step == 0:
            raise ValueError("No validation loss found in log file, finetuning failed")
        checkpoint_path = os.path.join(lora_path, f"global_step_{best_step}")
        
        cmd = [
            "python -m sft.utils.merge_lora",
            f"--base_model_name={self.base_model}",
            f"--lora_model_path={checkpoint_path}",
            f"--output_path={merged_model_path}"
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)
        
        logger.info(f"Model merged from {checkpoint_path} and saved to {merged_model_path}")
        return merged_model_path
    
    def validate_model(self, merged_model_path: str) -> None:
        """Validate the model on the validation set using the RL script."""
        logger.info(f"Validating model from {merged_model_path}")
        
        log_file = os.path.join(merged_model_path, "validate.log")
        max_prompt_length = (self.config['training']['max_start_length'] +
                            self.config['training']['max_response_length'] * (self.config['training']['max_turns'] - 1) +
                            self.config['training']['max_obs_length'] * self.config['training']['max_turns'])
    
        # Define the command template with proper indentation
        env_kwargs = self.config['env']['env_kwargs']
        env_kwargs_str = " \\\n    ".join([
            f"+env.{key}={value}" for key, value in env_kwargs.items()
        ])
        
        
        cmd = [
            f"VLLM_ATTENTION_BACKEND={self.config['system']['vllm_attention_backend']}",
            f"CUDA_VISIBLE_DEVICES={self.config['system']['cuda_visible_devices']}",
            "python -m ragen.trainer.main_ppo",
            f"data.train_files={self.config['env']['data_dir']}/train.parquet",
            f"data.val_files={self.config['env']['data_dir']}/test.parquet",
            f"data.train_data_num={self.config['training']['train_data_num'] or 'null'}",
            f"data.val_data_num={self.config['training']['val_data_num'] or 'null'}",
            f"data.train_batch_size={self.config['training']['train_batch_size']}",
            f"data.val_batch_size={self.config['training']['val_batch_size']}",
            f"data.max_prompt_length={max_prompt_length}",
            f"data.max_response_length={self.config['training']['max_response_length']}",
            f"data.max_start_length={self.config['training']['max_start_length']}",
            f"data.max_obs_length={self.config['training']['max_obs_length']}",
            "data.shuffle_train_dataloader=True",
            f"algorithm.adv_estimator={self.config['optimization']['adv_estimator']}",
            f"actor_rollout_ref.model.path={merged_model_path}",
            f"actor_rollout_ref.model.enable_gradient_checkpointing={str(self.config['model']['gradient_checkpointing']).lower()}",
            f"actor_rollout_ref.actor.optim.lr={self.config['optimization']['actor_lr']}",
            f"actor_rollout_ref.actor.use_kl_loss={self.config['training']['use_kl_loss']}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.config['training']['ppo_batch_size']}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size={self.config['training']['micro_batch_size']}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size={self.config['training']['micro_batch_size']}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config['training']['rollout_tp_size']}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.config['optimization']['gpu_memory_utilization']}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size={self.config['training']['micro_batch_size']}",
            f"actor_rollout_ref.actor.kl_loss_coef={self.config['optimization']['kl_coef']}", # for use_kl_loss=True. ARPO/BRPO/GRPO needs the original model with "low_var_kl"
            f"actor_rollout_ref.actor.kl_loss_type={self.config['optimization']['kl_loss_type']}",
            f"algorithm.no_think_rl={self.config['training']['no_think_rl']}",
            f"actor_rollout_ref.rollout.n_agent={self.config['training']['n_rollout']}",
            f"actor_rollout_ref.rollout.temperature={self.config['training']['temperature']}",
            f"actor_rollout_ref.actor.state_masking={self.config['training']['state_masking']}",
            f"trainer.logger={self.config['logging']['mode']}",
            f"+trainer.val_only=true",
            f"+trainer.val_before_train=true",
            f"trainer.default_hdfs_dir={self.config['trainer']['default_hdfs_dir'] or 'null'}",
            f"trainer.n_gpus_per_node={self.config['system']['n_gpus']}",
            f"trainer.nnodes={self.config['trainer']['nnodes']}",
            f"trainer.save_freq={self.config['trainer']['save_freq']}",
            f"trainer.test_freq={self.config['trainer']['test_freq']}",
            f"trainer.project_name={self.config['trainer']['project_name']}",
            f"trainer.experiment_name={self.config['model']['experiment_name']}",
            f"trainer.total_epochs={self.config['training']['total_epochs']}",
            f"trainer.total_training_steps={self.config['training']['total_training_steps'] or 'null'}",
            f"env.name={self.config['env']['name']}",
            env_kwargs_str,
            f"max_turns={self.config['training']['max_turns']}",
            f"logging.log_images={str(self.config['logging']['log_images']).lower()}",
            f"logging.log_image_dir={self.config['logging']['log_image_dir']}",
            f"logging.log_image_step_size={self.config['logging']['log_image_step_size']}",
            f"logging.log_n_image_per_batch={self.config['logging']['log_n_image_per_batch']}",
            f"2>&1 | tee {log_file}"
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)
        
        logger.info(f"Model validation completed, log saved to {log_file}")

    def run(self) -> None:
        """Run the complete SFT pipeline."""
        try:
            # Step 1: Generate SFT data
            self.generate_data()
            
            # Step 2: Finetune the model
            lora_path = self.finetune_model()
            
            # Step 3: Merge the model
            merged_model_path = self.merge_model(lora_path)
            
            # Step 4: Validate the model
            self.validate_model(merged_model_path)
            
            logger.info("SFT pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in SFT pipeline: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run SFT Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--env_type', type=str, required=None, default=None, help='Environment type')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    if args.env_type is not None:
        config['sft']['env_type'] = args.env_type
    # Run pipeline
    pipeline = SFTPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()