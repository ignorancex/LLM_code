import os
import torch
from tqdm import tqdm
import numpy as np

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed

from torch.utils.data import DataLoader
from model.nltoolkit import init_nltk, postprocess_text
from utils.converter import batch_text2traj

from utils.dataloader import get_dataloader
from utils.homography import generate_homography, image2world, world2image
from utils.postprocessor import postprocess_trajectory

import warnings
warnings.filterwarnings('ignore')

from accelerate.logging import get_logger
logger = get_logger(__name__)
from accelerate import Accelerator


@torch.no_grad()
def test(cfg):
    # Initialize the Natural language toolkit
    init_nltk()
    
    # Initialize the accelerator.
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    accelerator_log_kwargs = {}
    if cfg.use_logger:
        accelerator_log_kwargs["log_with"] = cfg.logger_type
        accelerator_log_kwargs["project_dir"] = checkpoint_path

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Reproducibility settings
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Get the datasets
    dataloader = get_dataloader(os.path.join(cfg.dataset_path, cfg.dataset_name), 'test', cfg.obs_len, cfg.pred_len, batch_size=1e8)
    obs_traj = dataloader.dataset.obs_traj.numpy()
    pred_traj = dataloader.dataset.pred_traj.numpy()
    non_linear_ped = dataloader.dataset.non_linear_ped.numpy()
    homography = dataloader.dataset.homography
    scene_id = dataloader.dataset.scene_id
    scene_img = dataloader.dataset.scene_img
    scene_map = dataloader.dataset.scene_map
    seq_start_end = dataloader.dataset.seq_start_end

    batch_size_per_gpu = obs_traj.shape[0] // accelerator.state.num_processes + 1
    if batch_size_per_gpu < cfg.per_device_inference_batch_size:
        print(f"per_device_inference_batch_size is automatically reduced from {cfg.per_device_inference_batch_size} to {batch_size_per_gpu}.")
        cfg.per_device_inference_batch_size = batch_size_per_gpu

    # Scale down the scene
    for k, v in homography.items():
        cfg.image_scale_down = 0.25
        homography[k] = v.copy() @ generate_homography(scale=cfg.image_scale_down)

    preprocessed_test_dataset_name = f"{cfg.dataset_name}-test-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")

    data_files = {}
    data_files["test"] = os.path.join(preprocessed_dataset_path, preprocessed_test_dataset_name)
    
    if not os.path.exists(data_files["test"]):
        raise ValueError(f"Preprocessed dataset files not found: {data_files['train']} or {data_files['validation']}. Please run `./script/preprocessor.sh` first.")
    
    extension = data_files["test"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    # Load the model
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=False, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=False, cache_dir=cfg.cache_dir, use_fast=not cfg.use_slow_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if accelerator.is_local_main_process:
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {count_parameters(model)}")

    # Preprocessing the datasets.
    column_names = raw_datasets["test"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(f"--history_column' value '{cfg.history_column}' needs to be one of: {', '.join(column_names)}")
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(f"--future_column' value '{cfg.future_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if cfg.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[history_column]
        targets = examples[future_column]
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    test_dataset = raw_datasets["test"].map(preprocess_function,
                                            batched=True,
                                            num_proc=cfg.preprocessing_num_workers,
                                            remove_columns=column_names,
                                            load_from_cache_file=not cfg.overwrite_cache,
                                            desc="Running tokenizer on test dataset")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id,)
    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=cfg.per_device_inference_batch_size)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    progress_bar = tqdm(range(len(obs_traj)), desc="Generating", disable=not accelerator.is_local_main_process)
    progress_step = cfg.per_device_inference_batch_size * accelerator.state.num_processes

    all_obs = np.array(raw_datasets['test']['obs_traj']).astype(np.float32)
    all_gts = np.array(raw_datasets['test']['pred_traj']).astype(np.float32)
    all_preds = []
    error_ids = []
    
    for step, batch in enumerate(eval_dataloader):
        if cfg.deterministic:
            # Most-likely prediction
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"].to(device),
                                                                        attention_mask=batch["attention_mask"].to(device),
                                                                        max_length=cfg.max_target_length,
                                                                        num_beams=cfg.num_beams)
        else:
            # Probabilistic sampling
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"].to(device),
                                                                        attention_mask=batch["attention_mask"].to(device),
                                                                        max_length=cfg.max_target_length,
                                                                        do_sample=True,
                                                                        num_return_sequences=cfg.num_samples,
                                                                        temperature=cfg.temperature,
                                                                        top_k=cfg.top_k)
            
        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        generated_tokens = accelerator.gather_for_metrics((generated_tokens.view(-1, cfg.num_samples, generated_tokens.size(-1))))
        generated_tokens = generated_tokens.view(-1, generated_tokens.size(-1)).cpu().numpy()
        generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

        if not cfg.use_slow_tokenizer:
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            # make sure that special tokens are not decoded using sentencepiece model
            filtered_tokens = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
            decoded_preds = tokenizer.sp_model.decode(filtered_tokens.tolist())
        decoded_preds = [pred.strip() for pred in decoded_preds]
        traj_data = batch_text2traj(decoded_preds, frame=cfg.pred_len, dim=2)

        for pid in range(len(traj_data)):
            if traj_data[pid] is None:
                ped_id = cfg.per_device_inference_batch_size * accelerator.state.num_processes * step + pid // cfg.num_samples
                error_ids.append(ped_id)
                # Assume the pedestrian is not moving
                traj_data[pid] = np.tile(all_obs[ped_id, -1], (cfg.pred_len, 1))

        traj_data = np.stack(traj_data, axis=0).reshape(-1, cfg.num_samples, cfg.pred_len, 2)
        all_preds.append(traj_data)
        progress_bar.update(progress_step)

    all_preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    progress_bar.n = len(obs_traj)
    progress_bar.close()
    
    # Evaluate the prediction
    if accelerator.is_local_main_process:
        all_preds = postprocess_trajectory(all_preds, obs_traj, seq_start_end, scene_id, homography, scene_map, cfg)
        ADE = []
        FDE = []
        for ped_id in range(all_preds.shape[0]):
            
            # Homography warping
            if cfg.metric == "pixel":
                H = homography[scene_id[ped_id]]
                all_preds[ped_id] = image2world(all_preds[ped_id], H)
                all_gts[ped_id] = pred_traj[ped_id]

            error = np.linalg.norm(all_preds[ped_id] - all_gts[ped_id], ord=2, axis=-1)
            ADE.append(np.mean(error, axis=-1).min())
            FDE.append(error[:, -1].min())

        print(f"Test dataset: {cfg.dataset_name}")
        print(f"Total pedestrian number: {all_preds.shape[0]}")
        print(f"ADE: {np.mean(ADE)}")
        print(f"FDE: {np.mean(FDE)}")
        
        
if __name__ == "__main__":
    from utils.config import get_exp_config, DotDict
    args = get_exp_config()
    cfg = DotDict(args.__dict__)
    test(cfg)
