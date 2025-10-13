import os
import json
import torch
import torchvision
import pickle
import functools
import argparse
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from loguru import logger
from einops import rearrange

from src.factory import create_model_and_transforms
from train.distributed import world_info_from_env, init_distributed_device
from train.data_utils import WebsightDataset

def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def preprocess_websight_text(sample, tokenizer):
    """
    Preprocess text for WebSight HF.
    Captions are truncated to 1024 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<graph><image>{s.strip()}") for s in sample
    ]
    text = tokenizer(
        sample
    )
    return text["input_ids"][0], text["attention_mask"][0]


def prepare_prompt_template():
    output_template = f"""Generate HTML code for the following webpage idea:"""
    return output_template


def load_model_checkpoint(pth_to_ckpt):
    model_dict = torch.load(pth_to_ckpt)
    args = model_dict["args"]

    model, image_processor, clip_model, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        enable_graph_input=args.enable_graph_input,
    )
    sd = model.state_dict()
    sd_keys = sd.keys()

    # Load trainable parameters from checkpoint
    ckp_sd = model_dict["model_state_dict"]
    for k in sd_keys:
        if k in ckp_sd:
            sd[k] = ckp_sd[k]
    model.load_state_dict(sd, strict=False)

    return model, image_processor, tokenizer, args


def prepare_dataloader(preprocess_image_fn, preprocess_text_fn):
    with open("./websight_hf/data_20100.pickle", 'rb') as handle:
        websight_hf = pickle.load(handle)
    
    benchmark = []
    for idx, dt in enumerate(websight_hf):
        if idx < 20000:
            continue
        processed_image = preprocess_image_fn([dt['image']]).squeeze(0)
        text_input = prepare_prompt_template()
        text_input_ids, attention_mask = preprocess_text_fn([text_input])
        
        processed_dt = dict()
        processed_dt['idx'] = idx
        processed_dt['image'] = processed_image
        processed_dt['text'] = torch.tensor(text_input_ids, dtype=torch.long)
        processed_dt['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        processed_dt['graph'] = dt['graph']
        benchmark.append(processed_dt)
    if torch.distributed.get_rank() == 0:
        logger.info("Completely prepared DataLoader")

    bm = WebsightDataset(benchmark)
    eval_sampler = DistributedSampler(bm, shuffle=True)
    eval_loader = DataLoader(
        bm,
        batch_size=1,
        num_workers=0,
        sampler=eval_sampler,
        pin_memory=True,
        drop_last=True
    )

    return eval_loader, len(eval_loader)
    

class InferenceRunner:
    def __init__(self, checkpoint_path):
        self.model, self.image_processor, self.tokenizer, self.args = load_model_checkpoint(checkpoint_path)
        self.device_id = self._setup_distributed()
        self.cast_dtype = get_cast_dtype(self.args.precision)
        self.preprocess_image_fn = functools.partial(preprocess_image, image_processor=self.image_processor)
        self.preprocess_text_fn = functools.partial(preprocess_websight_text, tokenizer=self.tokenizer)

    def _setup_distributed(self):
        self.args.local_rank, self.args.rank, self.args.world_size = world_info_from_env()
        return init_distributed_device(self.args)

    def _prepare_model(self):
        torch.cuda.empty_cache()
        self.model.to(self.device_id)
        self.model.eval()

    def run_inference(self):
        torch.set_grad_enabled(False)
        self._prepare_model()

        eval_loader, num_batch_eval = prepare_dataloader(self.preprocess_image_fn, self.preprocess_text_fn)
        results = {}

        for batch_websight in tqdm(eval_loader, disable=self.args.rank != 0, total=num_batch_eval):
            sample_idx = batch_websight['idx'].tolist()[0]
            images, input_ids, attention_mask, graph = self._prepare_batch(batch_websight)
            
            output = self.model.generate(
                vision_x=images,
                lang_x=input_ids,
                graph_x=graph,
                attention_mask=attention_mask,
                max_new_tokens=1050,
                num_beams=4
            )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            results[sample_idx] = generated_text
            torch.cuda.empty_cache()

        self._save_results(results)

    def _prepare_batch(self, batch):
        images = batch['image'].to(self.device_id, dtype=self.cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch['text'].to(self.device_id, dtype=self.cast_dtype, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device_id, dtype=self.cast_dtype, non_blocking=True)
        graph = batch['graph'].to(self.device_id) if self.args.enable_graph_input else None
        return images, input_ids, attention_mask, graph

    def _save_results(self, results):
        os.makedirs("./results", exist_ok=True)
        with open(f"results_{self.args.rank}", "w") as file:
            json.dump(results, file)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_to_ckpt", default="model.pt", type=str)
    return parser.parse_args()

def main():
    args = parse_arguments()
    runner = InferenceRunner(args.pth_to_ckpt)
    runner.run_inference()

if __name__ == "__main__":
    main()