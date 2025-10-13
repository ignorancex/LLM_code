import os
import logging
import json
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from utils import read_jsonl


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Process %(processName)s] %(levelname)s: %(message)s')

def get_target_embeddings(images, messages, processor, model, num_img_tokens=81):
    images[-1] = images[-1].resize((252, 252))

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    input_embeds = model.model.embed_tokens(inputs['input_ids'])
    image_embeds = model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    ground_truth_image_embeds = image_embeds[-num_img_tokens:]
    input_image_embeds = image_embeds[:-num_img_tokens]

    image_mask = inputs['input_ids'] == model.config.image_token_id
    indices = image_mask.cumsum(dim=1)
    input_image_mask = torch.logical_and(indices <= (image_embeds.shape[0] - ground_truth_image_embeds.shape[0]), image_mask)
    gt_image_mask = torch.logical_and(image_mask, ~input_image_mask)
    input_image_mask = input_image_mask.unsqueeze(-1).expand_as(input_embeds)
    input_embeds = input_embeds.masked_scatter(input_image_mask, input_image_embeds)

    image_prefill_embeds = model.image_prefill_embeds(
        torch.arange(81, device=model.device).long()
    )
    input_embeds = input_embeds.masked_scatter(gt_image_mask.unsqueeze(-1).expand_as(input_embeds), image_prefill_embeds)

    position_ids, _ = model.get_rope_index(inputs['input_ids'],
                                                inputs['image_grid_thw'],
                                                attention_mask=inputs['attention_mask'])
    position_ids = position_ids.contiguous()
    outputs = model(inputs_embeds=input_embeds, position_ids=position_ids, attention_mask=inputs['attention_mask'], return_dict=True)
    output_image_embeddings = outputs.image_embeddings[:, :-1, :]
    output_image_embeddings = output_image_embeddings[gt_image_mask[:, 1:]]
    return output_image_embeddings, input_image_embeds

def get_image_embeds(rank, samples, out_embed_dir, output_jsonl, lock):
    torch.cuda.set_device(rank)
    logging.info(f"Process {os.getpid()} using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
    ckpt_path = 'models/Nexus-GenV2'
    model_config = AutoConfig.from_pretrained(ckpt_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path,
                                                               config=model_config,
                                                               trust_remote_code=True,
                                                               torch_dtype="auto",
                                                               device_map=f"cuda:{rank}")
    processor = Qwen2_5_VLProcessor.from_pretrained(ckpt_path)

    for sample in tqdm(samples, desc=f"Process {rank}", position=rank):
        try:
            image_id = sample['id']
            sample['embed_source'] = os.path.join(out_embed_dir, f"{image_id}_0.pt")
            sample['embed_target'] = os.path.join(out_embed_dir, f"{image_id}_1.pt")

            # transform image
            images = sample['images']
            images = [Image.open(image).convert('RGB') for image in images]
            images[0] = images[0].resize((504, 504))
            images[1] = images[1].resize((252, 252))
            with torch.no_grad():
                # process source image (image[0]) to 324 tokens, process target image (image[1]) to 81 tokens
                target_image_embeddings, source_image_embeddings = get_target_embeddings(images, sample['messages'], processor, model, num_img_tokens=81)
            assert target_image_embeddings.shape[0] == 81, f"Target image embeddings should have 81 tokens, got {target_image_embeddings.shape[1]}"
            assert source_image_embeddings.shape[0] == 324, f"Source image embeddings should have 324 tokens, got {source_image_embeddings.shape[1]}"
            torch.save(source_image_embeddings, sample['embed_source'])
            torch.save(target_image_embeddings, sample['embed_target'])

            # Lock the writing to output_jsonl
            with lock:
                with open(output_jsonl, 'a', encoding='utf-8') as out_file:
                    out_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"Error processing image {image_id}: {e}")


if __name__ == "__main__":
    output_folder = "assets/example_datasets/embeds_edit"
    input_jsonl = "assets/example_datasets/edit_decoder_dataset.jsonl"
    output_jsonl = os.path.join(output_folder, "edit_decoder_embeds_dataset.jsonl")
    out_embed_dir = os.path.join(output_folder, "embeds")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(out_embed_dir, exist_ok=True)
    image_field = 'id'

    processed_images = set()
    print('reading processed images...')
    processed_samples = read_jsonl(output_jsonl) if os.path.exists(output_jsonl) else []
    processed_images = set([sample[image_field] for sample in processed_samples if image_field in sample])
    print(f'finished reading {len(processed_images)} processed images')

    samples = []
    samples = read_jsonl(input_jsonl)
    samples = [sample for sample in samples if image_field in sample and sample[image_field] not in processed_images]
    print(f"reading {len(samples)} new images")

    available_gpus = torch.cuda.device_count()
    num_processes = min(1, available_gpus)
    mp.set_start_method('spawn', force=True)

    # Create a lock for file writing synchronization
    manager = mp.Manager()
    lock = manager.Lock()

    processes = []
    total_missing = len(samples)
    chunk_size = total_missing // num_processes
    for rank in range(num_processes):
        process_start = rank * chunk_size
        process_end = (rank + 1) * chunk_size if rank < num_processes - 1 else total_missing
        p = mp.Process(target=get_image_embeds, args=(rank, samples[process_start:process_end], out_embed_dir, output_jsonl, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
