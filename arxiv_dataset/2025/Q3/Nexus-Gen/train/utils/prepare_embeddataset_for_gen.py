import torch
import torch.multiprocessing as mp
import os
from tqdm import tqdm
import logging
import json
from modelscope import snapshot_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torchvision import transforms
from PIL import Image
from utils import read_jsonl


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Process %(processName)s] %(levelname)s: %(message)s')


def get_image_embeds(rank, samples, out_embed_dir, output_jsonl, lock, height=252):
    torch.cuda.set_device(rank)
    logging.info(f"Process {os.getpid()} using device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
    path = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype="auto", device_map=f"cuda:{rank}"
    )
    model.eval()
    visual_model = model.visual
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    for sample in tqdm(samples, desc=f"Process {rank}", position=rank):
        try:
            image_path = sample['images'][0]
            image_id = sample['id']
            embed_path = os.path.join(out_embed_dir, f"{image_id}.pt")
            sample['embed_path'] = embed_path

            # transform image (center crop to square)
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size
            scale = max(height / img_width, height / img_height)
            shape = [round(img_height * scale), round(img_width * scale)]
            image_scaled = transforms.functional.resize(image, shape, interpolation=transforms.InterpolationMode.BILINEAR)
            img_transform = transforms.CenterCrop((height, height))
            image = img_transform(image_scaled)

            # Process image
            media_inputs = processor.image_processor(images=[image], videos=None, return_tensors='pt', do_resize=False)
            pixel_values = media_inputs["pixel_values"].to(visual_model.device)
            image_grid_thw = media_inputs["image_grid_thw"].to(visual_model.device)
            pixel_values = pixel_values.type(visual_model.dtype)
            with torch.no_grad():
                image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw).cpu()

            torch.save(image_embeds, embed_path)

            # Lock the writing to output_jsonl
            with lock:
                with open(output_jsonl, 'a', encoding='utf-8') as out_file:
                    out_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    height = 252
    output_folder = "assets/example_datasets/embeds_gen"
    input_jsonl = "assets/example_datasets/gen_decoder_dataset.jsonl"
    output_jsonl = os.path.join(output_folder, "gen_decoder_embeds_dataset.jsonl")
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
    num_processes = min(8, available_gpus)
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
        p = mp.Process(target=get_image_embeds, args=(rank, samples[process_start:process_end], out_embed_dir, output_jsonl, lock, height))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
