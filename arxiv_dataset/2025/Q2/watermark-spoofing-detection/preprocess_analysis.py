from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from watermark_stealing.server import Server
import argparse
from src.sentence_analyzer import SentenceAnalyzer, batch_save_analyzers
import os
from tqdm.auto import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Text.")
    parser.add_argument("--cfg_path", type=str, help="Path to the config file.")
    parser.add_argument("--dataset", type=str, help="Dataset.", default="c4")
    parser.add_argument("--split", type=str, help="Dataset split", default=None)
    parser.add_argument(
        "--spoofed_only", type=str, help="Only generate spoofed text Y/N", default="N"
    )
    return parser.parse_args()

def process_text_file(path: str):
    line_begin = "###START###"
    with open(path, "r") as file:
        whole_text = file.read()
    return whole_text.split(line_begin)[1:]


def check_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_server(cfg_path: str):
    configs = get_pydantic_models_from_path(cfg_path)
    cfg = configs[0]
    cfg.server.model.skip = True
    server = Server(cfg.meta, cfg.server)
    return cfg, server


def get_model_name(cfg):
    model_name = cfg.server.model.name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"
    return model_name


def save_analyzers(
    completions, path, server, batch_size=100
):
    check_folder_exists(path)
    batch = []
    k = 0
    for completion in tqdm(completions):
        try:
            analyzer = SentenceAnalyzer(
                completion, server,
            )

            if analyzer.color_mask is None:
                continue

            batch.append(analyzer)
        except ValueError:
            print(f"Error in sentence: {completion}")
            continue
        if len(batch) == batch_size:
            batch_save_analyzers(f"{path}{k}.pkl", batch)
            k += 1
            batch = []
    if batch:
        batch_save_analyzers(f"{path}{k}.pkl", batch)


def get_paths(cfg, args):
    model_name = get_model_name(cfg)
    prefix_path = f"out/{model_name}/{args.dataset}"
    prefix_path_analyzer = f"data/sentence_analyzer/{model_name}/{args.dataset}"
    attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")
    
    if args.split is not None:
        prefix_path += f"_{args.split}"
        prefix_path_analyzer += f"_{args.split}"

    rng_device = cfg.meta.rng_device
    if rng_device == "cuda":
        watermarked_path = f"{prefix_path}/cuda_watermarked.txt"
        watermarked_analyzer_path = (
            f"{prefix_path_analyzer}/watermarked_cuda/"
        )
    else:
        watermarked_path = f"{prefix_path}/watermarked.txt"
        watermarked_analyzer_path = (
            f"{prefix_path_analyzer}/watermarked/"
        )

    spoofed_path = (
        f"{prefix_path}/spoofed_{attacker_short_name}.txt"
    )
    spoofed_analyzer_path = (
        f"{prefix_path_analyzer}/spoofed/{attacker_short_name}/"
    )

    return (
        watermarked_path,
        watermarked_analyzer_path,
        spoofed_path,
        spoofed_analyzer_path,
    )


def main():
    args = parse_arguments()
    cfg, server = initialize_server(args.cfg_path)

    watermarked_path, watermarked_save_path, spoofed_path, spoofed_save_path = (
        get_paths(cfg, args)
    )

    # Spoofed text
    if args.split is None:
        spoofed_completions = process_text_file(spoofed_path)
        save_analyzers(
            spoofed_completions,
            spoofed_save_path,
            server,
        )

    if args.spoofed_only == "Y":
        return

    # Watermarked text
    if True:
        watermarked_completions = process_text_file(watermarked_path)
        save_analyzers(
            watermarked_completions,
            watermarked_save_path,
            server,
        )

    
    

if __name__ == "__main__":
    main()
