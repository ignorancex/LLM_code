from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from watermark_stealing.server import Server
from src.sentence_analyzer import SentenceAnalyzer
import argparse
import os
from tqdm.auto import tqdm
import pickle
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate distribution of statistics")
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset.", default="c4"
    )
    parser.add_argument(
        "--spoofed_only", type=str, help="Only generate spoofed text Y/N", default="N"
    )
    return parser.parse_args()


def get_model_name(cfg):
    model_name = cfg.server.model.name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"
    return model_name

def load_sentences(cfg, dataset, spoofed_only: bool):
    model_name = get_model_name(cfg)
    prefix = f"out/reprompting/{model_name}/{dataset}/"
    attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")
    
    if not spoofed_only:
        rng_device = cfg.meta.rng_device
        if rng_device == "cpu":
            watermarked_sentences = pd.read_json(prefix + "watermarked.jsonl", lines=True)
        else:
            watermarked_sentences = pd.read_json(prefix + "cuda_watermarked.jsonl", lines=True)
    else:
        watermarked_sentences = None
    spoofed_sentences = pd.read_json(f"{prefix}spoofed_{attacker_short_name}.jsonl", lines=True)
    
    return watermarked_sentences, spoofed_sentences

def load_server(cfg):
    cfg.server.model.skip = True
    server = Server(cfg.meta, cfg.server)
    tokenizer = server.model.tokenizer
    return server, tokenizer


def _analyze_sentence(sentences, server):

    analyzers = []
    
    for row in tqdm(sentences.iterrows(), total=len(sentences)):
        row = row[1]
        watermarked_sentence = row.watermarked
        original_sentence = row.original
        try:
            watermarked_analyzer = SentenceAnalyzer(watermarked_sentence, server)
            original_analyzer = SentenceAnalyzer(original_sentence, server)
            analyzers.append((watermarked_analyzer, original_analyzer))
        except ValueError:
            print("Error in sentence")
            print(watermarked_sentence)
            print(original_sentence)
            continue
    return analyzers

def analyze_sentences(cfg, dataset, spoofed_only):
    watermarked_sentences, spoofed_sentences = load_sentences(cfg, dataset, spoofed_only)
    
    server, tokenizer = load_server(cfg)
    
    if not spoofed_only:
        watermarked_analyzers = _analyze_sentence(watermarked_sentences, server)
    else:
        watermarked_analyzers = None
    spoofed_analyzers = _analyze_sentence(spoofed_sentences, server)
    
    return watermarked_analyzers, spoofed_analyzers

def get_out_path(cfg, dataset):
    model_name = get_model_name(cfg)
    prefix = f"data/reprompting/{model_name}/{dataset}/"
    attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")
    
    rng_device = cfg.meta.rng_device
    if rng_device == "cpu":
        watermarked_path = prefix + "watermarked/"
    else:
        watermarked_path = prefix + "cuda_watermarked/"
    spoofed_path = prefix + f"spoofed_{attacker_short_name}/"
    
    return watermarked_path, spoofed_path

def check_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    spoofed_only = args.spoofed_only == "Y" 
    cfg = get_pydantic_models_from_path(args.cfg_path)[0]   
    watermarked_analyzers, spoofed_analyzers = analyze_sentences(cfg, args.dataset, spoofed_only)
    watermarked_path, spoofed_path = get_out_path(cfg, args.dataset)
    check_folder_exists(watermarked_path)
    check_folder_exists(spoofed_path)
    
    if args.spoofed_only == "Y":
        with open(spoofed_path + "analyzers.pkl", "wb") as file:
            pickle.dump(spoofed_analyzers, file)
        return
    
    with open(watermarked_path + "analyzers.pkl", "wb") as file:
        pickle.dump(watermarked_analyzers, file)
    with open(spoofed_path + "analyzers.pkl", "wb") as file:
        pickle.dump(spoofed_analyzers, file)
    
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    
    