from src.sentence_analyzer import load_sentence_analyzer
from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from watermark_stealing.server import Server
import glob
import pickle as pkl
import argparse
import pandas as pd
from src.ngram_counter import load_ngram_counter_from_cfg
from scipy import stats
import numpy as np
import pickle
import os
from watermark_stealing.watermarks.kgw.alternative_prf_schemes import (
    seeding_scheme_lookup,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Text.")
    parser.add_argument("--cfg_path", type=str, help="Path to the config file.")
    parser.add_argument("--reprompting", type=str, help="Use reprompting, Y/N?")
    parser.add_argument("--dataset", type=str, help="Dataset.", default="c4")
    parser.add_argument(
        "--ordered", type=str, help="Use ordered ngram Y/N.", default="N"
    )
    parser.add_argument(
        "--token_target", type=int, help="Token target normalization.", default=500
    )
    parser.add_argument(
        "--overwrite", type=str, help="Overwrite the existing files. Y/N?", default="N"
    )
    parser.add_argument(
        "--ngrams_score",
        type=str,
        help="Use ngrams score instead of unigram",
        default="N",
    )
    parser.add_argument(
        "--ngram_dataset",
        type=str,
        default="c4"
    )
    return parser.parse_args()


def get_model_name(cfg):
    model_name = cfg.server.model.name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"
    return model_name


def load_reprompting_analyzers(cfg, dataset):
    model_name = get_model_name(cfg)
    prefix = f"data/reprompting/{model_name}/{dataset}/"
    attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")

    rng_device = cfg.meta.rng_device
    if rng_device == "cpu":
        watermarked_path = prefix + "watermarked/"
    else:
        watermarked_path = prefix + "cuda_watermarked/"

    spoofed_path = prefix + f"spoofed_{attacker_short_name}/"

    with open(watermarked_path + "analyzers.pkl", "rb") as f:
        watermarked_analyzers = pickle.load(f)
    with open(spoofed_path + "analyzers.pkl", "rb") as f:
        spoofed_analyzers = pickle.load(f)

    return watermarked_analyzers, spoofed_analyzers


def load_sentence_analyzer_from_config(cfg, dataset: str):
    server, tokenizer = load_server(cfg)

    # Watermarked text
    model_name = cfg.server.model.name
    model_name = model_name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"

    rng_device = cfg.meta.rng_device
    if rng_device == "cpu":
        watermarked_path = f"data/sentence_analyzer/{model_name}/{dataset}/watermarked/"
    else:
        watermarked_path = (
            f"data/sentence_analyzer/{model_name}/{dataset}/watermarked_cuda/"
        )

    # Spoofed text
    short_name = cfg.attacker.model.short_str().replace("/", "_")
    spoofed_path = (
        f"data/sentence_analyzer/{model_name}/{dataset}/spoofed/{short_name}/"
    )

    # Get all files within folders
    watermarked_files = glob.glob(f"{watermarked_path}*.pkl")
    spoofed_files = glob.glob(f"{spoofed_path}*.pkl")

    watermarked_analyzers, spoofed_analyzers = [], []

    for file in watermarked_files:
        with open(file, "rb") as f:
            data = pkl.load(f)
        watermarked_analyzers += [
            load_sentence_analyzer(None, server, analyzer) for analyzer in data
        ]

    for file in spoofed_files:
        with open(file, "rb") as f:
            data = pkl.load(f)
        spoofed_analyzers += [
            load_sentence_analyzer(None, server, analyzer) for analyzer in data
        ]

    return watermarked_analyzers, spoofed_analyzers


def load_server(cfg):
    cfg.server.model.skip = True
    server = Server(cfg.meta, cfg.server)
    tokenizer = server.model.tokenizer
    return server, tokenizer


def compute_pvalue_reprompting(analyzer_tuple, unigram, gamma, context_length):
    watermarked_analyzer, original_analyzer = analyzer_tuple

    if original_analyzer.compute_z_score(gamma) < 4:
        return np.nan, np.nan
    sampled_stat, sampled_length = watermarked_analyzer.compute_correlation(
        unigram, context_length
    )
    og_stat, og_length = original_analyzer.compute_correlation(unigram, context_length)

    scale1 = np.sqrt(1.06 / (sampled_length - 3))
    scale2 = np.sqrt(1.06 / (og_length - 3))

    scale = np.sqrt(scale1**2 + scale2**2)
    z_score = (sampled_stat - og_stat) / scale

    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    return pvalue, z_score


def compute_pvalue_normal(analyzer, unigram, gamma, context_length):
    stat, length = analyzer.compute_correlation(unigram, context_length)
    scale = np.sqrt(1.06 / (length - 3))
    z_score = stat / scale

    # One sided test (spoofer correlation should be higher)
    pvalue = 1 - stats.norm.cdf(z_score)

    return pvalue, z_score


def merge_reprompting_analyzer_to_size(analyzers, token_target: int, server, distilled):
    # Shuffle the analyzers
    np.random.seed(0)
    np.random.shuffle(analyzers)

    tokenizer = server.model.tokenizer

    merged_analyzers = []
    current_analyzer1, current_analyzer2 = analyzers[0]
    if not distilled:
        trim_beginning(current_analyzer1)
    for analyer_tuple in analyzers[1:]:
        analyzer1, analyzer2 = analyer_tuple
        if not distilled:
            throw = filter_analyzer(analyzer1, tokenizer)
        else:
            throw = False

        if throw:
            continue

        if not distilled:
            trim_beginning(analyzer1)

        length1, length2 = analyzer1.get_length(), analyzer2.get_length()
        min_length = min(length1, length2)
        analyzer1.shallow_clean_sentence(min_length)
        analyzer2.shallow_clean_sentence(min_length)

        if current_analyzer2.get_length() >= token_target:
            current_analyzer1.shallow_clean_sentence(token_target)
            current_analyzer2.shallow_clean_sentence(token_target)

            merged_analyzers.append((current_analyzer1, current_analyzer2))
            current_analyzer1, current_analyzer2 = analyer_tuple
        else:
            current_analyzer1.merge(analyzer1)
            current_analyzer2.merge(analyzer2)

    return merged_analyzers


def filter_analyzer(analyzer, tokenizer):
    encoded_sentence = analyzer._get_tokenized_sentence()
    prefix = tokenizer.decode(encoded_sentence[:20])

    if "I apologize" in prefix:
        return True
    return False


def trim_beginning(analyzer, n_trim: int = 10):
    encoded_sentence = analyzer._get_tokenized_sentence()

    if len(encoded_sentence) < n_trim:
        return

    new_encoded_sentence = encoded_sentence[n_trim:]

    analyzer.color_mask = analyzer.color_mask[n_trim:]
    analyzer.sentence_tokens = new_encoded_sentence


def merge_analyzer_to_size(analyzers, token_target, server):
    merged_analyzers = []

    current_analyzer = analyzers[0]

    for analyzer in analyzers[1:]:
        if current_analyzer.get_length() < token_target:
            current_analyzer.merge(analyzer)
        else:
            current_analyzer.shallow_clean_sentence(
                token_target
            )
            merged_analyzers.append(current_analyzer)
            current_analyzer = analyzer

    if current_analyzer.get_length() >= token_target:
        current_analyzer.shallow_clean_sentence(
            token_target
        )
        merged_analyzers.append(current_analyzer)

    return merged_analyzers


def main(args):
    cfg_path = args.cfg_path
    reprompting = args.reprompting == "Y"
    token_target = args.token_target
    overwrite = args.overwrite == "Y"

    cfg = get_pydantic_models_from_path(cfg_path)[0]

    server_name = cfg.server.model.name.replace("/", "_")
    delta = cfg.server.watermark.generation.delta
    gamma = cfg.server.watermark.generation.gamma
    seeding_scheme = cfg.server.watermark.generation.seeding_scheme

    attacker_name = cfg.attacker.model.short_str().replace("/", "_")
    test_type = "normal" if not reprompting else "reprompting"
    save_path = f"data/pvalues/{server_name}/{delta}/{gamma}/{seeding_scheme}/{attacker_name}/{args.dataset}/ngramdataset_{args.ngram_dataset}/{token_target}/{test_type}/pvalues.csv"

    distilled = True if "cygu" in attacker_name else False

    if not overwrite:
        if os.path.exists(save_path):
            print(f"Path exists: {save_path}. Skipping.")
            return

    _, context_size, _, _ = seeding_scheme_lookup(seeding_scheme)
    if seeding_scheme == "selfhash":
        context_size -= 1

    if reprompting:
        watermarked_analyzers, spoofed_analyzers = load_reprompting_analyzers(
            cfg, args.dataset
        )
    else:
        watermarked_analyzers, spoofed_analyzers = load_sentence_analyzer_from_config(
            cfg, args.dataset
        )

    if len(spoofed_analyzers) == 0:
        print("No spoofed analyzers")
        return

    ordered = args.ordered == "Y"
    if args.ngrams_score == "Y":
        ngram_size = context_size + 1
        unigram = load_ngram_counter_from_cfg(
            cfg, ngram_size, ordered=ordered, dataset=args.ngram_dataset
        )
        context_size = 0
    else:
        unigram = load_ngram_counter_from_cfg(
            cfg, 1, ordered=ordered, dataset=args.ngram_dataset
        )
        context_size = None

    server, _ = load_server(cfg)

    if reprompting:
        # Cleaning step
        watermarked_analyzers = [
            analyzer
            for analyzer in watermarked_analyzers
            if analyzer[1].sentence_tokens is not None
        ]
        watermarked_analyzers = [
            analyzer
            for analyzer in watermarked_analyzers
            if analyzer[0].sentence_tokens is not None
        ]
        spoofed_analyzers = [
            analyzer
            for analyzer in spoofed_analyzers
            if analyzer[1].sentence_tokens is not None
        ]
        spoofed_analyzers = [
            analyzer
            for analyzer in spoofed_analyzers
            if analyzer[0].sentence_tokens is not None
        ]

        # Merge the analyzers to the target size
        watermarked_analyzers = merge_reprompting_analyzer_to_size(
            watermarked_analyzers, token_target, server, distilled
        )
        spoofed_analyzers = merge_reprompting_analyzer_to_size(
            spoofed_analyzers, token_target, server, distilled
        )

        # Compute the p-values
        pvalues_watermarked, pvalues_spoofed, zscores_watermarked, zscores_spooed = (
            [],
            [],
            [],
            [],
        )
        for analyzer_tuple in watermarked_analyzers:
            pvalue, zscore = compute_pvalue_reprompting(
                analyzer_tuple, unigram, gamma, context_size
            )
            pvalues_watermarked.append(pvalue)
            zscores_watermarked.append(zscore)
        for analyzer_tuple in spoofed_analyzers:
            pvalue, zscore = compute_pvalue_reprompting(
                analyzer_tuple, unigram, gamma, context_size
            )
            pvalues_spoofed.append(pvalue)
            zscores_spooed.append(zscore)

    else:
        watermarked_analyzers = [
            analyzer
            for analyzer in watermarked_analyzers
            if analyzer.sentence_tokens is not None
        ]
        spoofed_analyzers = [
            analyzer
            for analyzer in spoofed_analyzers
            if analyzer.sentence_tokens is not None
        ]

        # Merge the analyzers to the target size
        watermarked_analyzers = merge_analyzer_to_size(
            watermarked_analyzers, token_target, server
        )
        spoofed_analyzers = merge_analyzer_to_size(
            spoofed_analyzers, token_target, server
        )

        # Ensure the watermark detects
        watermarked_analyzers = [
            analyzer
            for analyzer in watermarked_analyzers
            if analyzer.compute_z_score(gamma) > 4
        ]
        spoofed_analyzers = [
            analyzer
            for analyzer in spoofed_analyzers
            if analyzer.compute_z_score(gamma) > 4
        ]

        if len(spoofed_analyzers) == 0:
            print("No spoofed analyzers")
            return

        # Compute the p-values
        pvalues_watermarked = [
            compute_pvalue_normal(analyzer, unigram, gamma, context_size)
            for analyzer in watermarked_analyzers
        ]
        pvalues_spoofed = [
            compute_pvalue_normal(analyzer, unigram, gamma, context_size)
            for analyzer in spoofed_analyzers
        ]

        pvalues_watermarked, zscores_watermarked = zip(*pvalues_watermarked)
        pvalues_spoofed, zscores_spooed = zip(*pvalues_spoofed)

    pvalues = pvalues_watermarked + pvalues_spoofed
    zscores = zscores_watermarked + zscores_spooed
    pvalues_type = ["watermarked"] * len(pvalues_watermarked) + ["spoofed"] * len(
        pvalues_spoofed
    )

    distilled = "Y" if "cygu" in server_name else "N"

    p_values_df = pd.DataFrame(
        {
            "pvalue": pvalues,
            "zscore": zscores,
            "type": pvalues_type,
            "server": server_name,
            "attacker": attacker_name,
            "dataset": args.dataset,
            "token_target": token_target,
            "test_type": test_type,
            "delta": delta,
            "gamma": gamma,
            "seeding_scheme": seeding_scheme,
            "distilled": distilled,
            "ngram_dataset": args.ngram_dataset
        }
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    p_values_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
