import argparse
from data_synthesis import FastConceptSampler as ConceptSampler
import json
from tqdm import tqdm
import random
import os


def load_embeddings(embed_path):
    embeddings = {}
    with open(embed_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Loading embeddings"):
            item = json.loads(line)
            embeddings[item['sentence']] = item['embedding']
    return embeddings


def load_concept_lists(input_file):
    concept_lists = []
    with open(input_file, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            lst = item["concepts"]
            concept_lists.append(lst)
    return concept_lists


def generate_samples(sampler, data_size, difficulty_levels):
    results = []
    concept_text_pool = set()

    with tqdm(total=data_size, desc="Generating samples") as pbar:
        while len(results) < data_size:
            concept_list = sampler.sample_concept_list(size=5, temperature=0.2)
            concept_text = "\n".join(f"{i + 1}. {concept}" for i, concept in enumerate(concept_list))

            if concept_text in concept_text_pool:
                continue

            concept_text_pool.add(concept_text)
            level = random.choice(difficulty_levels)

            prompt = (
                f"Given foundational concepts and difficulty level, identify connections and "
                f"develop a question that integrates these concepts with appropriate complexity.\n\n"
                f"Foundational Concepts:\n{concept_text}\n\nDifficulty Level: {level}"
            )

            results.append({
                "foundational_concepts": concept_list,
                "level": level,
                "prompt": prompt,
            })
            pbar.update(1)

    return results


def save_results(results, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate problem concepts dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--data_size',
        type=int,
        default=2000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--embed_path',
        type=str,
        default="data/embeddings.jsonl",
        help='Path to embeddings file'
    )
    args = parser.parse_args()

    # Difficulty levels configuration
    difficulty_levels = [
        "AMC12", "HMMT-Nov", "HMMT-Feb", "AIME",
        "USAJMO", "USAMO", "USOJMO", "USOMO"
    ]
    # all_levels = {'AMC10', 'HMMT-Nov', 'AIME', 'AHSME',
    #               'HMMT-Feb', 'USOMO', 'AMC12', 'USOJMO',
    #               'AJHSME', 'AMC8', 'USAMO', 'USAJMO'}

    # Load data
    embeddings = load_embeddings(args.embed_path)
    concept_lists = load_concept_lists(args.data_path)

    # Initialize sampler and generate samples
    sampler = ConceptSampler(concept_lists=concept_lists, concept_embeddings=embeddings)
    results = generate_samples(sampler, args.data_size, difficulty_levels)

    # Save results
    save_results(results, args.output_path)


if __name__ == "__main__":
    main()