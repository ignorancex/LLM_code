# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import logging
import re

import openai
import yaml
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cluster_classes_with_openai(classes, num_clusters, client, model_name):
    """
    Use OpenAI's LLM to cluster the given classes into the specified number of clusters.
    """
    # Get clusters' names
    prompt = (
        f"Cluster the following list of classes into {num_clusters} clusters based on their semantic similarity. "
        f"Provide a name for each cluster and list the classes under each cluster. "
        f"Here is the list of classes:\n\n{', '.join(classes)}\n\n"
        f"Output the result in the following format:\n"
        f"Cluster_1: [Suggested name]\n"
        f"Cluster_2: [Suggested name]\n"
        f"...\n"
        f"Don't output anything else but the clusters' names."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=None,
    )
    clusters = []
    content = response.choices[0].message.content.strip()
    for line in content.split("\n"):
        if line.startswith("Cluster_"):
            cluster = line.split(":")[1]
            cluster = re.sub(r"[^a-zA-Z0-9_\s]", "", cluster)
            cluster = cluster.strip()
            clusters.append(cluster)
    logging.info(f"Detected clusters: {clusters}")

    # Match classes to clusters
    clusters_dict = {cluster: [] for cluster in clusters}
    clusters_dict["Unknown"] = []
    for class_ in tqdm(classes, desc="Matching classes to clusters"):
        prompt = (
            f"Classify the following class into one of the categories: {', '.join(clusters)}. "
            f"Here is the class:\n\n{class_}\n\n"
            f"Output the result in the following format:\n"
            f"Cat: [Category name]\n"
            f"...\n"
            f"Don't output anything else but the categories."
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=None,
        )
        content = response.choices[0].message.content.strip()
        class_cluster = content.split(":")[1].strip()
        if class_cluster in clusters_dict:
            clusters_dict[class_cluster].append(class_)
        else:
            matched_cluster = [
                cluster
                for cluster in clusters
                if (class_cluster.lower() in cluster.lower())
                or (cluster.lower() in class_cluster.lower())
            ]

            if matched_cluster:
                class_cluster = matched_cluster[0]
                clusters_dict[class_cluster].append(class_)
            else:
                logging.warning(
                    f"Class '{class_}' does not match any cluster name (Suggested cluster: {class_cluster}). It will be added to the 'unknown' cluster."
                )
                clusters_dict["Unknown"].append(class_)
    # Removing empty clusters
    for cluster in list(clusters_dict.keys()):
        if not clusters_dict[cluster]:
            logging.warning(f"Cluster '{cluster}' is empty. It will be removed.")
            del clusters_dict[cluster]
    # Final number of clusters
    if len(clusters_dict) < num_clusters:
        logging.warning(
            f"Requested {num_clusters} clusters, but only {len(clusters_dict)} clusters available. "
        )
    return clusters_dict


def cluster_classes_with_huggingface(
    classes, num_clusters, pipeline_instance, tokenizer
):
    """
    Use Hugging Face's LLM to cluster the given classes into the specified number of clusters.
    """
    # Get clusters' names
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that clusters classes based on semantic similarity.",
        },
        {
            "role": "user",
            "content": (
                f"Suggest {num_clusters} non-overlaping categories of the following classes based on their semantic similarity. "
                f"Provide a name for each category\n"
                f"Here is the list of classes:\n\n{', '.join(classes)}\n\n"
                f"Output the result in the following format:\n"
                f"Cluster_1: [Suggested name]\n"
                f"Cluster_2: [Suggested name]\n"
                f"...\n"
                f"Don't output anything else but the clusters' names."
            ),
        },
    ]
    response = pipeline_instance(
        messages,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=1.0,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    clusters = []
    content = response[0]["generated_text"].strip()
    for line in content.split("\n"):
        if line.startswith("Cluster_"):
            cluster = line.split(":")[1]
            cluster = re.sub(r"[^a-zA-Z0-9_\s]", "", cluster)
            cluster = cluster.strip()
            clusters.append(cluster)
    logging.info(f"Detected clusters: {clusters}")

    # Match classes to clusters
    clusters_dict = {cluster: [] for cluster in clusters}
    clusters_dict["Unknown"] = []
    for class_ in tqdm(classes, desc="Matching classes to clusters"):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that clusters classes based on semantic similarity.",
            },
            {
                "role": "user",
                "content": (
                    f"Classify the following class into one of the categories: {', '.join(clusters)}. "
                    f"Here is the class:\n\n{class_}\n\n"
                    f"Output the result in the following format:\n"
                    f"Cat: [Category name]\n"
                    f"...\n"
                    f"Don't output anything else but the categories."
                ),
            },
        ]
        response = pipeline_instance(
            messages,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=1.0,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        content = response[0]["generated_text"].strip()
        class_cluster = content.split(":")[1].strip()
        if class_cluster in clusters_dict:
            clusters_dict[class_cluster].append(class_)
        else:
            matched_cluster = [
                cluster
                for cluster in clusters
                if (class_cluster.lower() in cluster.lower())
                or (cluster.lower() in class_cluster.lower())
            ]

            if matched_cluster:
                class_cluster = matched_cluster[0]
                clusters_dict[class_cluster].append(class_)
            else:
                logging.warning(
                    f"Class '{class_}' does not match any cluster name (Suggested cluster: {class_cluster}). It will be added to the 'unknown' cluster."
                )
                clusters_dict["Unknown"].append(class_)
    # Removing empty clusters
    for cluster in list(clusters_dict.keys()):
        if not clusters_dict[cluster]:
            logging.warning(f"Cluster '{cluster}' is empty. It will be removed.")
            del clusters_dict[cluster]
    # Final number of clusters
    if len(clusters_dict) != num_clusters:
        logging.warning(
            f"Requested {num_clusters} clusters, but only {len(clusters_dict)} clusters available. "
        )
    return clusters_dict


def recursive_clustering(
    classes, num_clusters, steps, clustering_type, client, model_name, pipeline_instance
):
    """
    Perform recursive clustering for a specified number of steps.
    """
    if steps == 0 or not classes:
        return classes

    if clustering_type == "openai":
        clusters = cluster_classes_with_openai(
            classes, num_clusters[0], client, model_name
        )
    elif clustering_type == "huggingface":
        tokenizer = pipeline_instance.tokenizer
        clusters = cluster_classes_with_huggingface(
            classes, num_clusters[0], pipeline_instance, tokenizer
        )
    else:
        raise ValueError(f"Unsupported clustering type: {clustering_type}")

    # recursively cluster subclusters
    for name, cls_list in clusters.items():
        clusters[name] = recursive_clustering(
            cls_list,
            num_clusters[1:],
            steps - 1,
            clustering_type,
            client,
            model_name,
            pipeline_instance,
        )
    return clusters


def save_clusters_as_yaml(clusters, output_file):
    """
    Save the clusters to a YAML file.
    """
    with open(output_file, "w") as f:
        yaml.dump(clusters, f, default_flow_style=False)


def load_classes(classes_file):
    """
    Load classes from a file.
    """
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def init_openai_client(variant):
    if variant.lower() == "azure":
        client = AzureOpenAI(
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2023-03-15-preview"
            ),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        )
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        client = openai
    return client


def main():
    parser = argparse.ArgumentParser(
        description="Cluster classes into specified clusters using various models."
    )
    parser.add_argument(
        "--classes_file", type=str, required=True, help="Path to txt file with classes."
    )
    parser.add_argument(
        "--num_clusters",
        type=lambda x: [int(i) for i in x.split(",")],
        required=True,
        help="Comma-separated list of number of clusters for each step. If a single value is provided, it will be used for all steps.",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of recursive clustering steps."
    )
    parser.add_argument(
        "--output_file", type=str, default="clusters.yaml", help="YAML output file."
    )
    parser.add_argument(
        "--clustering_type",
        type=str,
        choices=["openai", "huggingface"],
        required=True,
        help="Clustering type to use.",
    )
    parser.add_argument(
        "--openai_variant",
        type=str,
        choices=["azure", "openai"],
        default="azure",
        help="Azure or OpenAI API for openai type.",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Model name for LLM."
    )
    args = parser.parse_args()

    if len(args.num_clusters) == 1:
        args.num_clusters = [args.num_clusters[0]] * args.steps
    elif len(args.num_clusters) != args.steps:
        raise ValueError(
            f"If the number of clusters is provided as a list, it must match the number of steps.\n"
            f"Got {len(args.num_clusters)} clusters for {args.steps} steps.\n"
            f"For example if 'steps' is 3, the number of clusters should be like: --num_clusters 3,5,7"
        )

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        logging.warning("No .env file found. Set env vars manually.")

    classes = load_classes(args.classes_file)
    logging.info(f"Loaded {len(classes)} classes.")

    client = None
    pipeline_instance = None
    if args.clustering_type == "openai":
        client = init_openai_client(args.openai_variant)

    elif args.clustering_type == "huggingface":
        pipeline_instance = pipeline("text-generation", model=args.model_name)

    clusters = recursive_clustering(
        classes,
        args.num_clusters,
        args.steps,
        args.clustering_type,
        client,
        args.model_name,
        pipeline_instance,
    )

    save_clusters_as_yaml(clusters, args.output_file)
    logging.info(f"Clusters saved to {args.output_file}.")


if __name__ == "__main__":
    main()
