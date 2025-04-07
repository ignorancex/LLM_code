import argparse
import json
from sentence_transformers import SentenceTransformer, models
import os


def create_last_token_pooling_model(model_path):
    # Load the transformer model
    word_embedding_model = models.Transformer(model_path)

    # Add a Pooling layer with `last_token` pooling
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=False,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        pooling_mode_lasttoken=True  # Enable last token pooling
    )

    # Combine the transformer and pooling model into a SentenceTransformer
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.tokenizer.pad_token = model.tokenizer.eos_token
    return model


def encode_sentences(sentences, model_path, n_gpus, batch_size):
    model = create_last_token_pooling_model(model_path)

    eos_token = model.tokenizer.eos_token
    if eos_token is None:
        raise ValueError(
            "The tokenizer does not have an EOS token. Please define one or use a different model."
        )

    sentences = [f"{sentence}{eos_token}" for sentence in sentences]
    pool = model.start_multi_process_pool(target_devices=[f"cuda:{i}" for i in range(n_gpus)])
    embeddings = model.encode_multi_process(
        sentences,
        pool,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    model.stop_multi_process_pool(pool)

    return [embed.tolist() for embed in embeddings]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    sentences = set()
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            lst = json.loads(line)["concepts"]
            sentences.update(lst)
    sentences = list(sentences)

    embeddings = encode_sentences(
        sentences,
        args.model_path,
        args.n_gpus,
        args.batch_size
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for sentence, embedding in zip(sentences, embeddings):
            f.write(json.dumps({
                "sentence": sentence,
                "embedding": embedding,
            }) + "\n")


if __name__ == "__main__":
    main()
