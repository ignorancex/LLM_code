import json
from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter

input_path_l = ["translated_gpt4o_results.jsonl", "translated_results_llama_original.jsonl", "translated_results_llama.jsonl",
                "translated_results_mistral_original_cleaned.jsonl", "translated_results_mistral_cleaned.jsonl"]

predictions = []
references = []

for input_path in input_path_l:
    # Read JSONL file
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item.get("Model Output (Hutsul)", "").strip()
            ref = item.get("Reference (Hutsul)", "").strip()

            if pred and ref:
                predictions.append(pred)
                references.append(ref)

    file_references = [[ref] for ref in references]

    bleu = corpus_bleu(predictions, file_references)
    chrf = corpus_chrf(predictions, file_references, word_order=2)
    ter = corpus_ter(predictions, file_references)

    print("\n=== Evaluation Results ===")
    print(input_path)
    print(f"Number of predictions: {len(predictions)}")
    print(f"BLEU:  {bleu.score:.2f}")
    print(f"chrF++: {chrf.score:.2f}")
    print(f"TER:   {ter.score:.2f}")

    predictions = []
    references = []

