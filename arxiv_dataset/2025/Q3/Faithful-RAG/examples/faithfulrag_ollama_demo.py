import asyncio
from datasets import Dataset
import numpy as np
import json
import os

from faithfulrag import FaithfulRAG 

async def main():
    # 1. Load dataset
    with open('./datas/faitheval_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_pandas(data)
    
    # 2. Initialize FaithfulRAG pipeline
    rag = FaithfulRAG(
        backend_type="ollama",          # Using OpenAI backend
        model_name="llama3.1",
        similarity_model="bge-large-en-v1.5",  # Sentence Transformer model
        base_url = 'http://localhost:11434',
    )
    
    # 3. Generate self-consistent facts
    print("Generating self-consistent facts...")
    self_facts = await rag.get_self_facts(
        dataset,
        fact_mining_type="default",
    )
    print(f"Generated facts sample: {self_facts[0]['facts'][:1]}\n")
    
    # 4. Retrieve top-k contextual chunks
    print("Retrieving contextual chunks...")
    topk_chunks = rag.get_topk_chunks(
        dataset,
        self_facts
    )
    print(f"Top chunks sample: {topk_chunks[0]['topk_chunks'][0]}\n")
    
    # 5. Generate predictions
    print("Generating predictions...")
    predictions = await rag.get_predictions(
        dataset,
        topk_chunks,
        generation_type="normal_cot",  # Try "scheduled_cot" or "wo_cot"
        max_tokens=400  # Override generation parameter
    )
    print(f"Predictions: {predictions}\n")
    
    # 6. Evaluate results
    print("Evaluating predictions...")
    evaluation = rag.evaluate(
        dataset,
        predictions,
        detailed_output=True
    )
    
    print("\nEvaluation Results:")
    print(f"Exact Match: {evaluation['exact_match']:.2f}%")
    print(f"Accuracy: {evaluation['acc']:.2f}%")
    print(f"F1 Score: {evaluation['f1']:.2f}%")

    os.makedirs("results", exist_ok=True)
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())