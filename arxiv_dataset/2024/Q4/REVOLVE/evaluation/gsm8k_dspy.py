import dspy
import argparse
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
import numpy as np
import random
import pandas as pd
import json
import os
import torch
from dotenv import load_dotenv
load_dotenv(override=True)


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="The identifier of the model to use or path to the model if it is a local model.")
    return parser.parse_args()


args = config()

# Set up the LM.
turbo = dspy.HFModel(model=args.model, max_tokens=2000, top_p=0.99, temperature=1e-6)
dspy.settings.configure(lm=turbo)

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset, gsm8k_testset = gsm8k.train, gsm8k.dev, gsm8k.test
indices = np.arange(len(gsm8k_trainset))
np.random.seed(42)
random.seed(42)
np.random.shuffle(indices)
gsm8k_trainset = [gsm8k_trainset[i] for i in indices[:36]]


class StudentCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


class TeacherCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
        self.lm = dspy.AzureOpenAI(model='gpt4o', max_tokens=2000, top_p=0.99,
                                   api_base=os.getenv('AZURE_OPENAI_API_BASE'),
                                   api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                                   api_key=os.getenv('AZURE_OPENAI_API_KEY')
                                   )

    def forward(self, question):
        return self.prog(question=question, lm=self.lm)


# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=8, num_candidate_programs=10)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(StudentCoT(), trainset=gsm8k_trainset, valset=gsm8k_devset, teacher=TeacherCoT())

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_testset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
overall_metric, results, individual_scores = evaluate(
    optimized_cot,
    return_all_scores=True,
    return_outputs=True
)

# Save the results.
try:
    with open("./results/results_GSM8K_DSPy_dspy_overall_metric.txt", "w") as f:
        f.write(f"Overall Metric: {overall_metric}\n")

    results_df = pd.DataFrame([
        {
            'example': result[0],
            'prediction': result[1],
            'score': result[2]
        }
        for result in results
    ])

    results_df.to_csv("./results/results_GSM8K_DSPy_dspy_evaluation_results.csv", index=False)

    with open("./results/results_GSM8K_DSPy_dspy_individual_scores.json", "w") as f:
        json.dump(individual_scores, f)

    pd.DataFrame(individual_scores, columns=["score"]).to_csv("./results/results_GSM8K_DSPy_dspy_individual_scores.csv", index=False)

    print("Evaluation results saved successfully.")
except Exception as e:
    print(f"Failed to save evaluation results: {e}")
