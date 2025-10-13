from openai import OpenAI
from generator import PromptGenerator
import fire
import os


def main(subject, dataset_dir, num_prompts=200):

    assert(dataset_dir is not None)
    assert(subject is not None)
    output_path = f"{dataset_dir}/{subject}_prompts.csv"
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generator = PromptGenerator(openai_client)
    generator.generate_via_loop(subject=subject, num_prompts=num_prompts, step_size=20, enable_tqdm=True, outputfile_path=output_path)

if __name__ == '__main__':
    fire.Fire(main)