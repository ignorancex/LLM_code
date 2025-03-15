import os
import sys
import random
import json

def get_json_files(directory):
    """Get a list of JSON files in the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def sample_condition(item):
    """Check if the item meets the sampling condition."""
    return len(item['good_questions']) > 1

def load_and_sample_data(directory, condition):
    """Load data from JSON files in the directory and sample items based on the condition."""
    sampled_data = []
    files = get_json_files(directory)
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            data = json.load(f)
            for item in data:
                if condition(item):
                    sampled_data.append(item)
    return sampled_data

def save_sampled_data(data, filepath):
    """Save the sampled data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    base_dir = os.path.join('enhanced', 'gpt-4o-mini')
    sampled_data = load_and_sample_data(base_dir, sample_condition)
    
    # Sample 100 items, seed = 42
    random.seed(42)
    sampled_data = random.sample(sampled_data, 100)
    
    # Remove 'good_questions' from each item
    sampled_data = [{k: v for k, v in item.items() if k != 'good_questions'} for item in sampled_data]
    
    # Save the sampled data
    save_sampled_data(sampled_data, 'data/sampled_data.json')

if __name__ == "__main__":
    main()

