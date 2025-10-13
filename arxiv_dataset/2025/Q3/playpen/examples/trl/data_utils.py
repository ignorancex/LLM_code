import argparse
import json
import os.path
import random
from glob import glob
from tqdm import tqdm

from clemcore.clemgame.resources import load_json


def create_conversational_dataset_for(top_dir):
    """NOTE: This script requires interactions generated with clemcore >=2.4.0 !"""
    interactions_files = glob(f"{top_dir}/**/interactions.json", recursive=True)
    dataset_file = "results.jsonl"
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_file)
    print(f"Writing dataset file to {dataset_path} interactions")
    exceptions = set()
    with open(dataset_path, "w", encoding="utf-8") as f:
        print(f"Collecting {len(interactions_files)} interactions")
        for interactions_file in tqdm(interactions_files):
            interactions = load_json(interactions_file)
            # read from meta info (since clemcore 2.4)
            game_name = interactions["meta"]["game_name"]
            experiment_name = interactions["meta"]["experiment_name"]
            game_id = interactions["meta"]["game_id"]
            outcome = None  # this should also become part of the meta in later clemcore versions
            try:
                scores = load_json(os.path.join(os.path.dirname(interactions_file), "scores.json"))
                episodes_scores = scores["episode scores"]
                if episodes_scores["Aborted"]:
                    outcome = "aborted"
                if episodes_scores["Success"]:
                    outcome = "success"
                if episodes_scores["Lose"]:
                    outcome = "failure"
            except Exception as e:  # cannot determine outcome
                pass
            # We collect each episode from the perspective of all players individually
            for player_name, player_details in interactions["players"].items():
                try:  # since clemcore 2.4
                    player_name = player_details["player_name"]
                    game_role = player_details["game_role"]
                    model_name = player_details["model_name"]
                except Exception as e:
                    exceptions.add((game_name, player_details))
                    continue
                if player_name == "GM":
                    continue  # ignore game master perspective (we dont want to learn that here)
                if model_name == "programmatic":
                    continue  # do not train on programmatic behaviors
                # print(f"Going through {len(interactions['turns'])} rounds")
                messages = []
                for events in interactions["turns"]:
                    # print(f"Scanning {len(events)} round events")
                    for event in events:
                        if event["to"] == player_name:  # a message to the player (assistant)
                            messages.append(dict(role="user", content=event["action"]["content"]))
                        if event["from"] == player_name:  # a message from the player (assistant)
                            messages.append(dict(role="assistant", content=event["action"]["content"]))
                if messages:  # ignore episodes where player had no turn because of initial failures of the other
                    f.write(json.dumps({
                        "messages": messages,
                        "meta": {
                            "game": game_name,
                            "experiment": experiment_name,
                            "task_id": game_id,
                            "player_name": player_name,
                            "game_role": game_role,
                            "model": model_name,
                            "outcome": outcome
                        }
                    }) + '\n')
    for ex in exceptions:
        print(ex)
    counter = 0
    random_examples = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            counter += 1
            if random.randint(0, 100) < 1:
                random_examples.append(json.loads(line))
    print(f"Written {counter} examples to {dataset_path}")
    print()
    print(f"See {len(random_examples)} examples:")
    for example in random_examples:
        print("Meta:", example["meta"])
        print("Messages:")
        for message in example["messages"]:
            print(f"  {message}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("top_dir",
                        help="The directory containing benchmark results for one or more models")
    args = parser.parse_args()
    create_conversational_dataset_for(args.top_dir)


if __name__ == "__main__":
    main()
