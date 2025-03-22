import os, sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mod.mod import MoD
from mod.config import read_config
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(os.environ["HF_AUTH_TOKEN"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save the merged model"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    model = MoD(config)
    model.merge(args.out_path)
