import json

from eval.utils import load_runs

version, run_id = [
    ("mlp", 46),
    ("vit", 54),
    ("vit", 71),
    ("llm", 14),
][2]

[run] = load_runs("andylolu2", f"flash-dropout-{version}", [run_id])

with open(f"logs/{version}_{run_id}_config.json", "w") as f:
    json.dump(run.config, f, indent=4)
