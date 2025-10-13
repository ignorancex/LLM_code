from pathlib import Path

import numpy as np
import pandas as pd

from judgetuning.judge_paths import judge_tuning_datadir
from judgetuning.utils import download_hf

# 1) download table from huggingface
# 2) compute bootstrap human-agreement
# 3) display human-agreement in latex table


def load_dataframe(name: str):
    path = judge_tuning_datadir / "annotations-lmsys" / name
    if not path.exists():
        print(f"Local folder {path} not found, downloading.")
        download_hf(
            name=f"annotations-lmsys/{name}",
            local_path=judge_tuning_datadir,
        )
    else:
        print(f"Local folder {path} found, loading eval data from there.")
    return pd.read_csv(path)


path = Path(__file__).parent.parent / "notebooks" / "annotations-lmsys"
names = [
    "random",
    "Length",
    "PandaLM",
    "JudgeLM",
    "Arena-Hard",
    "Ours-tiny",
    "Ours-small",
    "Ours-medium",
    "Ours-large",
]
df_lmsys_per_name = {name: load_dataframe(name + ".csv.zip") for name in names}
baseline = df_lmsys_per_name["Length"]
instructions = baseline.instruction_index
human_pref = baseline.human_preference

for name in df_lmsys_per_name.keys():
    df_name = df_lmsys_per_name[name]
    pref = (
        df_name.groupby("instruction_index")["preference"]
        .mean()
        .loc[instructions]
        .values
    )
    pref[pref < 0.5] = 0
    pref[pref > 0.5] = 1
    scores = []
    for _ in range(100):
        indices = np.random.choice(
            np.arange(len(instructions)), len(instructions), replace=True
        )
        scores.append((pref[indices] == human_pref[indices]).mean())
    print(f"{name.capitalize()} & {np.mean(scores):.2f} +/- {np.std(scores):.2f} \\\\")
