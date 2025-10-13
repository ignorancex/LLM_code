from pathlib import Path

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from judgetuning.annotation_dataset import ArenaHardDataset, AlpacaEvalDataset
from judgetuning.chatbot_arena_utils import load_battles


def model_sim(model1, model2):
    return SequenceMatcher(None, model1.lower(), model2.lower()).ratio()


def most_similar(models, query):
    sims = [model_sim(x, query) for x in models]
    index = np.argmax(sims)
    return index, sims[index]


if __name__ == "__main__":
    battles = load_battles()
    cb_arena_models = list(set(battles.model_a).union(battles.model_b))

    ds = ArenaHardDataset(
        load_annotations=False,
        rename_chatbot_arena=False,
        keep_only_chatbot_arena=False,
    )
    # ds = AlpacaEvalInstructions()

    models = ds.models
    ratios = []
    for x in models:
        most_similar_index, sim = most_similar(cb_arena_models, x)
        ratios.append((x, cb_arena_models[most_similar_index], sim))
    ratios = sorted(ratios, key=lambda x: x[-1])
    df_similarity = pd.DataFrame(
        ratios, columns=[ds.name, "chatbot-arena", "similarity"]
    )
    print(df_similarity.to_string(index=False))
    df_similarity = df_similarity[
        ~df_similarity[ds.name].str.contains(r"verbose|concise", regex=True)
    ]
    df_similarity = df_similarity.sort_values(by="chatbot-arena")
    df_similarity.loc[:, ["chatbot-arena", ds.name]].to_csv(
        Path(__file__).parent / f"model_mapping_{ds.name}_prefilter.csv", index=False
    )
