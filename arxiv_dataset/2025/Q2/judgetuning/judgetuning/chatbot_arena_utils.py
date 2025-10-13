import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression

from judgetuning.judge_paths import judge_tuning_datadir

# url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240422-2.json"
url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
chabot_arena_battles_file = judge_tuning_datadir / "local_file_name.json"
chabot_arena_battles_file.parent.mkdir(parents=True, exist_ok=True)


def download_chatbot_arena_battles():
    if not chabot_arena_battles_file.exists():
        print("download data from chatbot arena")
        response = requests.get(url)
        with open("local_file_name.json", "wb") as file:
            file.write(response.content)
        Path("local_file_name.json").rename(chabot_arena_battles_file)


def load_battles():
    download_chatbot_arena_battles()

    # load the JSON data from the local file
    with open(chabot_arena_battles_file, "r") as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    battles = battles[battles["anony"] == True]
    return battles


def compute_mle_elo(
    df,
    SCALE=400,
    BASE=10,
    INIT_RATING=1000,
    baseline_model: str | None = None,
    force_two_classes: bool = True,
) -> pd.Series:
    """
    :param df: a dataframe with battles between models with columns ["model_a", "model_b", "winner"], the values of
    "winner" columns must be in ["tie", "tie (bothbad)", "model_a", "model_b"]
    :param SCALE:
    :param BASE:
    :param INIT_RATING:
    :param baseline_model:
    :return:
    """
    for col in ["model_a", "model_b", "winner"]:
        assert col in df.columns
    assert {"tie", "tie (bothbad)", "model_a", "model_b"}.issuperset(df.winner.unique())
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    if force_two_classes and len(set(Y)) == 1:
        Y[0] = 1 - Y[0]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor for baseline (gpt-4-0314) = 1000
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index)


def load_chatbot_arena_elo_ratings() -> pd.Series:
    path = judge_tuning_datadir / "chatbot_arena_elo_rating.csv"
    if path.exists():
        # print(f"Loading cache from {path}")
        return pd.read_csv(path).set_index("model_name").loc[:, "elo_rating"]
    else:
        battles = load_battles()

        print("compute elo ratings")
        elo_mle_ratings = compute_mle_elo(battles)

        print(f"computed elo ratings, saving to file: {path}")
        print(elo_mle_ratings.to_string())
        elo_mle_ratings = elo_mle_ratings.reset_index()
        elo_mle_ratings = elo_mle_ratings.rename(
            {"index": "model_name", 0: "elo_rating"}, axis=1
        )
        elo_mle_ratings.to_csv(path, index=False)
        return elo_mle_ratings.set_index("model_name").loc[:, "elo_rating"]


def battle_matrix(model_metrics: pd.Series):
    n_models = len(model_metrics)
    model_metrics = model_metrics.reset_index(drop=True)
    mat_battles = np.full([n_models, n_models], np.nan)
    model_indices = [i for i in range(n_models) if not np.isnan(model_metrics.loc[i])]
    for i in model_indices:
        for j in model_indices:
            if i != j:
                i_better_j = model_metrics.loc[i] > model_metrics.loc[j]
                mat_battles[i][j] = i_better_j
                mat_battles[j][i] = not i_better_j
    return mat_battles


if __name__ == "__main__":
    battles = load_battles()
    df_elo = compute_mle_elo(battles)
    print(df_elo.sort_values(ascending=False).to_string())
    # df_elo.head()
    df_elo2 = compute_mle_elo(battles.sample(1000))
    model_series = pd.Series(
        [1000, 800, 1200, np.nan, 1700], index=["a", "b", "c", "d", "e"]
    )
    print(battle_matrix(model_series))
