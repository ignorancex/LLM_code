from pathlib import Path

import pandas as pd


def load_model_mapping(name: str = "alpaca-eval") -> pd.DataFrame:
    """
    Load the mapping between names of Alpaeval/Arena-Hard and Chatbot Arena.
    Note it could be a dictionary but we leave it be a dataframe as we may want to add another column (for instance
    HuggingFace model name)
    :return: a dataframe with columns "chatbot-arena", "alpaca-eval" with the different names of the models
    """
    assert name in ["alpaca-eval", "arena-hard"]
    name_mapping = pd.read_csv(Path(__file__).parent / f"model_mapping_{name}.csv")
    return name_mapping.dropna()


def chatbot_alpaca_arena_mapping():
    df_alpaca = load_model_mapping("alpaca-eval")
    df_arena = load_model_mapping("arena-hard")
    df = pd.DataFrame(
        {
            "alpaca-eval": df_alpaca.set_index("chatbot-arena")["alpaca-eval"],
            "arena-hard": df_arena.set_index("chatbot-arena")["arena-hard"],
        }
    )
    df = df.dropna(axis=0)
    return df


def common_models():
    x = load_model_mapping("alpaca-eval")
    y = load_model_mapping("arena-hard")
    res = list(sorted(set(x["chatbot-arena"]).intersection(y["chatbot-arena"])))
    # "gpt-4-0314" is not in AH, not sure why, is it because its the baseline?
    return [x for x in res if x not in ["gpt-4-0314"]]


if __name__ == "__main__":
    print(load_model_mapping())
    print(chatbot_alpaca_arena_mapping())
    print(len(common_models()))
