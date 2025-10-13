import torch

from consts import *
import pandas as pd
import numpy as np
from pathlib import Path
import wandb


class Hotels:
    def __init__(self, config, path_prefix="", load_vectors=True):
        print(path_prefix)
        self.hotel2reviews = {}
        self.reviews = {}
        self.hotel2score = {}
        for h in range(1, N_HOTELS + 1):
            hotel_path = f"{path_prefix}{DATA_GAME_REVIEWS_PATH}/{h}.csv"
            hotel_csv = pd.read_csv(f"{hotel_path}", header=None)
            self.hotel2reviews[h] = {}
            self.hotel2score[h] = hotel_csv[4].mean()
            for review in hotel_csv.iterrows():
                review_id = review[1][0]
                review_score = review[1][4]
                self.hotel2reviews[h][review_id] = review_score
                self.reviews[review_id] = review_score

        if load_vectors:
            self.review_vectors = pd.read_csv(f"{path_prefix}{config.FEATURES_PATH}", index_col=0)
            self.review_vectors.index = self.review_vectors.index.astype(int)
            self.review_vectors.columns = [f"EFs_{f}" for f in self.review_vectors.columns]
            if config['FEATURES_PATH'] == "data/37EFs":
                raise ValueError("there is a new version for 37EFs")

            counter = 0
            for r in self.reviews.keys():
                if r not in self.review_vectors.index:
                    counter += 1
            if counter > 0:
                print(f"There are {counter} reviews without EFs vector. We will avoid to use them during training.")

    def get_feature_options(self, hotel_id):
        return {"hotel_is_good": int(self.hotel2score[hotel_id] >= 8)}
        # return self.get_all_reviews(hotel_id).max().to_dict()

    def get_all_reviews(self, hotel_id):
        return self.review_vectors.loc[
            [h for h in self.hotel2reviews[hotel_id].keys() if h in self.review_vectors.index]]

    def get_best_review(self, hotel_id):
        best_review = -1
        best_review_score = 0
        for review, review_score in self.hotel2reviews[hotel_id].items():
            if review_score > best_review_score:
                best_review_score = review_score
                best_review = review
        return self.review_vectors.loc[best_review].to_numpy()


class StrategicSituation:
    def __init__(self, prev_round_situation: 'StrategicSituation' = None, prev_round_results=None, from_dict={}):
        if len(from_dict):
            self.round = from_dict["round"]
            self.last_didGo = from_dict["last_didGo"]
            self.last_didWin = from_dict["last_didWin"]
            self.last_last_didGo = from_dict["last_last_didGo"]
            self.last_last_didWin = from_dict["last_last_didWin"]
            self.user_points = from_dict["user_points"]
            self.bot_points = from_dict["bot_points"]
        elif prev_round_situation is None:
            self.round = 1
            self.last_didGo = None
            self.last_didWin = None
            self.last_last_didGo = None
            self.last_last_didWin = None
            self.user_points = 0
            self.bot_points = 0
        else:
            self.round = prev_round_situation.round + 1
            assert prev_round_results["didGo"] is not None
            assert prev_round_results["didWin"] is not None
            self.last_didGo = prev_round_results["didGo"]
            self.last_didWin = prev_round_results["didWin"]
            self.last_last_didGo = prev_round_situation.last_didGo
            self.last_last_didWin = prev_round_situation.last_didWin
            self.user_points = prev_round_situation.user_points + prev_round_results["didWin"]
            self.bot_points = prev_round_situation.bot_points + prev_round_results["didGo"]

    def __call__(self, *args, **kwargs):
        data =  {"roundNum": self.round,
                "user_points": self.user_points,
                "bot_points": self.bot_points,
                "last_didGo_True": self.last_didGo == True,
                "last_didGo_False": self.last_didGo == False,
                "last_didWin_True": self.last_didWin == True,
                "last_didWin_False": self.last_didWin == False,
                "last_last_didGo_True": self.last_last_didGo == True,
                "last_last_didGo_False": self.last_last_didGo == False,
                "last_last_didWin_True": self.last_last_didWin == True,
                "last_last_didWin_False": self.last_last_didWin == False,
                "user_earned_more": self.user_points > self.bot_points,
                "user_not_earned_more": not self.user_points > self.bot_points}

        if USING_REACTION_TIME:
            data.update({feature: 0 for feature in reaction_time_columns_names})

        return np.array([data[f] for f in STRATEGIC_FEATURES_ORDER])


class ModelBasedLoss:
    def __init__(self, force_training=False):
        raise NotImplemented

    def __call__(self, vector, *args, **kwargs):
        raise NotImplemented


def predict_go_proba(env_model, strategic_situation: StrategicSituation, review_vector, update_vectors, vectors=None):
    if not isinstance(review_vector, torch.Tensor):
        review_vector = torch.Tensor(review_vector)

    vec = torch.cat([torch.Tensor(strategic_situation()), review_vector.flatten()])
    if vectors:
        output = env_model.predict_proba({"x": vec,
                                          "user_vector": vectors["user_vector"],
                                          "game_vector": vectors["game_vector"]},
                                         update_vectors=update_vectors, vectors_in_input=True)
        return output["proba"].flatten()[1], output
    else:
        output = env_model.predict_proba({"x": vec},
                                         update_vectors=update_vectors)
        return output["proba"].flatten()[1]


class Metrics:
    def __init__(self, prefix=None):
        self.prefix = str(prefix) + "_" if prefix else ""
        self.current_stage = None
        self.current_epoch = 0
        self.all = {}

    def reset_epoch(self):
        self.current_epoch = 0

    def next_epoch(self):
        self.current_epoch += 1

    def set_stage(self, stage):
        self.current_stage = stage

    def write(self, metric, value):
        metric_name = self.prefix + str(self.current_stage) + "_" + str(metric)
        # print(metric_name + "_epoch" + str(self.current_epoch), value)
        self.all[metric_name] = value
        self.all[metric_name + "_epoch" + str(self.current_epoch)] = value


class ResultSaver:
    def __init__(self, config, epoch):
        self.config = config
        self.epoch = epoch
        self.ids = []
        self.user_id = []
        self.bot_strategy = []
        self.accuracy = []
        self.directory = Path('predictions', config["wandb_run_id"])

        self.directory.mkdir(parents=True, exist_ok=True)

    def add_results(self, ids, user_id, bot_strategy, accuracy):
        self.ids += [ids]
        self.accuracy += [accuracy]
        self.user_id += [user_id]
        self.bot_strategy += [bot_strategy]

    def next_epoch(self):
        data = pd.DataFrame({
            'ID': torch.cat(self.ids).numpy().astype(int),
            'User_ID': torch.cat(self.user_id).numpy().astype(int),
            'Bot_Strategy': torch.cat(self.bot_strategy).numpy().astype(int),
            'Accuracy': torch.cat(self.accuracy).numpy()
        })
        path = str(self.directory) + f"/{self.epoch}.csv"
        data.to_csv(path, index=False, header=True, float_format='%.4f')
        if self.config["save_artifacts"]:
            run_name = self.config["wandb_run_id"]
            artifact = wandb.Artifact(f"{run_name}_predictions_epoch_{self.epoch}", type="predictions")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        return data



class GumbelSigmoid(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        # Uniform noise for Gumbel trick
        uniform_noise = torch.rand_like(logits)

        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbel_noise) / self.temperature

        # Apply sigmoid to noisy logits
        y_soft = torch.sigmoid(noisy_logits)

        # Straight-through estimator for the hard sample
        y_hard = (y_soft > 0.5).float()
        y = y_hard - y_soft.detach() + y_soft
        return y


class TextFileWriter:
    def __init__(self, filename):
        self.file = open(filename, "a")

    def append_text(self, text):
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

