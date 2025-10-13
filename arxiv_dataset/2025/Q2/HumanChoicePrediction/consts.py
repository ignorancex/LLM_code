import torch
import pandas as pd
import numpy as np

DATA_GAME_REVIEWS_PATH = "data/game_reviews"

DATA_CLEAN_ACTION_PATH_X = "data/games_clean_X.csv"
DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS = 210
DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS = 35
DATA_CLEAN_ACTION_PATH_Y = "data/games_clean_Y.csv"
OFFLINE_SIM_DATA_PATH = "data/LLM_games_personas.csv"

REVIEW_ENCODERS_PATH = "models/reviews_encoders"
REVIEW_VECTORS_PATH = "data/reviews_vectors"

USING_REACTION_TIME = True
reaction_time_bins = [(0, 400),
                      (400, 800),
                      (800, 1200),
                      (1200, 1600),
                      (1600, 2500),
                      (2500, 4000),
                      (4000, 6000),
                      (6000, 12000),
                      (12000, 20000),
                      (20000, np.inf)]
reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in reaction_time_bins]

STRATEGIC_FEATURES_ORDER = ['roundNum', 'user_points', 'bot_points',
                            'last_didGo_True', 'last_didGo_False',
                            'last_didWin_True', 'last_didWin_False',
                            'last_last_didGo_True', 'last_last_didGo_False',
                            'last_last_didWin_True', 'last_last_didWin_False',
                            "user_earned_more", "user_not_earned_more"]
if USING_REACTION_TIME:
    STRATEGIC_FEATURES_ORDER += reaction_time_columns_names

N_HOTELS = 1068

STRATEGY_DIM = len(STRATEGIC_FEATURES_ORDER)

DEEPRL_LEARNING_RATE = 4e-4

DATA_ROUNDS_PER_GAME = 10
SIMULATION_BATCH_SIZE = 4
ENV_BATCH_SIZE = 4

SIMULATION_MAX_ACTIVE_USERS = 2000
SIMULATION_TH = 9

DATA_N_BOTS = 1179

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

DATA_BLANK_ROW_DF = lambda s: pd.DataFrame.from_dict({"user_id": [-100],
                                                      "strategy_id": [s],
                                                      "gameId": [-100],
                                                      "roundNum": [-1],
                                                      "hotelId": [-1],
                                                      "reviewId": [-1],
                                                      "hotelScore": [-1],
                                                      "didGo": [-100],
                                                      "didWin": [-100],
                                                      "correctAnswers": [-1],
                                                      "reaction_time": [1],
                                                      "review_positive": [""],
                                                      "review_negative": [""],
                                                      "last_didGo_True": [0],
                                                      "last_didWin_True": [0],
                                                      "last_didGo_False": [0],
                                                      "last_didWin_False": [0],
                                                      "last_last_didGo_True": [0],
                                                      "last_last_didWin_True": [0],
                                                      "last_last_didGo_False": [0],
                                                      "last_last_didWin_False": [0],
                                                      "last_reaction_time": [-1],
                                                      "user_points": [-1],
                                                      "bot_points": [-1],
                                                      "is_sample": [False],
                                                      "weight": 0,
                                                      "action_id": [-1]})

bot2strategy_X = {0: 3, 1: 0, 2: 2, 3: 5, 4: 59, 5: 19}
bot2strategy_Y = {0: 132, 1: 23, 2: 107, 3: 43, 4: 17, 5: 93}

bot_thresholds_X = {0: 10, 1: 7, 2: 9, 3: 8, 4: 8, 5: 9}
bot_thresholds_Y = {0: 10, 1: 9, 2: 9, 3: 9, 4: 9, 5: 9}

AGENT_LEARNING_TH = 8
