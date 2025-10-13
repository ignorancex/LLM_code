from utils.datasets import OfflineDataSet, OnlineSimulationDataSet, ConcatDatasets
from utils.samplers import NewUserBatchSampler, SimulationSampler
from consts import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from utils.usersvectors import UsersVectors
import torch.optim as optim
from itertools import chain
import wandb
from utils import *
import pickle
from utils import personas


class Environment:
    def __init__(self, model_path, config):
        self.training_mode = True
        self.model = None
        self.hidden_dim = config["hidden_dim"]
        self.use_user_vector = config["use_user_vector"]
        self.n_layers = config["layers"]
        self.env_learning_rate = config["ENV_LEARNING_RATE"]
        self.model_path = model_path
        self.init_model_arc(config=config)
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.currentDM = None
        self.currentGame = None
        self.config = config
        force_train = config["force_train"]
        if not force_train:
            try:
                self.load()
            except FileNotFoundError:
                print(f"No saved model found at {model_path}. Training a new model.")
                force_train = True
        if force_train:
            self.train()
            self.save()

    def __call__(self, *args, **kwargs):
        raise NotImplemented

    def train(self, do_eval=True):
        print("Start training the environment...")
        online_sim_type = self.config["online_sim_type"]
        assert online_sim_type in ["None", "mixed", "before_epoch", "init"]
        phases = []

        if online_sim_type == "init":
            raise NotImplementedError("The 'init' simulation type is not implemented yet.")
        
        elif self.config["task"] == "on_policy":
            human_train_size = self.config["human_train_size"]
            test_size = ON_POLICY_TEST_SIZE
            real_users = np.random.choice(range(test_size, DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS), human_train_size, replace=False)
            llm_real_train = real_users[:human_train_size]
            llm_real_test = np.arange(test_size)

        if self.config["human_train_size"] != 0:
            # if self.config["human_train_size"] != -1 and self.config["human_train_size"] != "all":
            #     real_users = np.random.choice(range(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS), self.config["human_train_size"], replace=False)
            # else:
            #     real_users = np.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
            if self.config["ENV_HPT_mode"]:
                all_users = np.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
                train_users = np.random.choice(all_users,
                                               int(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS * 0.8), replace=False)
                test_users = np.setdiff1d(all_users, train_users)
                train_dataset = OfflineDataSet(user_groups="X", strategies=[3, 0, 2, 5], users=train_users,
                                               weight_type=self.config.loss_weight_type, config=self.config)
            else:
                train_dataset = OfflineDataSet(user_groups="X", weight_type=self.config.loss_weight_type,
                                               config=self.config)

            train_sampler = NewUserBatchSampler(train_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
            phases += [("Train", train_dataloader)]
        
        if self.config["offline_simulation_size"] != 0:
            if self.config.personas_group_number == -1:
                llm_users_options = range(TOTAL_LLM_USERS)
                llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]), replace=False)
            else:
                groups = personas.get_personas_in_group(self.config.personas_group_number)
                personas_df = pd.read_csv(self.config["OFFLINE_SIM_DATA_PATH"])
                if self.config["personas_balanced"]:
                    group_size = int(self.config["offline_simulation_size"]) // len(groups)
                    llm_users = []
                    for group in groups:
                        llm_users_options = personas_df[personas_df["persona"] == group]["user_id"].unique()
                        persona_users = np.random.choice(llm_users_options, group_size, replace=False)
                        llm_users += [persona_users]
                    llm_users = np.concatenate(llm_users)
                else:
                    llm_users_options = personas_df[personas_df["persona"].isin(groups)]["user_id"].unique()
                    llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]), replace=False)
            offline_dataset = OfflineDataSet(user_groups="L", users=llm_users, config=self.config,
                                             weight_type=self.config.loss_weight_type,
                                             strategies=self.config.strategies)
            offline_sim_sampler = NewUserBatchSampler(offline_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            offline_sim_dataloader = DataLoader(offline_dataset, batch_sampler=offline_sim_sampler, shuffle=False)
            phases.insert(0, ("Offline Simulation", offline_sim_dataloader))

        if do_eval:
            if self.config["ENV_HPT_mode"]:
                test_dataset = OfflineDataSet(user_groups="X", users=test_users, strategies=[19, 59], weight_type=self.config.loss_weight_type, config=self.config)
            elif self.config["task"] == "off_policy":
                test_dataset = OfflineDataSet(user_groups="Y", strategies=self.config.strategies,
                                              weight_type="sender_receiver", config=self.config)
            else:
                assert self.config["task"] == "on_policy"
                test_dataset = OfflineDataSet(user_groups="X", users=llm_real_test, weight_type="sender_receiver", config=self.config,
                                              strategies=self.config.strategies)

            test_sampler = NewUserBatchSampler(test_dataset, batch_size=ENV_BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, shuffle=False)
            phases += [("Test", test_dataloader)]

        if self.config["online_simulation_size"] > 0 and online_sim_type == "before_epoch":
            phases.insert(0, ("Online Simulation", "sim_dataloader"))

        self.model.to(device)
        optimizer = torch.optim.Adam([p for p in chain(self.model.parameters()) if p.requires_grad],
                                     lr=self.env_learning_rate)
        self.set_train_mode()
        metrics = Metrics("ENV")
        for epoch in range(self.config["total_epochs"]):
            result_saver = ResultSaver(config=self.config, epoch=epoch)
            print("#" * 16)
            print(f"# Epoch {epoch}")
            print("#" * 16)
            if self.config["online_simulation_size"] > 0 and online_sim_type in ["before_epoch", "mixed"]:
                online_simulation_dataset = OnlineSimulationDataSet(config=self.config)
                online_simulation_sampler = SimulationSampler(online_simulation_dataset, SIMULATION_BATCH_SIZE)
                online_simulation_dataloader = DataLoader(online_simulation_dataset,
                                                          batch_sampler=online_simulation_sampler, shuffle=False)
            for phase, dataloader in phases:
                # print(phase)
                metrics.set_stage(phase)
                if phase == "Online Simulation" and online_sim_type == "before_epoch":
                    dataloader = online_simulation_dataloader
                if self.use_user_vector:
                    self.model.user_vectors.delete_all_users()
                    self.model.game_vectors.delete_all_users()
                total_loss = 0
                total_proba_to_right_action = 0
                total_proba_weighted = 0
                total_right_action = 0
                total_right_action_weighted = 0
                total_weight = 0
                n_actions = 0
                if phase != "Test":
                    self.set_train_mode()
                    if online_sim_type == "mixed":
                        dataloader = ConcatDatasets(dataloader, online_simulation_dataloader)
                else:
                    self.set_eval_mode()
                for batch in tqdm(dataloader, desc=phase):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    batch_size, _ = batch["hotels_scores"].shape
                    review_vector = batch["review_vector"].reshape(batch_size, DATA_ROUNDS_PER_GAME, -1)
                    env_input = [batch[feature].unsqueeze(-1) for feature in STRATEGIC_FEATURES_ORDER]
                    env_input += [review_vector]
                    env_input = torch.cat(env_input, dim=2).double().to(device)
                    model_vectors = {"x": env_input}

                    if self.use_user_vector:
                        model_vectors["user_vector"] = self.model.user_vectors[batch["user_id"].to("cpu").numpy()].to(
                            device)
                        model_vectors["game_vector"] = self.model.game_vectors[batch["user_id"].to("cpu").numpy()].to(
                            device)
                    if phase != "Test":
                        model_output = self.model(model_vectors)
                    else:
                        with torch.no_grad():
                            model_output = self.model(model_vectors)
                    output = model_output["output"]
                    mask = (batch["action_taken"] != -100).flatten()
                    relevant_predictions = output.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1)[mask]
                    relevant_ground_truth = batch["action_taken"].flatten()[mask]
                    relevant_weight = batch["weight"][batch["is_sample"]]

                    proba_to_right_action = torch.exp(relevant_predictions[torch.arange(len(relevant_predictions), device=device),
                                                      relevant_ground_truth])
                    total_proba_to_right_action += proba_to_right_action.sum().item()
                    total_proba_weighted += (proba_to_right_action * relevant_weight).sum().item()
                    total_right_action += (proba_to_right_action >= 0.5).sum().item()
                    total_right_action_weighted += ((proba_to_right_action >= 0.5) * relevant_weight).sum().item()
                    n_actions += len(proba_to_right_action)
                    target = batch["action_taken"].reshape(-1)[batch["is_sample"].reshape(-1)]
                    total_weight += batch["weight"][batch["is_sample"]].sum().item()
                    loss = (self.loss_fn(relevant_predictions, relevant_ground_truth) * relevant_weight
                            ).mean()
                    total_loss += loss.item()
                    if phase != "Test":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        result_saver.add_results(ids=batch["action_id"].flatten()[mask].cpu(),
                                                 user_id=torch.repeat_interleave(batch["user_id"],
                                                                                 batch["bot_strategy"].shape[-1])[mask].cpu(),
                                                 bot_strategy=batch["bot_strategy"].flatten()[mask].cpu(),
                                                 accuracy=proba_to_right_action.cpu())

                    if self.use_user_vector:
                        updated_user_vectors = model_output["user_vector"].to("cpu").detach()
                        self.model.user_vectors[batch["user_id"].to("cpu").numpy()] = updated_user_vectors.squeeze()
                        updated_game_vectors = model_output["game_vector"].to("cpu").detach()
                        self.model.game_vectors[batch["user_id"].to("cpu").numpy()] = updated_game_vectors.squeeze()

                metrics.write("TotalLoss", total_loss)
                if n_actions:
                    metrics.write("Right action", total_right_action / n_actions)
                    metrics.write("Probability to choose the right action", total_proba_to_right_action / n_actions)
                if total_weight:
                    metrics.write("Weighted right action", total_right_action_weighted / total_weight)
                    metrics.write("Weighted probability to choose the right action:", total_proba_weighted / total_weight)
                if phase == "Test":
                    results_df = result_saver.next_epoch()
                    for prefix in ["proba_", ""]:
                        if "proba_" not in prefix:
                            results_df["Accuracy"] = results_df["Accuracy"] > 0.5
                        accuracy = results_df["Accuracy"].mean()
                        metrics.write(prefix+"accuracy", accuracy)
                        for strategy in results_df["Bot_Strategy"].unique():
                            bot_accuracy = results_df[results_df["Bot_Strategy"] == strategy]["Accuracy"].mean()
                            metrics.write(prefix+f"accuracy_strategy_{strategy}", bot_accuracy)
                        accuracy_per_mean_strategy = results_df.groupby("Bot_Strategy").mean()["Accuracy"].mean()
                        metrics.write(prefix+f"accuracy_per_mean_strategy", accuracy_per_mean_strategy)
                        accuracy_per_mean_user = results_df.groupby("User_ID").mean()["Accuracy"].mean()
                        metrics.write(prefix+"accuracy_per_mean_user", accuracy_per_mean_user)
                        accuracy_per_mean_user_and_bot = results_df.groupby(["User_ID", "Bot_Strategy"]).mean()["Accuracy"].mean()
                        metrics.write(prefix+"accuracy_per_mean_user_and_bot", accuracy_per_mean_user_and_bot)
                        print(prefix+"accuracy_per_mean_user_and_bot: ", accuracy_per_mean_user_and_bot)
                wandb.log(metrics.all)
            metrics.next_epoch()
        self.model.to("cpu")
        self.set_eval_mode()

    def save(self):
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def init_model_arc(self, config):
        raise NotImplemented

    def init_user_vector(self):
        raise NotImplemented

    def init_game_vector(self):
        raise NotImplemented

    def get_curr_vectors(self):
        raise NotImplemented

    def set_train_mode(self, do_training=True):
        self.training_mode = do_training
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def set_eval_mode(self, do_training=False):
        for param in self.model.parameters():
            param.requires_grad = False
        self.training_mode = do_training
        self.model.eval()
