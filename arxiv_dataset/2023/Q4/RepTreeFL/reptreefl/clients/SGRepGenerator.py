import os.path as osp
import numpy
import torch
import warnings
import copy
from torch_geometric.data import DataLoader
import copy
import random
from models.generators import Generator1, Generator2
from utils.preprocess import*
import constants
from utils.utils import *
from datasets.dataset_brain_connectomes import MultiResolutionBrainConnectomeDataset

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on ")

random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

class SGRepGenerator():
    def __init__(self, resolution, run_index, client_index, is_replica=False, replica_index=0, replica_depth=0, fed_alg="FedAvg", training=True):
        self.generator_save_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights/weight_run{run_index}_client{client_index}_replica{replica_index}_generator.model'
        self.resolution = resolution
        self.is_replica = is_replica

        self.replicas_list = []
        if fed_alg == "FedRep" and training:
            if replica_depth < constants.MAX_REPLICA_DEPTH:
                for new_replica_index in range(constants.NUMBER_REPLICAS):
                    self.replicas_list.append(SGRepGenerator(resolution=resolution, run_index=run_index, client_index=client_index, is_replica=True, 
                                                            replica_index=new_replica_index, replica_depth=replica_depth + 1, 
                                                            fed_alg=fed_alg))

        self.client_index = client_index
        self.replica_index = replica_index
        self.replica_depth = replica_depth

        self.fed_alg = fed_alg

        self.best_train_loss = float('inf')
        self.source_test = None
        self.best_predicted_test = None
        self.best_target = None

        self.mae = None

        if self.resolution == 160:
            self.generator = Generator1()
        elif self.resolution == 268:
            self.generator = Generator2()

        self.replica_aggregated_model = copy.deepcopy(self.generator)

        self.generator.to(device)
        self.replica_aggregated_model.to(device)

        self.generator_optimizer = torch.optim.SGD(self.generator.parameters(), lr=0.1)

        self.global_model_param = None

        old_grad = copy.deepcopy(self.generator)
        for param in old_grad.parameters():
            param.data = torch.zeros_like(param.data)
        self.old_grad = get_mdl_params_resnet([old_grad])[0]

        self.aggregating_layers = ["conv1.bias", "conv1.nn.0.weight", "conv1.nn.0.bias", "conv1.lin.weight", "conv1.root",
                                        "conv2.bias", "conv2.nn.0.weight", "conv2.nn.0.bias", "conv2.lin.weight", "conv2.root"]

    def train(self):
        self.generator.train()

        global_loss_train = []
        l1_loss = torch.nn.L1Loss()
        l1_loss.to(device)

        for epochs in range(constants.NUM_LOCAL_EPOCHS):
            global_loss_train_current_epoch = []

            with torch.autograd.set_detect_anomaly(True):
                # for data_source, data_target in zip(X_casted_train_source, X_casted_train_target):
                for batch_idx, batch_samples in enumerate(self.data_loader):
                    data_source, data_target, target_edge_attributes = batch_samples['source'], batch_samples['target'], batch_samples['target_x']

                    self.generator_optimizer.zero_grad()

                    data_source = data_source.to(device)
                    G_output = self.generator(data_source)  # 35 x 35
                    torch.cuda.empty_cache()

                    target_edge_attributes = target_edge_attributes.to(device)
                    global_loss = l1_loss(target_edge_attributes, G_output)
                    torch.cuda.empty_cache()
                    
                    global_loss.backward()
                    self.generator_optimizer.step()
                    torch.cuda.empty_cache()

                    global_loss_train_current_epoch.append(global_loss.detach().cpu().numpy())
                    torch.cuda.empty_cache()

                global_loss_train.append(np.mean(global_loss_train_current_epoch))

        torch.cuda.empty_cache()

        return global_loss_train

    def test(self, X_test_source, X_test_target):
        X_casted_test_source = cast_data_vector_RH(X_test_source)
        if self.resolution == 160:
            X_casted_test_target = cast_data_vector_FC(X_test_target)
        elif self.resolution == 268:
            X_casted_test_target = cast_data_vector_HHR(X_test_target)

        dataset = MultiResolutionBrainConnectomeDataset(X_casted_test_source, X_casted_test_target, self.resolution)
        data_loader = DataLoader(dataset, batch_size=1)

        self.generator.load_state_dict(torch.load(self.generator_save_path), strict=False)
        self.generator.eval()

        mae_test = []

        for batch_idx, batch_samples in enumerate(data_loader):
            data_source, data_target, target_x, data_source_test = batch_samples['source'], batch_samples['target'], batch_samples['target_x'], batch_samples['source_x']

            data_source = data_source.to(device)
            G_output_test = self.generator(data_source)

            torch.cuda.empty_cache()

            source_test = data_source_test.detach().cpu().clone().numpy()
            predicted_test = G_output_test.detach().cpu().clone().numpy()

            torch.cuda.empty_cache()
            target_x = target_x.to(device)
            G_output_test = G_output_test.to(device)

            data_target = target_x.detach().cpu().clone().numpy()
            residual_error = np.abs(data_target - predicted_test)
            residual_error_mean = np.mean(residual_error)
            mae_test.append(residual_error_mean)

        mean_mae_test = np.mean(mae_test)

        self.source_test = source_test
        self.best_predicted_test = predicted_test
        self.best_target = data_target

        self.mae_test = mean_mae_test

        return mean_mae_test
    
    def run_one_round(self):
        global_loss_train = self.train()
        self.no_epochs_trained += constants.NUM_LOCAL_EPOCHS

        print(f'Epochs trained: {self.no_epochs_trained}')

        # Run one round replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].run_one_round()

        # Aggregate the replicas
        if constants.AGGREGATION == "diversity" and len(self.replicas_list) > 0:
            self.aggregate_diversity()

            # Update the new model parameters
            model_state = self.generator.state_dict()

            for name, param in self.generator.named_parameters():
                if name in self.aggregating_layers:
                    model_updates = [copy.deepcopy(self.generator.state_dict()[name].data), copy.deepcopy(self.replica_aggregated_model.state_dict()[name].data)]
                    averaged_update = torch.stack(model_updates).mean(dim=0)
                    model_state[name] = averaged_update

            self.generator.load_state_dict(model_state)
        else:
            self.aggregate_replicas()

        return global_loss_train
    
    def aggregate_replicas(self):
        model_state = self.generator.state_dict()

        weight = 1. / (1 + constants.NUMBER_REPLICAS)
        for name, param in self.generator.named_parameters():
            if name in self.aggregating_layers:
                replicas_updates = [weight * copy.deepcopy(replica.get_replica_aggregated_model().state_dict()[name].data) for replica in self.replicas_list]
                replicas_updates.append(weight * copy.deepcopy(self.generator.state_dict()[name].data))
                averaged_update = torch.stack(replicas_updates).mean(dim=0)
                model_state[name] = averaged_update

        # Update the replica aggregated model with the aggregated parameters
        self.generator.load_state_dict(model_state)

    def aggregate_diversity(self):
        normalized_weights = []
        if constants.DIV_METRIC == "l2":
            l2_norms_replicas = [l2_norm_models(replica.get_generator(), self.model) for replica in self.replicas_list]
            normalized_weights = compute_weight_from_l2_norm(l2_norms_replicas)
        elif constants.DIV_METRIC == "kl":
            kl_div_replicas = [kl_divergence_models(replica.get_generator(), self.model) for replica in self.replicas_list]
            normalized_weights = compute_weights_from_kl(kl_div_replicas)

        # Aggregate model parameters based on the weights
        model_state = self.generator.state_dict()
        for name, param in self.generator.named_parameters():
            if name in self.aggregating_layers:
                client_updates = [normalized_weights[i] * copy.deepcopy(replica.get_generator().state_dict()[name].data) for i, replica in enumerate(self.replicas_list)]
                averaged_update = torch.stack(client_updates).sum(dim=0)
                model_state[name] = averaged_update

        # Update the replica aggregated model with the aggregated parameters
        self.replica_aggregated_model.load_state_dict(model_state)

    def save_model(self):
        torch.save(self.generator.state_dict(), self.generator_save_path)

    def set_generator_saving_path(self, new_path):
        self.generator_save_path = new_path
    
    def get_source_target_prediction(self):
        return self.source_test, self.best_predicted_test, self.best_target
    
    def get_mae(self):
        return self.mae_test
    
    def get_generator(self):
        return self.generator
    
    def get_replica_aggregated_model(self):
        return self.replica_aggregated_model
    
    def get_replicas_list(self):
        return self.replicas_list

    def get_old_grad(self):
        return self.old_grad
    
    def initialize_model_parameters(self, global_model_state_dict):
        self.generator.load_state_dict(global_model_state_dict)
        self.replica_aggregated_model.load_state_dict(global_model_state_dict)

        # Set training dataset for replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].initialize_model_parameters(global_model_state_dict)

    def update_parameters(self, global_model):
        generator_new_state_dict = copy.deepcopy(self.generator.state_dict())
        for param_name in self.generator.state_dict():
            if param_name in self.aggregating_layers:
                generator_new_state_dict[param_name].data = copy.deepcopy(global_model.state_dict()[param_name])
                
        self.generator.load_state_dict(generator_new_state_dict)

        for params in self.generator.parameters():
            params.requires_grad = True

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].update_parameters(global_model)

    def set_global_model_params(self, global_model_param):
        self.global_model_param = global_model_param

        # Set global model params for replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].set_global_model_params(global_model_param)
    
    def set_training_data(self, X_train_source, X_train_target, offset_perturbed_samples = 0):
        X_train_source = copy.deepcopy(X_train_source)
        X_train_target = copy.deepcopy(X_train_target)

        if self.is_replica:
            if offset_perturbed_samples >= len(X_train_source):
                offset_perturbed_samples = offset_perturbed_samples % len(X_train_source)

            replica_indices_1 = [i for i in range(0, offset_perturbed_samples)]
            replica_indices_2 = [i for i in range(offset_perturbed_samples + constants.NUMBER_PERTURBED_SAMPLES, len(X_train_source))]
            replica_indices = replica_indices_1 + replica_indices_2

            X_train_source = X_train_source[replica_indices]
            X_train_target = X_train_target[replica_indices]
        
        X_casted_train_source = cast_data_vector_RH(X_train_source)
        if self.resolution == 160:
            X_casted_train_target = cast_data_vector_FC(X_train_target)
        elif self.resolution == 268:
            X_casted_train_target = cast_data_vector_HHR(X_train_target)

        dataset = MultiResolutionBrainConnectomeDataset(X_casted_train_source, X_casted_train_target, self.resolution)
        self.data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE)

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                no_perturbed_samples_next_depth = int((constants.NUMBER_PERTURBED_SAMPLES / 100.0) * len(X_train_source))
                self.replicas_list[replica].set_training_data(X_train_source, X_train_target, no_perturbed_samples = no_perturbed_samples_next_depth, offset_perturbed_samples = replica * no_perturbed_samples_next_depth)

    def get_next_data_loader_batch(self):
        return next(iter(self.data_loader))