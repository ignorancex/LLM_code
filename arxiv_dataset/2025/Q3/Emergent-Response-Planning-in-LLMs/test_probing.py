import torch
import yaml
import numpy as np
from tqdm import tqdm
import numpy as np
import argparse
import copy

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

from torch.cuda.amp import GradScaler, autocast

from data_utils import DatasetManager
from model_utils import ModelWrapper

import wandb

import json
import os

import sys
sys.path.append('../')


def train_and_predict_with_probe(args, best_model_state, x_train, y_train, x_val, y_val, x_test, y_test, task_type, all_train_labels, layer_id, hidden_size, seed=0):
    ## seed
    torch.manual_seed(seed)
    
    ## process labels
    if task_type == 'classification':
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
        # align commonsense_qa with social_iqa
        if 'commonsense' in args.train_dataset_name:
            # For training set
            x_train = x_train[torch.from_numpy(y_train < 3).to(x_train.device)]
            y_train = y_train[y_train < 3]

            # For validation set
            x_val = x_val[torch.from_numpy(y_val < 3).to(x_val.device)]
            y_val = y_val[y_val < 3]

            # For test set
            x_test = x_test[torch.from_numpy(y_test < 3).to(x_test.device)]
            y_test = y_test[y_test < 3]

        # convert integer labels to array
        label_encoder = LabelEncoder()
        if args.train_dataset_name == args.test_dataset_name and args.process_step == -1:
            y_all = np.concatenate([y_train, y_val, y_test])
        else:
            y_all = y_train
        labels_ = y_all
        label_encoder.fit(labels_)
        # Add an extra class by extending classes_ array
        label_encoder.classes_ = np.append(label_encoder.classes_, max(label_encoder.classes_) + 1)
        y_train = label_encoder.transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)
        # set number of class including the extra uncertainty class
        args.class_num = len(label_encoder.classes_)
        # Store original class count for later filtering
        original_class_count = len(label_encoder.classes_) - 1
    else:
        scaler = StandardScaler()
        if args.train_dataset_name == args.test_dataset_name and args.process_step == -1:
            y_all = np.concatenate([y_train, y_val, y_test])
        else:
            y_all = all_train_labels
        y_ = y_all
        y_ = scaler.fit_transform(y_.reshape(-1, 1))
        y_train = y_[:len(y_train)]
        y_val = y_[len(y_train):len(y_train)+len(y_val)]
        if args.train_dataset_name == args.test_dataset_name and args.process_step == -1:
            y_test = y_[len(y_train)+len(y_val):]
        else:
            y_test = scaler.transform(y_test.reshape(-1, 1))
    ## process data
    if isinstance(x_train, np.ndarray):
        x_train = torch.Tensor(x_train)
        x_val = torch.Tensor(x_val)
        x_test = torch.Tensor(x_test)
    if isinstance(y_train, np.ndarray):
        y_train = torch.Tensor(y_train)
        y_val = torch.Tensor(y_val)
        y_test = torch.Tensor(y_test)
    x_train_tensor = x_train.reshape(-1, x_train.shape[1]).to(torch.device('cuda')).to(torch.float32)
    x_val_tensor = x_val.reshape(-1, x_val.shape[1]).to(torch.device('cuda')).to(torch.float32)
    x_test_tensor = x_test.reshape(-1, x_test.shape[1]).to(torch.device('cuda')).to(torch.float32)
    if task_type == 'regression':
        y_train_tensor = y_train.reshape(-1, 1).to(torch.device('cuda')).to(torch.float32)
        y_val_tensor = y_val.reshape(-1, 1).to(torch.device('cuda')).to(torch.float32)
        output_size = 1
    else:
        y_train_tensor = y_train.reshape(-1).to(torch.device('cuda')).to(torch.long)
        y_val_tensor = y_val.reshape(-1).to(torch.device('cuda')).to(torch.long)
        output_size = args.class_num
    
    ## model structure
    layers = [
        torch.nn.Linear(x_train.shape[1], hidden_size),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden_size, output_size),
    ]
    if task_type == 'classification':
        layers.append(torch.nn.Softmax(dim=1))
    model = torch.nn.Sequential(*layers).to(torch.device('cuda'))

    # Add proper initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(init_weights)

    ## start traininig
    if not best_model_state:
        best_model_state = copy.deepcopy(model.state_dict())
        best_val_result = 1e6 if task_type == 'regression' else 0
        N_EPOCHS = args.n_epochs
        N_VAL_INTERVAL = 1
        patience = 40  # Number of epochs to wait for improvement
        min_delta = 1e-6  # Minimum change in validation loss to qualify as an improvement
        patience_counter = 0
        min_lr = 1e-6
        
        # loss function
        loss_fn = torch.nn.MSELoss() if task_type == 'regression' else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler_obj = GradScaler()

        # train with batch size=8
        for t in range(N_EPOCHS):
            # Generate shuffled indices
            indices = torch.randperm(len(x_train_tensor))
            x_train_shuffled = x_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]
            
            total_train_loss = []

            for j in range(0, len(x_train_tensor), args.batch_size):
                # breakpoint()
                with autocast():
                    y_pred = model(x_train_shuffled[j:j+args.batch_size])
                    loss = loss_fn(y_pred, y_train_shuffled[j:j+args.batch_size])
                optimizer.zero_grad()

                scaler_obj.scale(loss).backward()
                scaler_obj.step(optimizer)
                scaler_obj.update()

                total_train_loss.append(loss.item())
                

            if t % N_VAL_INTERVAL == 0:
                with torch.no_grad():
                    y_pred_val = model(x_val_tensor)
                    
                    if task_type == 'classification':
                        # For classification, compute accuracy
                        val_pred = torch.argmax(y_pred_val, dim=1)
                        val_accuracy = (val_pred == y_val_tensor).float().mean().item()
                        loss_val = loss_fn(y_pred_val, y_val_tensor)
                        
                        # Log metrics
                        wandb.log({
                            f'h{hidden_size}_l{layer_id}': {
                                'train_loss': np.mean(total_train_loss),
                                'val_loss': loss_val.item(),
                                'val_accuracy': val_accuracy
                            },
                            'epoch': t
                        })
                        
                        # Save best model based on accuracy
                        if val_accuracy > best_val_result + min_delta:
                            best_val_result = val_accuracy
                            best_model_state = copy.deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    else:
                        # Original regression logic
                        loss_val = loss_fn(y_pred_val, y_val_tensor)
                        
                        wandb.log({
                            f'h{hidden_size}_l{layer_id}': {
                                'train_loss': np.mean(total_train_loss),
                                'val_loss': loss_val.item()
                            },
                            'epoch': t
                        })
                        
                        if loss_val < best_val_result - min_delta:
                            best_val_result = loss_val
                            best_model_state = copy.deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    
                    # Early stopping
                    current_lr = optimizer.param_groups[0]['lr']
                    if patience_counter >= patience or current_lr < min_lr:
                        break
    # load model; do not train
    else:
        pass
    
    # predict
    with torch.no_grad():
        # load best model state
        model.load_state_dict(best_model_state)
        
        if task_type == 'classification':
            # Get predictions and probabilities
            logits_test = model(x_test_tensor)
            logits_train = model(x_train_tensor)
            logits_val = model(x_val_tensor)
            
            # Get both predictions and probabilities from the same logits
            probs_test = torch.softmax(logits_test, dim=1)
            probs_train = torch.softmax(logits_train, dim=1)
            probs_val = torch.softmax(logits_val, dim=1)
            
            # Get predictions from probabilities
            y_pred_test = torch.argmax(probs_test[:, :-1], dim=1).cpu().numpy()
            y_pred_train = torch.argmax(probs_train[:, :-1], dim=1).cpu().numpy()
            y_pred_val = torch.argmax(probs_val[:, :-1], dim=1).cpu().numpy()
            
            # Store probabilities
            y_prob_test = probs_test[:, :original_class_count].cpu().numpy()  # Exclude uncertainty class probabilities
            y_prob_train = probs_train[:, :original_class_count].cpu().numpy()
            y_prob_val = probs_val[:, :original_class_count].cpu().numpy()
            
            # Transform predictions back to original label space
            y_pred_test = label_encoder.classes_[y_pred_test]
            y_pred_train = label_encoder.classes_[y_pred_train]
            y_pred_val = label_encoder.classes_[y_pred_val]
            
            # Calculate scores using original labels
            train_score = accuracy_score(label_encoder.inverse_transform(y_train.cpu().to(int)), y_pred_train)
            val_score = accuracy_score(label_encoder.inverse_transform(y_val.cpu().to(int)), y_pred_val)
            test_score = accuracy_score(label_encoder.inverse_transform(y_test.cpu().to(int)), y_pred_test)
        else:
            # Original regression prediction logic
            y_pred_test = model(x_test_tensor).cpu().detach().numpy()
            y_pred_train = model(x_train_tensor).cpu().detach().numpy()
            y_pred_val = model(x_val_tensor).cpu().detach().numpy()
            y_prob_test, y_prob_train, y_prob_val = None, None, None

            train_score = mean_squared_error(y_train, y_pred_train)
            val_score = mean_squared_error(y_val, y_pred_val)
            test_score = mean_squared_error(y_test, y_pred_test)
            
            # Rescale for regression
            y_pred_test = scaler.inverse_transform(y_pred_test)
            y_pred_train = scaler.inverse_transform(y_pred_train)
            y_pred_val = scaler.inverse_transform(y_pred_val)

    return best_model_state, y_pred_test, y_pred_train, y_pred_val, (y_prob_test, y_prob_train, y_prob_val), {'train': float(train_score), 'test': float(test_score), 'val': float(val_score)}
    
    
def process_dataset_predictions(args, dataset, y_pred_lists, y_pred_prob_lists, split_type, output_log, task_type, hidden_sizes, save_mode):
    """Process predictions for a given dataset partition with multiple hidden sizes"""
    y_pred_index = 0
    layer_num = len(y_pred_lists[0])  # Number of layers
    
    for idx, (_, row) in enumerate(dataset.iterrows()):
        prompt = row['prompt']
        
        for label_idx, real_label in enumerate(row['augmented_labels']):
            response = row['response'] if len(row['augmented_labels']) == 1 else row['truncated_responses'][label_idx]

            result = {
                'prompt': prompt,
                'response': response,
                'real_label': real_label,
                'split_type': split_type,
                'predictions': {}
            }
            # align commonsense_qa with social_iqa
            if 'commonsense' in args.train_dataset_name:
                if real_label >= 3:
                    continue
            
            # Add predictions for each hidden size
            for hidden_size_idx, hidden_size in enumerate(hidden_sizes):
                hidden_size_predictions = {}
                
                # Add predictions for each layer
                if save_mode == 1 or save_mode == 0:
                    for layer_idx in range(layer_num):
                        # for test only with the first label
                        try:
                            pred = y_pred_lists[hidden_size_idx][layer_idx][y_pred_index].item()
                        except:
                            break
                        
                        if layer_idx == layer_num-1:
                            hidden_size_predictions[f'pred_label_all'] = pred
                        else:
                            hidden_size_predictions[f'pred_label_{layer_idx}'] = pred
                        
                        if task_type == 'classification' and y_pred_prob_lists[hidden_size_idx][layer_idx] is not None:
                            prob = y_pred_prob_lists[hidden_size_idx][layer_idx][y_pred_index].tolist()[1]
                            if layer_idx == layer_num-1:
                                hidden_size_predictions[f'pred_prob_all'] = prob
                            else:
                                hidden_size_predictions[f'pred_prob_{layer_idx}'] = prob
                result['predictions'][str(hidden_size)] = hidden_size_predictions
            
            y_pred_index += 1
            output_log.append(result)


def should_train_probes(args, dataset_manager, HIDDEN_SIZES):
    """Determines if probes should be trained or loaded."""
    (probe_exists, results_path_exists) = dataset_manager.check_if_results_exists(HIDDEN_SIZES)
    force_train = args.retrain_if_state_exists
    different_dataset = args.train_dataset_name != args.test_dataset_name
    different_model = args.activation_model_name != ''
    ablation_flag = args.shuffle_activations or args.randomize_labels or args.non_trained_probes
    
    should_train = force_train or (not different_dataset and not probe_exists) and args.process_step == -1
    should_save = should_train and not different_dataset and args.process_step == -1 and not ablation_flag and not different_model
    should_quit = False #(probe_exists or different_model) and not force_train and results_path_exists #and not ablation_flag # and not different_dataset; not ablation_flag can be removed
    # save modes: 1 - layer_wise (in detail); 0 - best results (only record the best layer)
    save_mode = not ablation_flag and not different_model and not different_dataset and args.process_step == -1
    
    print(f"Probe exists: {probe_exists}, result file exists: {results_path_exists}")
    print(f"Training probes: {should_train}")
    print(f"Will save probes: {should_save}")
    print(f"Should quit: {should_quit}")
    print(f"Save mode: {save_mode}")
    
    return force_train, should_train, should_save, should_quit, save_mode


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def main(): 
    ## basic settings & args
    with open('config.yaml') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=global_config['model_path'].keys(), help='model name')
    parser.add_argument('--activation_model_name', type=str, default='', choices=list(global_config['model_path'].keys()) + [''], help='activation model name')

    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')


    # setting
    parser.add_argument('--retrain_if_state_exists', type=str2bool, default=False, help='whether or not to overwrite the existing file')
    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs')


    # data
    parser.add_argument('--test_dataset_name', type=str, default='thu-coai/SafetyBench', help='train/test data')
    parser.add_argument('--train_dataset_name', type=str, default='thu-coai/SafetyBench', help='train data')
    parser.add_argument('--in_train_data_prefix', type=str, default='', help='train data prefix')
    parser.add_argument('--in_test_data_prefix', type=str, default='', help='test data prefix')
    parser.add_argument('--out_data_prefix', type=str, default='', help='out data prefix')
    parser.add_argument('--class_num', type=int, default=2, help='class number')
    parser.add_argument('--prompt_template_type', type=str, default='default', help='prompt template')
    parser.add_argument('--data_prefix', type=str, default='', help='data prefix')
    
    # probe
    parser.add_argument('--probe_type', type=str, default='LogisticRegression', help='probe type')

    # training
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='val ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    
    # ablation_study
    parser.add_argument('--shuffle_activations', type=str2bool, default=False, help='shuffle activations')
    parser.add_argument('--randomize_labels', type=str2bool, default=False, help='randomize labels')
    parser.add_argument('--non_trained_probes', type=str2bool, default=False, help='non trained probes')
    parser.add_argument('--process_step', type=int, default=-1, help='process study, -1 stands for not implemented')

    # log
    parser.add_argument('--log_dir_base', type=str, default='data/results', help='log directory')

    args = parser.parse_args()

    ## configs
    args.task_type = global_config['template_types'][args.prompt_template_type]['task_type']
    
    # init results
    if not args.shuffle_activations and not args.randomize_labels and not args.non_trained_probes:
        HIDDEN_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    else:
        HIDDEN_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        args.n_epochs = 200
        if args.non_trained_probes:
            args.n_epochs = 0
    all_score_by_layer = {str(size): {'test': {}, 'train': {}, 'val': {}} for size in HIDDEN_SIZES}
    all_predictions = {
        'test': {str(size): [] for size in HIDDEN_SIZES},
        'train': {str(size): [] for size in HIDDEN_SIZES},
        'val': {str(size): [] for size in HIDDEN_SIZES}
    }
    all_probabilities = {
        'test': {str(size): [] for size in HIDDEN_SIZES},
        'train': {str(size): [] for size in HIDDEN_SIZES},
        'val': {str(size): [] for size in HIDDEN_SIZES}
    }
    

    ## data
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    ## load model
    args.model_path = global_config['model_path'][args.model_name]
    dataset_manager = DatasetManager(args, "testing_probes")
    features_and_labels = {}
    for param in ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']:
        features_and_labels[param] = getattr(dataset_manager, param)
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = getattr(dataset_manager, f"{split}_dataset")


    ## Training and Testing
    # check if results exists
    force_train, should_train, should_save, should_quit, save_mode = should_train_probes(args, dataset_manager, HIDDEN_SIZES)
    
    if should_quit:
        print(f"Probe and results already exists. Quitting...")
        return
    elif not should_train:
        print(f"Probe already exists. Load the already trained probe.")
    else:
        pass
    if force_train:
        print(f"Force training probes.")
        probe_state_dicts = {str(hidden_size): {str(i) if i < features_and_labels['x_train'].shape[1] else 'all_layers': None for i in range(features_and_labels['x_train'].shape[1]+1)} for hidden_size in HIDDEN_SIZES}
    else:
        probe_state_dicts = dataset_manager.load_probes(HIDDEN_SIZES, model_layer_num=features_and_labels['x_train'].shape[1])
    
    # train probes
    if should_train:
        wandb.init(
            project='edging',
            name=f"{args.data_prefix}{args.prompt_template_type}_{args.train_dataset_name.split('/')[-1]}_{args.model_name}_seed{args.seed}",
            config={
                'model': args.model_name,
                'hidden_sizes': HIDDEN_SIZES,
                'task_type': args.task_type,
                'probe_type': args.probe_type,
                'batch_size': args.batch_size,
                'seed': args.seed
            }
        )
        
    for hidden_size in HIDDEN_SIZES:
        print(f"\nTesting with hidden size {hidden_size}")
        # Initialize probe state dict for current hidden size
        y_pred_test_list, y_pred_train_list, y_pred_val_list = [], [], []
        y_prob_test_list, y_prob_train_list, y_prob_val_list = [], [], []
        
        probe_state_dict = probe_state_dicts[str(hidden_size)]
        
        
        for layer_idx in tqdm(range(features_and_labels['x_train'].shape[1]+1), desc=f'Probing Model {args.model_name} with probe {args.probe_type} and hidden size {hidden_size} under seed {args.seed}', total=features_and_labels['x_train'].shape[1]+1):
            # if last layer: train a probe with all layers
            layer_name = str(layer_idx) if layer_idx < features_and_labels['x_train'].shape[1] else 'all_layers'
            if layer_idx == features_and_labels['x_train'].shape[1]:
                x_train_for_layer = features_and_labels['x_train'].reshape(features_and_labels['x_train'].shape[0], -1)
                x_test_for_layer = features_and_labels['x_test'].reshape(features_and_labels['x_test'].shape[0], -1)
                x_val_for_layer = features_and_labels['x_val'].reshape(features_and_labels['x_val'].shape[0], -1)
            else:
                x_train_for_layer = features_and_labels['x_train'][:, layer_idx, :]
                x_test_for_layer = features_and_labels['x_test'][:, layer_idx, :]
                x_val_for_layer = features_and_labels['x_val'][:, layer_idx, :]
            
            # train and predict
            best_model_state, y_pred_test, y_pred_train, y_pred_val, (y_prob_test, y_prob_train, y_prob_val), score = train_and_predict_with_probe(
                args, 
                probe_state_dict[layer_name],
                x_train_for_layer, 
                features_and_labels['y_train'], 
                x_val_for_layer,
                features_and_labels['y_val'],
                x_test_for_layer, 
                features_and_labels['y_test'], 
                args.task_type, 
                dataset_manager.all_train_labels,
                layer_name,
                hidden_size, 
                args.seed
            )
            
            # second layer for classification: output num of class
            if args.task_type == 'classification' and layer_idx == 0:
                tqdm.write(f"Number of classes: {args.class_num}")
            
            # save model state
            if best_model_state is not None and should_save:
                dataset_manager.save_results(
                    results=best_model_state, 
                    layer_name=layer_name,
                    hidden_size=hidden_size
                )
                
            # save results
            all_score_by_layer[str(hidden_size)]['train'][layer_name] = score['train']
            all_score_by_layer[str(hidden_size)]['val'][layer_name] = score['val']
            all_score_by_layer[str(hidden_size)]['test'][layer_name] = score['test']
            
            y_pred_test_list.append(y_pred_test)
            y_pred_train_list.append(y_pred_train)
            y_pred_val_list.append(y_pred_val)
            if args.task_type == 'classification':
                y_prob_test_list.append(y_prob_test)
                y_prob_train_list.append(y_prob_train)
                y_prob_val_list.append(y_prob_val)
                
        # Store predictions for current hidden size
        all_predictions['test'][str(hidden_size)] = y_pred_test_list
        all_predictions['train'][str(hidden_size)] = y_pred_train_list
        all_predictions['val'][str(hidden_size)] = y_pred_val_list
        
        if args.task_type == 'classification':
            all_probabilities['test'][str(hidden_size)] = y_prob_test_list
            all_probabilities['train'][str(hidden_size)] = y_prob_train_list
            all_probabilities['val'][str(hidden_size)] = y_prob_val_list

    if should_train:
        wandb.finish()
            

    ## save results
    output_log = [{'scores': all_score_by_layer}]

    # Process predictions for each dataset split
    process_dataset_predictions(
        args,
        datasets['test'],
        [all_predictions['test'][str(size)] for size in HIDDEN_SIZES],
        [all_probabilities['test'][str(size)] for size in HIDDEN_SIZES],
        'test',
        output_log,
        args.task_type,
        HIDDEN_SIZES,
        save_mode
    )
    
    # process_dataset_predictions(
    #     datasets['train'],
    #     [all_predictions['train'][str(size)] for size in HIDDEN_SIZES],
    #     [all_probabilities['train'][str(size)] for size in HIDDEN_SIZES],
    #     'train',
    #     output_log,
    #     args.task_type,
    #     HIDDEN_SIZES,
    #     save_mode
    # )
    
    # process_dataset_predictions(
    #     datasets['val'],
    #     [all_predictions['val'][str(size)] for size in HIDDEN_SIZES],
    #     [all_probabilities['val'][str(size)] for size in HIDDEN_SIZES],
    #     'val',
    #     output_log,
    #     args.task_type,
    #     HIDDEN_SIZES,
    #     save_mode
    # )
        
    
    # save results
    dataset_manager.save_results(results=output_log)
    
if __name__ == "__main__":
    main()