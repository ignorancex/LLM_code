import argparse
import os
import glob
import math
import yaml
import shutil
import pickle
import random
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms as T

from torch.utils.tensorboard import SummaryWriter
from dataset import SceneDataset, GraphDataset, prepare_dataset_graphs_mp
from models import LANDER, load_feature_extractor
from utils import metrics

from test import inference

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

def collate(batch):
    graphs = []
    for sample in batch:
        graphs.extend([g for g in sample['graphs']])
    return graphs

def main(config_path, device, collate_fun):
    # Load config
    config = yaml.safe_load(open(config_path, 'r'))

    #################################
    # Load feature extraction model #
    #################################
    feature_model = load_feature_extractor(config, device)

    #############
    # Load data #
    #############

    # Transformations
    img_transform = T.Compose([
        T.Resize(config['IMG_TRANSFORM']['RESIZE']),
        T.ToTensor(),
        T.Normalize(mean=config['IMG_TRANSFORM']['MEAN'],
                    std=config['IMG_TRANSFORM']['STD'])
    ])

    data_root = config['DATA_ROOT']

    # Load training
    train_ds = []
    for d in config['DATASET_TRAIN']:
        data_path = os.path.join(data_root, d + '_crops.pkl')
        coo2meter = config['MAX_DIST'][d]
        with open(data_path, "rb") as f:
            train_seq = pickle.load(f)

        train_ds.append(SceneDataset(train_seq,
                                     coo2meter,
                                     feature_model,
                                     device,
                                     img_transform))
    train_ds = torch.utils.data.ConcatDataset(train_ds)
    print(f'Training length: {len(train_ds)}')

    scene_seq = [sample for sample in train_ds]
    gss = prepare_dataset_graphs_mp(sequence=scene_seq,
                                    k=config['GNN_MODEL']['K'],
                                    levels=config['GNN_MODEL']['LEVELS'],
                                    device=device,
                                    num_workers=config['DATALOADER']['NUM_WORKERS'])
    train_gs_ds = GraphDataset(scene_seq, gss)
    train_gs_dl = torch.utils.data.DataLoader(dataset=train_gs_ds,
                                              batch_size=config['BATCH_SIZE'],
                                              shuffle=True,
                                              collate_fn=collate_fun,
                                              drop_last=False)

    # Load validation
    data_path = os.path.join(data_root, config['DATASET_VAL'] + '_crops.pkl')
    coo2meter = config['MAX_DIST'][config['DATASET_VAL']]
    with open(data_path, "rb") as f:
        val_seq = pickle.load(f)
    val_ds = SceneDataset(val_seq,
                          coo2meter,
                          feature_model,
                          device,
                          img_transform)
    print(f'Validation length: {len(val_ds)}')

    # Final validation ds can be created once
    scene_seq = [sample for sample in val_ds]
    gss = prepare_dataset_graphs_mp(sequence=scene_seq,
                                    k=config['GNN_MODEL']['K'],
                                    levels=config['GNN_MODEL']['LEVELS'],
                                    device=device,
                                    num_workers=config['DATALOADER']['NUM_WORKERS'])
    val_gs_ds = GraphDataset(scene_seq, gss)
    val_gs_dl = torch.utils.data.DataLoader(dataset=val_gs_ds,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=False,
                                            collate_fn=collate_fun,
                                            drop_last=False)

    ##################
    # Model Definition
    node_feature_dim = train_ds[0]['node_embeds'].shape[1]
    model = LANDER(feature_dim=node_feature_dim,
                   nhid=config['GNN_MODEL']['HIDDEN_DIM'],
                   num_conv=config['GNN_MODEL']['NUM_CONV'],
                   dropout=config['GNN_MODEL']['DROPOUT'],
                   use_GAT=config['GNN_MODEL']['GAT'],
                   K=config['GNN_MODEL']['GAT_K'],
                   use_cluster_feat=config['GNN_MODEL']['USE_CLUSTER_FEATURE'],
                   use_focal_loss=config['GNN_MODEL']['USE_FOCAL_LOSS'])
    model = model.to(device)
    model.train()
    best_vmes = -np.Inf

    #################
    # Hyperparameters
    opt = optim.Adam(
        model.parameters(),
        lr=config['BASE_LR'],
        weight_decay=config['WEIGHT_DECAY'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=config['BASE_LR'],
        steps_per_epoch=math.ceil(len(train_ds) / config['BATCH_SIZE']),
        epochs=config['EPOCHS']
    )

    #################
    # Checkpoint paths
    model_dir = os.path.join(config['CHECKPOINT_DIR'], config['MODEL_NAME'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb_writer = SummaryWriter(log_dir=model_dir) # init tb writer
    shutil.copy(config_path, model_dir)

    ###############
    # Training Loop
    for epoch in range(config['EPOCHS']):
        all_loss = 0
        model.train()

        for batch in train_gs_dl:
            loss = 0
            opt.zero_grad()
            curr_loss = 0
            for g in batch:
                processed_g = model(g)
                loss_conn = model.compute_loss(processed_g)
                loss = loss_conn.mean()
                curr_loss += loss.item()

            loss /= config['BATCH_SIZE']
            all_loss += curr_loss / config['BATCH_SIZE']
            loss.backward()
            opt.step()
        scheduler.step()

        # Record
        avg_loss = all_loss / len(train_gs_dl)
        print(
            "Training, epoch: %d, loss: %.6f"
            % (epoch, avg_loss)
        )

        # Add to tensorboard
        tb_writer.add_scalar('Train/Loss', avg_loss, epoch)

        # Report validation
        all_loss_val = 0

        with torch.no_grad():
            for val_batch in val_gs_dl:
                curr_loss = 0
                for g in val_batch:
                    processed_g = model(g)
                    loss_conn = model.compute_loss(processed_g)
                    curr_loss += loss_conn.mean().item()
                all_loss_val += curr_loss / config['BATCH_SIZE']

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

        avg_loss_val = all_loss_val / len(val_gs_dl)
        
        # Record
        print(
            "Validation, epoch: %d, loss: %.6f"
            % (epoch, avg_loss_val)
        )
        # Add to tensorboard
        tb_writer.add_scalar('Val/Loss', avg_loss_val, epoch)

        # Validation metrics
        rand_index = []
        ami = []
        homogeneity = []
        completeness = []
        v_measure = []
        if epoch % 5 == 0:
            model.eval()
            for sample in val_ds:
                labels = sample['node_labels']
                predictions = inference(sample['node_embeds'],
                                        labels.copy(),
                                        sample['xws'],
                                        sample['yws'],
                                        sample['cam_ids'],
                                        model,
                                        device,
                                        config)

                rand_index.append(metrics.ari(labels, predictions))
                ami.append(metrics.ami(labels, predictions))
                homogeneity.append(metrics.homg_score(labels, predictions))
                completeness.append(metrics.cmplts_score(labels, predictions))
                v_measure.append(metrics.v_mesure(labels, predictions))

            m_ridx = np.mean(np.asarray(rand_index))
            m_ami = np.mean(np.asarray(ami))
            m_hom = np.mean(np.asarray(homogeneity))
            m_cmplt = np.mean(np.asarray(completeness))
            m_vmes = np.mean(np.asarray(v_measure))

            # Save ckpt
            if m_vmes > best_vmes:
                best_vmes = m_vmes
                print("\nNew best epoch", epoch)
                best_model = glob.glob(os.path.join(model_dir, 'model_best-*.pth'))
                if len(best_model):
                    os.remove(best_model[0])
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_best-{epoch}.pth'))

            print(f'Rand index mean = {m_ridx}')
            print(f'Mutual index mean = {m_ami}')
            print(f'homogeneity mean = {m_hom}')
            print(f'completeness mean = {m_cmplt}')
            print(f'v_measure mean = {m_vmes}')
            print('\n')

            tb_writer.add_scalar('Val_Metrics/Rand_idx', m_ridx, epoch)
            tb_writer.add_scalar('Val_Metrics/AMI', m_ami, epoch)
            tb_writer.add_scalar('Val_Metrics/Homogeneity', m_hom, epoch)
            tb_writer.add_scalar('Val_Metrics/Completeness', m_cmplt, epoch)
            tb_writer.add_scalar('Val_Metrics/V_Measure', m_vmes, epoch)

if __name__== '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='config/config_training.yaml',
                        type=str,
                        help='Path of the config file')

    args = parser.parse_args()

    ###########################
    # Environment Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args.config, device, collate)