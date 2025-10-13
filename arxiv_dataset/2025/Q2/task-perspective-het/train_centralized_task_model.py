import os
import pandas as pd
import numpy as np
import glob
import clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
import time
from argparse import ArgumentParser

import utils.metrics
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from datasets import BaseMultiTaskDataset
from models import HugeModel, HugeModelCBAM
from losses import return_task_loss
import utils


# Define argument parser
def parse_input():

    parser = ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='../data/taskonomy_dataset/')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_improvement_tolerance', type=int, default=5)
    parser.add_argument('--task', type=str)
    parser.add_argument('--init_lr', type=float, default=1e-1)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--checkpoint_path', type=str, default='./trained_models/')
    parser.add_argument('--init_model_dir', type=str, default='./initial_models/')
    parser.add_argument('--observe_performance', type=bool, default=False)
    parser.add_argument('--use_accelerator', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--use_cbam', type=bool, default=False)
    parser.add_argument('--use_adam', type=bool, default=False)
    
    
    return parser.parse_args()


args = parse_input()
print(args.__dict__)
path_middle_args = ['train_centralized', args.task]
current_run_path = utils.create_result_dir(args.result_path, path_middle_args)
chkpt_save_path = os.path.join(current_run_path, f"centralized_{args.task}_training{'_cbam' if args.use_cbam else None}_checkpoint.pth")
print_every = 1 if args.test_mode else 100

device = utils.get_device(args.use_accelerator)
global_dataset = utils.data.create_global_dataset(args, [args.task])

# print(f"{len(total_classes_df.class_idx.unique())} classes")

shuffled_indices = np.arange(len(global_dataset.data))
np.random.shuffle(shuffled_indices)

# Create train and validation datasets as subsets of the global dataset
train_dataset = Subset(global_dataset, shuffled_indices[:-100])
val_dataset = Subset(global_dataset, shuffled_indices[-100:])

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

sample_clip_model, ـ = clip.load("ViT-B/32", device=device)
if args.use_cbam:
    sample_model = HugeModelCBAM(sample_clip_model, args.task).to(device)
else:
    sample_model = HugeModel(sample_clip_model, args.task).to(device)
if args.use_adam:
    optimizer = torch.optim.Adam(sample_model.parameters(), lr=args.init_lr)
else:
    optimizer = torch.optim.SGD(sample_model.parameters(), lr=args.init_lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
loss_hist = []
best_loss = np.inf
no_improvement = 0
task_specific_metric_hist = None
if args.task in ["class_scene", "segment_semantic"]:
    task_specific_metric = utils.metrics.get_task_metric(args.task)
    task_specific_metric_hist = []

# f"{args.task}_model_clip_latest.pth"
# Check if a latest model is available in the trained_models directory
if args.resume_training:
    chkpt_load_path = os.path.join(args.checkpoint_path, f"centralized_{args.task}_training{'_cbam' if args.use_cbam else None}_checkpoint.pth")
    assert os.path.exists(chkpt_load_path), f"Checkpoint file not found at {chkpt_load_path}"
    print("Resuming training")
    training_checkpoint = torch.load(chkpt_load_path)
    sample_model.load_state_dict(training_checkpoint['model_state_dict'])
    optimizer.load_state_dict(training_checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(training_checkpoint['scheduler_state_dict'])
    loss_hist = training_checkpoint['loss_hist']
    best_loss = training_checkpoint['best_loss']
    no_improvement = training_checkpoint['no_improvement']
else:
    init_model_path = os.path.join(args.init_model_dir, f"init_model_{args.task}{'_cbam' if args.use_cbam else None}.pth")
    assert os.path.exists(init_model_path), "Initial model to load global model from does not exist! Use 'model_initialization.py' to create it!"
    sample_model.load_state_dict(torch.load(init_model_path))

criterion = return_task_loss(args.task)

for epoch in range(args.n_epochs):
    
    start_time = time.time()
    total_loss = 0
    
    for i, (images, labels) in enumerate(train_dataloader):

        if args.test_mode and i > 10:
            break
        
        # change images dtype to float16
        images = images.type(torch.float32).to(device)
        if args.task == "class_scene":
            labels[args.task] = labels[args.task].to(device)
        else:
            labels[args.task] = labels[args.task].type(torch.float32).to(device)

        # with torch.no_grad(): 
        #     image_features = sample_clip_model.encode_image(images)
        # _, outputs = decoder_under_train(image_features)
        
        _, outputs = sample_model(images)

        loss = criterion(outputs, labels[args.task])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress along with a hashtag bar
        if i % print_every == 0:
            print(f"Epoch {epoch+1}/{args.n_epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item()}", end='\r')

    # Get training stats
    with torch.no_grad():

        for i, (images, labels) in enumerate(val_dataloader):

            if args.test_mode and i > 10:
                break

            images = images.type(torch.float32).to(device)
            if args.task == "class_scene":
                labels[args.task] = labels[args.task].to(device)
            else:
                labels[args.task] = labels[args.task].type(torch.float32).to(device)

            # with torch.no_grad():
            #     image_features = sample_clip_model.encode_image(images)
            # _, outputs = decoder_under_train(image_features)
            
            _, outputs = sample_model(images)
            # print(outputs.shape)
            
            loss = criterion(outputs, labels[args.task])
            total_loss += loss.item()

            if args.task in ["segment_semantic", "class_scene"]:
                task_specific_metric_hist.append(task_specific_metric(outputs, labels[args.task]))

    end_time = time.time()
    loss_hist.append(total_loss/len(val_dataloader))
    print(f"Epoch {epoch+1}/{args.n_epochs}, Loss: {total_loss/len(val_dataloader)}, time: {end_time-start_time} seconds")
    if loss_hist[-1] < best_loss:
        best_loss = loss_hist[-1]
        no_improvement = 0
        torch.save(sample_model.state_dict(), os.path.join(current_run_path, f"{args.task}_model_clip{'_cbam' if args.use_cbam else None}.pth"))
        print("Improvement in loss. Saving model")
        
    else:
        no_improvement += 1
        print(f"No improvement in loss for {no_improvement} epochs")
    
    if not args.use_adam and no_improvement>=args.no_improvement_tolerance:
        scheduler.step()
    
    # Save checkpoint every 10 epochs
    if epoch%10==0:
        utils.save_chkpt(sample_model, optimizer, scheduler, loss_hist, best_loss, no_improvement, chkpt_save_path, task_specific_metric_hist)


# Plot the loss history
utils.plot_loss(loss_hist, current_run_path, args.task)

# Save training checkpoint
utils.save_chkpt(sample_model, optimizer, scheduler, loss_hist, best_loss, no_improvement, chkpt_save_path, task_specific_metric_hist)


# # Acquire the embeddings
# sample_model.load_state_dict(torch.load(os.path.join(current_run_path, f"{args.task}_model_clip.pth")))
# embeddings, test_outputs, test_targets = utils.extract_embeddings(sample_model, train_dataloader, device, args.task, args.test_mode)

# if args.observe_performance:
#     if args.task == "class_scene":
#         utils.plot_confusion_matrix(test_outputs, test_targets, args.task, current_run_path)
#     else:
#         utils.plot_outputs(test_outputs, test_targets, args.task, current_run_path)


# # Perform K-means on the embeddings
# print("Calculating K-means Clusters")
# embeddings = embeddings.view(embeddings.size(0), -1)
# kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, init='k-means++').fit(embeddings.cpu().numpy())
# centers = kmeans.cluster_centers_
# assignments = kmeans.labels_

# # Plot TSNE of the embeddings
# if args.task == "class_scene":
#     print("Plotting TSNE of the embeddings")
#     utils.plot_tsne(embeddings.cpu().numpy(), assignments, args.task, current_run_path)


