import os
from typing import Dict
import numpy as np
import glob
import random

from ray import train
from ray.air import session
from ray.train import Checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import torchvision
from torchvision import models

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


# Training loop.
def imagenet_dali_train_loop_per_worker(config: Dict):
    
        # set seeds
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])

        current_iteration = 1
        
        # get world size
        grank = session.get_world_rank()
        gwsize = session.get_world_size()
        lrank = session.get_local_rank()

        device = torch.device('cuda', lrank)
        torch.cuda.set_device(lrank)
        
        train.torch.accelerate(amp=config["amp"])

        # build model
        if config["model"] == 18:
            model = models.resnet18()

        if config["model"] == 50: 
            model = models.resnet50()

        if config["model"] == 101: 
            model = models.resnet101()
            
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(config["kernel_size"], config["kernel_size"]), stride=(2, 2), padding=(3, 3), bias=False)
        
        if config["activation_function"] == "ReLU":
            new_activation = torch.nn.ReLU
        elif config["activation_function"] == "LeakyReLU":
            new_activation = torch.nn.LeakyReLU        
        elif config["activation_function"] == "ELU":
            new_activation = torch.nn.ELU
        elif config["activation_function"] == "SELU":
            new_activation = torch.nn.SELU            
        elif config["activation_function"] == "Tanh":
            new_activation = torch.nn.Tanh                
        elif config["activation_function"] == "Sigmoid":
            new_activation = torch.nn.Sigmoid                
        elif config["activation_function"] == "GELU":
            new_activation = torch.nn.GELU                
            
        replace_activations(model, torch.nn.ReLU, new_activation)

        # define optimizer and loss function
        criterion = nn.CrossEntropyLoss().cuda()
        
        if config["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["lr"] * gwsize, weight_decay=config["wd"])
        elif config["optimizer"] == "adam": 
            optimizer = optim.AdamW(model.parameters(), lr=config["lr"]*np.sqrt(gwsize), betas=(0.9, 0.999), weight_decay=config["wd"])
        
        
        # prepare the model and optimizer for Ray Tune
        model = train.torch.prepare_model(model)
        optimizer = train.torch.prepare_optimizer(optimizer)

        # checkpoint loading
        loaded_checkpoint = train.get_checkpoint()

        if loaded_checkpoint:    
            with loaded_checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                model.load_state_dict(checkpoint_dict["model_state"])
                #optimizer.load_state_dict(checkpoint_dict["optimizer_state"])

                # Note: Make sure to increment the checkpointed step by 1 to get the current step.
                last_iteration = checkpoint_dict["current_iteration"]
                current_iteration = last_iteration + 1
        

        # dataset preperation
        train_tfrecord = sorted(glob.glob(config["data_dir"] + "/train/train-*-of-01024"))
        train_tfrecord_idx = sorted(glob.glob(config["data_dir"] + "/train/idx_files/train-*-of-01024.idx"))

        val_tfrecord_files = sorted(glob.glob(config["data_dir"] + "/val/validation-*-of-00128"))
        val_tfrecord_idx_files = sorted(glob.glob(config["data_dir"] + "/val/idx_files/validation-*-of-00128.idx"))
        
        
        # Zip the lists together
        zipped_list = list(zip(val_tfrecord_files, val_tfrecord_idx_files))

        # Shuffle the zipped list
        random.shuffle(zipped_list)

        # Unzip the lists
        val_tfrecord_files, val_tfrecord_idx_files = zip(*zipped_list)
        
        # Define the validation/test split ratio
        split_ratio = 0.5
        split_index = int(len(val_tfrecord_files) * split_ratio)

        # Split the TFRecord files into new validation and test sets
        validation_tfrecord = val_tfrecord_files[:split_index]
        test_tfrecord = val_tfrecord_files[split_index:]

        # Split the index files similarly
        validation_idx = val_tfrecord_idx_files[:split_index]
        test_idx = val_tfrecord_idx_files[split_index:]

        # image preprocessing steps:
        crop_size = 224
        val_size = 256
        
        # build dataloaders
        train_pipe = tfrecord_reader_pipeline(path=train_tfrecord, index_path=train_tfrecord_idx, batch_size=int(config["bs"]), num_threads=30, device_id=lrank, is_training=True, shard_id=grank, num_shards=gwsize, dali_cpu=False, crop=crop_size, size=val_size)

        train_pipe.build()

        train_loader = DALIGenericIterator(train_pipe, ['data', 'label'], reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)   
        
        val_pipe = tfrecord_reader_pipeline(path=sorted(validation_tfrecord), index_path=sorted(validation_idx), batch_size=int(config["bs"]), num_threads=30, device_id=lrank, is_training=False, shard_id=grank, num_shards=gwsize, dali_cpu=False, crop=crop_size, size=val_size)

        val_pipe.build()

        val_loader = DALIGenericIterator(val_pipe, ['data', 'label'], reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)           
        
        test_pipe = tfrecord_reader_pipeline(path=sorted(test_tfrecord), index_path=sorted(test_idx), batch_size=int(config["bs"]), num_threads=30, device_id=lrank, is_training=False, shard_id=grank, num_shards=gwsize, dali_cpu=False, crop=crop_size, size=val_size)

        test_pipe.build()

        test_loader = DALIGenericIterator(test_pipe, ['data', 'label'], reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)      

        # lr schedule
        train_loader_len = len(train_loader)
        if current_iteration == 1:
            warmup_steps = train_loader_len*int(config["init_warmup_steps"])
            scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step/warmup_steps)
        else:
            warmup_steps = train_loader_len*int(config["rescale_warmup_steps"])
            scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(step/warmup_steps, 0.5))
        
        
        # actual train loop
        for epoch in range(1, (config["num_epochs"]*config["report_interval"])+1):                                  
            # prepare metrics
            train_acc = 0
            train_correct = 0
            train_total = 0

            # prepare model for training and loop over training dataset
            model.train()
            for i, data in enumerate(train_loader):
                
                images = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()
                target = target - 1 
                images = images.to(device)
                target = target.to(device)
                                  
                scheduler_lr.step()
                
                # compute output
                optimizer.zero_grad()
                output = model(images)

                # compute loss
                loss = criterion(output, target)

                # count correct classifications
                tmp_correct, tmp_total = accuracy(output, target)    
                train_correct +=tmp_correct
                train_total +=tmp_total

                # backpropagation and optimization step
                #loss.backward() 
                train.torch.backward(loss)
                optimizer.step()

            # average the train metrics over all workers
            train_correct = par_sum(train_correct, device)
            train_total = par_sum(train_total, device)

            # compute final training accuracy
            train_acc = train_correct/train_total

            if (epoch % config["report_interval"] == 0):
                
                # validation loop
                                  
                # prepare metrics
                val_acc = 0
                val_correct = 0
                val_total = 0

                # prepare model for training and loop over training dataset
                model.eval()
                for i, data in enumerate(val_loader):

                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                    target = target - 1 
                    images = images.to(device)
                    target = target.to(device)

                    with torch.no_grad():
                        # compute output
                        output = model(images)

                    # count correct classifications
                    tmp_correct, tmp_total = accuracy(output, target)    
                    val_correct +=tmp_correct
                    val_total +=tmp_total

                # average the val metrics over all workers
                val_correct = par_sum(val_correct, device)
                val_total = par_sum(val_total, device)

                # compute final val accuracy
                val_acc = val_correct/val_total
                
                # test loop
                                  
                # prepare metrics
                test_acc = 0
                test_correct = 0
                test_total = 0

                # prepare model for training and loop over training dataset
                model.eval()
                for i, data in enumerate(test_loader):

                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                    target = target - 1 
                    images = images.to(device)
                    target = target.to(device)

                    with torch.no_grad():
                        # compute output
                        output = model(images)

                    # count correct classifications
                    tmp_correct, tmp_total = accuracy(output, target)    
                    test_correct +=tmp_correct
                    test_total +=tmp_total

                # average the val metrics over all workers
                test_correct = par_sum(test_correct, device)
                test_total = par_sum(test_total, device)

                # compute final val accuracy
                test_acc = test_correct/test_total


                # save checkpoint
                checkpoint = None
                if grank == 0:
                    os.makedirs("tune_model", exist_ok=True)
                    torch.save(
                        {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "current_iteration": current_iteration},
                        "tune_model/checkpoint.pt",
                    )
                    checkpoint = Checkpoint.from_directory("tune_model")
                session.report({"train_acc": train_acc.item(), "val_acc": val_acc.item(), "test_acc": test_acc.item(), "num_gpus": session.get_world_size()}, checkpoint=checkpoint)

# helper function for accuracy    
def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]

    # count correct classifications
    correct = pred.eq(target.view_as(pred)).cpu().float().sum()

    # count total samples
    total = target.size(0)
    return correct, total


def par_sum(field, device):
    """! function that sums a field across all workers to a worker
    @param field field in worker that should be summed up

    @return sum of all fields
    """
    # convert field to tensor
    res = torch.Tensor([field])

    # move field to GPU/worker
    res = res.to(device)

    dist.all_reduce(res,op=dist.ReduceOp.SUM)

    return res

def common_pipeline(images, labels, crop, size, dali_cpu=False, is_training=True):
    
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

@pipeline_def
def tfrecord_reader_pipeline(path, index_path, is_training, shard_id, num_shards, dali_cpu, crop, size):
    inputs = fn.readers.tfrecord(
        path = path,
        index_path = index_path,
        features = {
            "image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)},
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards, 
        name='Reader')
    
    return common_pipeline(inputs["image/encoded"], inputs["image/class/label"], crop, size, dali_cpu, is_training)


def replace_activations(model, old_activation, new_activation):
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation())
        else:
            replace_activations(module, old_activation, new_activation)
                                  


