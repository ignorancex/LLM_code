import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import utils
import dllogger
import sys
from . import pruning_utils
import torchvision.models as models
from . import WRN_AAP as wrn 
from torch.cuda.amp import autocast, GradScaler
import pickle
scaler = GradScaler()

class ModelAndLoss(nn.Module):
    def __init__(
        self,
        arch,
        loss,
        cuda=True,
        mask=False,
        data_path="",
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print(f"=> creating model '{arch}'")
        
        if arch[0] == 'resnet152':
            pretrained_weights = torch.load('aap/image_classification/models/ResNet_152_10.th', map_location='cpu')['state_dict']
                
            for k in list(pretrained_weights):
                nk = k.replace('module.', '')
                pretrained_weights[nk] = pretrained_weights[k]
                del pretrained_weights[k]
                 
                if 'bn' in nk:
                    order = nk.split('bn')[1][0]
                    nnk = nk.replace('bn', 'conv'+order+'.'+'bn')
                    pretrained_weights[nnk] = pretrained_weights[nk]
                    del pretrained_weights[nk]
                
            model = resnet.resnet152()
        
        elif arch[0] == 'densnenet':
            pretrained_weights = torch.load('aap/image_classification/models/dense_tinyimagenet_10_epochs.pth', map_location='cpu')

            for k in list(pretrained_weights):
                nk = k.replace('module.', '')
                pretrained_weights[nk] = pretrained_weights[k]
                del pretrained_weights[k]
            
            model = densenet.densenet121()
            model.classifier = nn.Linear(model.classifier.in_features, 200)
        
        elif arch[0] == 'wrn-101':
            pretrained_weights = torch.load('aap/image_classification/models/wide_resnet101_2-32ee1156.pth', map_location='cpu')
             
            for k in list(pretrained_weights):
                if 'bn' in k and 'layer' in k and 'down' not in k:
                    order = k.split('bn')[1][0]
                    nnk = k.replace('bn', 'conv'+order+'.'+'bn')
                    pretrained_weights[nnk] = pretrained_weights[k]
                    del pretrained_weights[k]
            
            model = wrn.wide_resnet101_2() 
        
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights, strict=False)
        
        if cuda:
            model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
         
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)
        
        return loss, output
    
    def load_model_state(self, state):
        if not state is None:
            self.get_model().load_state_dict(state, strict=False)
    
    def get_model(self):
        return self.model

    def prune_next_conv(self, curr_name, prune_index, resample, reinit):
        next_conv_dict = {}
        layer_name_list = [layer_name for layer_name, _ in self.model.named_modules() if "conv" in layer_name and "relu" not in layer_name and "bn" not in layer_name and 'layer' in layer_name and 'downsample' not in layer_name]
        for i in range(0, len(layer_name_list)-1):
            next_conv_dict[layer_name_list[i]] = layer_name_list[i+1]

        for _name, _module in self.model.named_modules():
            if curr_name in next_conv_dict.keys() and next_conv_dict[curr_name] == _name:
                #print(f"curr_name: {curr_name}, next layer name (_name): {_name}")        
                _module.mask_on_input_channels(prune_index=prune_index, resample=resample, reinit=reinit)

    
    def process_file(self, name, outputs):
        #print(name)
        #print(outputs.keys())
        fir_card = 'cuda:0.' + name[7:] 
        sec_card = 'cuda:1.' + name[7:]
        #return outputs[fir_card]
        mean_out = (outputs[fir_card] + outputs[sec_card].to('cuda:0'))/2
        return mean_out.data.cpu().numpy()

    def prune_filters_by_AAP(
        self, 
        resample=False, 
        reinit=False, 
        SAVE_NAME="", 
        pruning_round=-1, 
        args="",
        conv_threshold=-1,
        power_value=1,
        ):
        
        # calculate total_alive_relu, then calculate std and mean of total_alive_relu
        print("1. Calculate total_alive_relu, then calculate std and mean of total_alive_relu")
        total_alive_relu = []
        for name, p in self.model.named_parameters():
            if 'conv1' == name.split('.')[0] or 'layer' not in name:
                #print(name)
                continue
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue
            
            weights_arr = p.data.cpu().numpy() # conv: (out_channel, in_channel, kernal_size, kernal_size)
            
            # calculate the abs sum of each filter, filers_sum is 1-d array
            if "conv" in name:
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # dimension: (out_channel, )
            
            # find its alive relu_arr
            for module_name, module in self.model.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    
                    relu_arr = module.output.data.cpu().numpy() # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    assert relu_arr.shape[1] == weights_arr.shape[0]
           
                    # calculating std
                    # relu_arr: (bs, out_channel, image_size, image_size)
                    relu_arr_mean_3d = np.mean(relu_arr, axis=0) # (out_channel, image_size, image_size)
                    alive_index = list(np.nonzero(filers_sum)[0])
                    img_arr_org = relu_arr_mean_3d
                    mean_list = [] # store the mean value of the activations of each filter
                    for num_channel in range(0, img_arr_org.shape[0]):
                        img_arr = img_arr_org[num_channel] # image for one filter: (image_size, image_size)
                        img_arr_mean = np.mean(img_arr)
                        mean_list.append(img_arr_mean)
                    assert len(mean_list) == img_arr_org.shape[0]
                    alive_mean_list = list(np.array(mean_list)[alive_index])
                    total_alive_relu = total_alive_relu + alive_mean_list
        
        # calculate std according to total_alive_relu
        std_total_alive_relu = np.std(total_alive_relu, dtype=np.float64)
        mean_total_alive_relu = np.mean(total_alive_relu)
        
        #print("=====================>conv_threshold: {}".format(conv_threshold))

        # Calculate weights of each layer according to its number of parameters or flops
        # Remember to change to following 2 lines:
        # curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
        # curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
        print("2. Calculate weights of each layer according to its number of parameters")
        para_total_nz = 0.0
        flops_total_nz = 0.0
        threshold_weights_dict_para = {}
        #threshold_weights_dict_flops = {}
        for name, module in self.model.named_modules():
            if 'conv1' == name.split('.')[0] or 'downsample' in name or 'layer' not in name:
                #print(name)
                continue
            if "conv" in name and 'mask' not in name and 'bn' not in name:
                # find relu's conresponding conv layer
                for parameter_name, p in self.model.named_parameters():
                    if "conv" in parameter_name and 'mask' not in parameter_name:
                        if name.split("_")[0] == parameter_name.replace(f'.{parameter_name.split(".")[-1]}',''):
                            # calculate parameters
                            tensor = p.data.cpu().numpy()
                            para_curr_nz = np.count_nonzero(tensor)
                            para_total_nz = para_total_nz + para_curr_nz
                            threshold_weights_dict_para[name]=para_curr_nz
                            
        for k,v in threshold_weights_dict_para.items():
            threshold_weights_dict_para[k] = v / para_total_nz
        #for k,v in threshold_weights_dict_flops.items():
        #    threshold_weights_dict_flops[k] = v / flops_total_nz
        
        assert round(sum(threshold_weights_dict_para.values()), 2) == 1.0
        #assert round(sum(threshold_weights_dict_flops.values()), 2) == 1.0

        # prune filters in each layer
        print("3. Prune filters in each layer")
        save_content_threshold=f"pruning_round = {pruning_round}\t"
        with open('outputs.pkl', 'rb') as file:
                outputs = pickle.load(file)
        for name, module in self.model.named_modules():
            if 'conv1' == name.split('.')[0] or 'downsample' in name or 'layer' not in name:
                #print(name)
                continue
            if "conv" in name and 'mask' not in name and 'bn' not in name:
                #relu_arr = module.output.data.cpu().numpy()
                relu_arr = self.process_file(name, outputs)
                relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                
                assert name in threshold_weights_dict_para.keys() 
                #assert name in threshold_weights_dict_flops.keys() 
                curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
                # curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
                prune_index = np.where(relu_arr_mean <= curr_conv_threshold)[0] 
                #print(f"curr_conv_threshold: {curr_conv_threshold}")
        
                #print(f"conv_threshold: {conv_threshold}, power_value: {power_value}")
                #print("relu_arr_mean.shape: {}, average of relu_arr_mean: {}".format(relu_arr_mean.shape, np.mean(relu_arr_mean)))

                # save relu_arr_mean of current layer
                metrics_relu_arr_mean = [np.std(relu_arr_mean, dtype=np.float64), np.mean(relu_arr_mean), np.min(relu_arr_mean), np.max(relu_arr_mean)]
                save_content_threshold = save_content_threshold + f"{name}={metrics_relu_arr_mean}\t"

                #print("====>Begin to find the next conv module, and put mask on input channel")
                self.prune_next_conv(
                    curr_name=name.split("_")[0], 
                    prune_index=prune_index, 
                    resample=resample, 
                    reinit=reinit)

                #print("====>Begin to find the fc or conv module, and prune filters (put mask on output channel)")
                for _name, _module in self.model.named_modules():
                    if name.split("_")[0] == _name:
                        #print(prune_index)
                        _module.prune_filters(prune_index=prune_index, resample=resample, reinit=reinit)
                        # calculate remaining filters to test inference time
                        #print("=====>Save number of remaining filters for inference time")
                        assert pruning_round != -1
                        #save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1)
        
        metric_list = [std_total_alive_relu, mean_total_alive_relu]
        return metric_list    


def get_optimizer(
    parameters,
    fp16,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    bn_weight_decay=False,
    ):

    print('getting optimizer lr', lr) 
    return torch.optim.SGD([v for n, v in parameters], lr=lr, momentum=0.9, weight_decay=1e-4)

def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        print('current lr', lr)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(model_and_loss, optimizer):

    def _zero_gradients():
        # zero-out all the gradients corresponding to the pruned connections
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_and_loss.get_model()
        for name, p in model.named_parameters():
            if 'mask' in name or "relu_para" in name:  # add relu_para
                continue

            tensor = p.data
            grad_tensor = p.grad.data
            grad_tensor = torch.where(tensor == 0.0, torch.Tensor([0.0]).to(device), grad_tensor)
            p.grad.data = grad_tensor

    def _step(input, target):
        optimizer.zero_grad()
        with autocast():
            loss, output = model_and_loss(input, target)
            reduced_loss = loss.data
            #loss.backward()
        scaler.scale(loss).backward()
        _zero_gradients()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        return reduced_loss

    return _step


def train(
    train_loader,
    model_and_loss,
    optimizer,
    lr_scheduler,
    logger,
    epoch):

    step = get_train_step(model_and_loss, optimizer)

    model_and_loss.train()

    optimizer.zero_grad() 

    data_iter = enumerate(train_loader)
    #print('train data length', len(train_loader))
    for i, (input, target) in data_iter:
        #print('current', i)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        lr_scheduler(optimizer, i, epoch) 
        loss = step(input, target)
        #print(target, loss)

def get_val_step(model_and_loss):
    def _step(input, target):
        loss, output = model_and_loss(input, target)
        #prec1 = utils.accuracy(output.data, target, topk=(1, 5))
        _, predicted = torch.max(output.data, 1)
        return (predicted == target).sum().item()

    return _step

def validate(val_loader, model_and_loss, logger, epoch):
    #model = model_and_loss.get_model()
    #model.eval()
    model_and_loss.eval()
    total = 0
    correct = 0
    
    start = time.time()
    #print('test data', len(val_loader))
    with autocast():
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                #print('current', i)
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                _, z = model_and_loss(input, target)        
                _, predicted = torch.max(z.data, 1)
                total += input.size(0)
                correct += (predicted == target).sum().item()
                current_time = time.time() - start
                #print('current time', current_time)
    torch.cuda.empty_cache()  
    return correct / total * 100

def write_to_log(fname, text):
    f = open(fname, "a")
    out_text = "[%s]\t%s" % (str(datetime.now()), text)
    f.write(out_text + "\n")
    f.close()


def train_loop(
    model_and_loss,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    fp16,
    logger,
    should_backup_checkpoint,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    rewind_filename=None,
    rewind_epoch=-1,
    log_filename="",
    remained_per=None,
    compress_ratio=None,
    flops_speedup=None,
    flops_remain=None,
    args = "", # for saving weights of the last round
    curr_round=-1, # for saving weights of the last round
):

    prec1 = -1
    #print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    
    for epoch in range(start_epoch, end_epoch):
        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            train(
                train_loader,
                model_and_loss,
                optimizer,
                lr_scheduler,
                logger,
                epoch
            )
        if not skip_validation:
            prec1 = validate(
                val_loader,
                model_and_loss,
                logger,
                epoch
            )
            print('current accuracy', prec1)
        
        if save_checkpoints: #and dist.get_rank() == 0:
            if not skip_validation:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
            else:
                is_best = False
                best_prec1 = 0

            if should_backup_checkpoint(epoch):
                backup_filename = "checkpoint-{}.pth.tar".format(epoch)
            else:
                backup_filename = None
            
            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": model_and_loss.arch,
                    "state_dict": model_and_loss.get_model().state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename,
                filename=checkpoint_filename,
            )
            
            # save rewinding weights
            if rewind_filename and epoch == rewind_epoch:
                utils.save_checkpoint(
                    {
                        "epoch": epoch,
                        "arch": model_and_loss.arch,
                        "state_dict": model_and_loss.get_model().state_dict(),
                        "best_prec1": best_prec1,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    checkpoint_dir=checkpoint_dir,
                    backup_filename=None,
                    filename=rewind_filename,
                )
            
    return prec1


