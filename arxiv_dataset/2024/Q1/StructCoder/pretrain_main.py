import warnings
from pretrain_utils import *
from torch.optim import AdamW
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_value_
    
def main():
    args = parse_args()
    set_seed(args.seed)
    set_output_dir(args)
    args.logger = Logger(args.output_dir, filename='log.txt')
    args.logger.write(args)
    set_tokenizer(args)
    args.device = torch.device('cuda')
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin') # to save ckpt with best validation result
    model_path_last = os.path.join(args.output_dir, 'checkpoint_last.bin')
    
    # load data
    read_data(args)
    num_samples = len(args.data)
    args.logger.write('\nPreparing pretraining IO for validation set')
    valid_io, valid_langs = prepare_io(np.arange(num_samples-args.num_valid_samples, num_samples), args)
    
    # load model
    model = load_model(args)
    
    # training loop
    cum_train_loss, num_steps = 0, 0
    train_cycler = CycleIndex(num_samples-args.num_valid_samples, args.train_batch_size)
    wait, best_val_metric = args.patience, -np.inf # higher val_metric is better
    optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                                           num_training_steps=200000)
    train_bar = tqdm(range(args.max_steps))
    patience_reached = False
    task_loss_weights = np.array([[weight1*weight2 for weight2 in [1, args.dfg_loss_weight, args.ast_loss_weight]] 
                                     for weight1 in [1, args.text2code_loss_weight, args.code2text_loss_weight]])
    args.logger.write('\nTask loss weights:')
    args.logger.write(task_loss_weights)
    
    # Resume from ckpt if args.resume=1. Rewrite necessary variables and set the right seeds.
    if args.resume==1:
        args.logger.write('Resuming training from ckpt at '+model_path_last)
        checkpoint = torch.load(model_path_last)
        model.load_state_dict(checkpoint['model_weights'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        train_cycler = checkpoint['train_cycler']
        wait, best_val_metric = checkpoint['wait'], checkpoint['best_val_metric']
        train_bar = tqdm(range(checkpoint['step']+1, args.max_steps))
        set_rng_states(checkpoint['random_rng_state'], checkpoint['np_rng_state'], \
                       checkpoint['torch_rng_state'], checkpoint['torch_cuda_rng_state'])
        del checkpoint
#     else:
#         # Get initial performance on validation set
#         args.logger.write('\nInitial performance on validation set i.e. before any training')
#         test(model, valid_io, valid_langs, args, -1)
    
    model.train()
    for step in train_bar:
        # load batch
        batch_io = get_batch(train_cycler.get_batch_ind(), args)
        # forward pass 
        ret_denoising = model(inputs=batch_io['corrupt_code_inputs'], outputs=batch_io['code_outputs']) # denoising
        ret_text2code = model(inputs=batch_io['text_inputs'], outputs=batch_io['code_outputs']) # text2code
        if args.code2text_loss_weight>0:
            ret_code2text = model(inputs=batch_io['code_inputs'], outputs=batch_io['text_outputs']) # code2text
        else:
            ret_code2text = {'lm_loss': torch.tensor(0.0).to(args.device)}
        
        # compute loss
        losses = [[ret[k].mean() for k in ['lm_loss','dfg_loss','ast_loss']] for ret in [ret_denoising, ret_text2code]] \
                    + [[ret_code2text['lm_loss'].mean(), 0, 0]] # 3x3
        loss = losses[2][0] * task_loss_weights[2,0]
        for main_task in range(2):
            for sub_task in range(3):
                loss += losses[main_task][sub_task] * task_loss_weights[main_task,sub_task]

        # backward pass
        loss.backward()
        clip_grad_value_(model.parameters(), args.clip_grad_max)
        if (step+1)%args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # add to cum loss
        cum_train_loss += np.array([[losses[main_task][sub_task].item() for sub_task in range(3)] for main_task in range(2)]\
                                   +[[losses[2][0].item(), 0, 0]])
        num_steps += 1
        
        # Log training losses.
        train_bar.set_description(str(round(loss.item(),5)))
        if (step+1)%args.print_train_loss_every == 0:
            args.logger.write('\nTrain-loss at step '+str(step))
            args.logger.write(cum_train_loss/num_steps, show_time=False)
            cum_train_loss, num_steps = 0,0
            
        # validate
        if (step+1)%args.validate_every==0:
            # get metrics on validation set
            results = test(model, valid_io, valid_langs, args, train_step=step)
            model.train()          
            
            # Save model and validation predictions if there is an improvement.
            curr_val_metric = sum(results['BLEU'].values())
            if curr_val_metric>best_val_metric:
                best_val_metric = curr_val_metric
                wait = args.patience
                args.logger.write('\nSaving ckpt at ' + model_path_best)
                save_ckpt(model_path_best, model, optimizer, step, train_cycler, 
                          wait, best_val_metric)
            elif wait>1:
                wait -= 1
                
            args.logger.write('Wait : '+str(wait))
            if wait==0:
                patience_reached = True
            
        # Save lastest ckpt.
        if (step+1)%args.validate_every==0:
            args.logger.write('\nSaving ckpt at ' + model_path_last)
            save_ckpt(model_path_last, model, optimizer, step, train_cycler, 
                          wait, best_val_metric)
        
        # Save best checkpoint till now as it will be overwritten in future.
        if (step+1)%args.checkpoint_every == 0:
            if os.path.exists(model_path_best):
                args.logger.write('\nSaving best checkpoint until step '+str(step+1))
                os.rename(model_path_best, model_path_best.replace('_best', '_best_at_'+str(step+1)))
            else:
                args.logger.write('\nBest checkpoint until step '+str(step+1)+' same as at step ' \
                                  +str(step+1-args.checkpoint_every))
        
        # stop if patience is reached
        if patience_reached:
            break
            
                          

main()