from finetune_translation_utils import *
from pretrain_utils import set_seed, Logger, set_tokenizer, CycleIndex, save_ckpt
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm


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
    data_by_split = read_data(args)
    train_data, valid_data, test_data = data_by_split['train'], data_by_split['validation'], data_by_split['test']
    
    # load model
    model = load_model(args, 'data/codexglue_translation/all_node_types.pkl')
    
    # training loop
    if args.do_train:
        cum_train_loss, num_steps, num_train = 0, 0, len(train_data['code_inputs']['input_ids'])
        train_cycler = CycleIndex(num_train, args.train_batch_size)
        wait, best_val_metric = args.patience, -np.inf # higher val_metric is better
        optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
        train_bar = tqdm(range(args.max_steps))
        patience_reached = False

        model.train()
        for step in train_bar:
            # load batch
            batch_io = get_batch(train_data, train_cycler.get_batch_ind(), args)

            # forward pass 
            ret = model(inputs=batch_io['code_inputs'], outputs=batch_io['code_outputs'])
            lm_loss, dfg_loss, ast_loss = ret.lm_loss.mean(), ret.dfg_loss.mean(), ret.ast_loss.mean()
            loss = lm_loss + args.dfg_loss_weight*dfg_loss + args.ast_loss_weight*ast_loss

            # backward pass
            loss.backward()
            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # add to cum loss
            cum_train_loss += np.array([lm_loss.item(), dfg_loss.item(), ast_loss.item()])
            num_steps += 1

            # Log training losses.
            train_bar.set_description(str(round(loss.item(),5)))
            if (step+1)%args.print_train_loss_every == 0:
                args.logger.write('\nTrain-loss at step '+str(step))
                args.logger.write(cum_train_loss/num_steps, show_time=False)
                cum_train_loss, num_steps = 0,0

            # validate
            if ((step+1)>=args.validate_after) and ((step+1)%args.validate_every==0):
                # get metrics on validation set
                results = test(model, valid_data, args, train_step=step, split='validation')
                test(model, test_data, args, train_step=step, split='test')
                model.train()          

                # Save model and validation predictions if there is an improvement.
                curr_val_metric = results['CodeBLEU'][-1] + results['xMatch']
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
            
    if args.do_test:
        results = test(model, test_data, args, train_step=step, split='test')
            
    
main()