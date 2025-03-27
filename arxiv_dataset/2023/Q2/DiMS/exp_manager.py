import sys
import json
import os

defualt_config = {
    'arch': 'cmlm_distill',
    'source-lang': 'de',
    'target-lang': 'en',
    'optimizer': 'adam',
    'adam-betas': "'(0.9,0.98)'",
    'criterion': 'nat_loss',
    'task': 'translation_lev',
    'label-smoothing': '0.1',
    'noise': 'random_mask',
    'lr-scheduler': 'inverse_sqrt',
    'warmup-init-lr': '1e-07',
    'lr': '1e-4',
    'warmup-updates': '1',
    'dropout': '0.3',
    'weight-decay': '0.01',
    'decoder-learned-pos': True,
    'encoder-learned-pos': True,
    'apply-bert-init': True,
    'share-all-embeddings': True,
    'max-tokens': '8192',
    'fixed-validation-seed': '7',
    'fp16': True,
    'batch-size-valid': '2048',
    'validate-interval': '1', 
    'batch-size-valid': '2048',
    'validate-interval': '1',
    'num-workers': '10',
    'no-epoch-checkpoints': True,
    'keep-best-checkpoints': '1',
    'teacher-path': 'results/checkpoints/IWSLTdeen_distill_CMLM_benchmark/checkpoint_best.pt',
    'teacher-ema': False,
    'teacher-ema-decay': '0.9997',
    'step-count': '2',
    'step-weight-update': '0',
    'step-weight-temp': '1',
    'eval-bleu': True,
    'eval-bleu-args': "'{\"iter_decode_max_iter\": 0, \"iter_decode_force_max_iter\": true}'",
    'eval-bleu-remove-bpe': True,
    'best-checkpoint-metric': 'bleu',
    'maximize-best-checkpoint-metric': True ,
    'eval-bleu-detok': 'moses', 
    'patience': '50',
    'mid-mask-policy': "half",
    'revealed-loss': False,  
    'beam-length': False,  
    'beam-sample': False,  
    'optimize-length-predictor': False,  
    'dataset': 'data-bin'
}

def main():
    os.system(f"rm -f done.bin")
    name = sys.argv[1]
    gpu_num = sys.argv[2]
    try:
        progressive = sys.argv[3]=='progressive'
    except:
        progressive = False
    with open(f'ExpSetting/{name}.json', 'r') as config_file:
        config_changes=json.load(config_file)
        max_steps=3
        step=0
        while step<max_steps:
            if int(config_changes['step-count'])==0:
                break
            step_count=config_changes.get('step-count', -1)
            config_changes['eval-bleu-args']= "'{\"iter_decode_max_iter\":" + (str(int(config_changes['step-count'])-1) if int(step_count)!=-1 else "0") +  ", \"iter_decode_force_max_iter\": true}'"
            retval=single_run(config_changes, name, gpu_num)
            if retval!=0 or not progressive:
                exit(retval)
            if step_count!=-1:
                config_changes['step-count']=str(int(config_changes['step-count'])//2) 
            config_changes['teacher-path']=f"{config_changes['save-dir']}/checkpoint_best.pt"
            name+=f"to{config_changes['step-count']}"
            step+=1


def single_run(config_changes, name, gpu_num):
    
    config=defualt_config
    command = 'fairseq-train'
    config_changes['save-dir'] =  f'./results/distillation/checkpoints/IWSLTdeen_distill_CMLM_benchmark_{name}'
    config_changes['log-file'] =  f'./results/distillation/checkpoints/IWSLTdeen_distill_CMLM_benchmark_{name}/log.txt'
    config_changes['tensorboard-logdir'] =  config_changes['save-dir']
    config.update(config_changes)
    dataset=config.pop("dataset")
    options = ''
    for key, val in config.items():
        if val == True:
            options += f' --{key}'
        elif val != False:
            options += f' --{key} {val}'
    retval = os.system(f'[ ! -d {config_changes["save-dir"]} ] && mkdir {config_changes["save-dir"]}')
    retval = os.system(f'CUDA_VISIBLE_DEVICES={gpu_num} fairseq-train {dataset} {options}')
    os.system(f"touch done.bin")
    if retval==0:
        with open(f'{config["save-dir"]}/config.json', 'w') as outfile:
            json.dump(json.dumps(config), outfile)
        # for step in [0, 1, 3, 7]:
        #    for beam in [1, 5]:
        #        eval_command=f'CUDA_VISIBLE_DEVICES={gpu_num} ./eval_model.sh {dataset} {config["save-dir"]}/checkpoint_best.pt {step} {beam} >> {config["save-dir"]}/eval.txt'
        #        print(eval_command)
        #        retval = os.system(eval_command)

    return retval

if __name__ == '__main__':
    main()
