import logging
from pathlib import Path
from model.train import train as train_encoder_encoder
from helpers import init_helper, data_helper
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
logger = logging.getLogger()
TRAINER = {
    'encoder-decoder': train_encoder_encoder
}

def get_trainer(model_type):
    assert model_type in TRAINER
    return TRAINER[model_type]


def main():
    args = init_helper.get_arguments()
    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(args.model)

    #保存ymal
    data_helper.dump_yaml(vars(args), model_dir / 'args.yml')


    for split_path in args.splits:


        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        results = {}
        stats = data_helper.AverageMeter('fscore')
        test_mean_fs = []
        train_mean_fs = []
        for split_idx, split in enumerate(splits):
            # print('idx--',split)
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')

            ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)

            fscore, model = trainer(args, split, ckpt_path, split_idx)
            test_mean_fs.append(fscore)


            stats.update(fscore=fscore)
            results[f'split{split_idx}'] = float(fscore)


        results['mean'] = float(stats.fscore)
        data_helper.dump_yaml(results, model_dir / '{split_path.stem}.yml')

        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}')








if __name__ == '__main__':
    main()
