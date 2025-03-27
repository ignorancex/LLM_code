import argparse
import json

import _jsonnet
import attr

from seq2struct.utils import evaluation

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', required=True)
    parser.add_argument('--inferred', required=True)
    parser.add_argument('--output')
    parser.add_argument('--logdir')
    args = parser.parse_args()
    return args

def main(args):
    real_logdir, metrics = evaluation.compute_metrics(args.config, args.config_args, args.section, args.inferred, args.logdir)
    #print(f"\nargs.config={args.config}\nargs.config_args={args.config_args}\nargs.section={args.section}\nargs.inferred={args.inferred}\nargs.logdir={args.logdir}\n")#teste para descobrir como fazer evaluation do Spider
    #print(f"\nreal_logdir={real_logdir}\nmetrics={metrics}\n")#teste para descobrir como fazer evaluation do Spider
    if args.output:
        if real_logdir:
            output_path = args.output.replace('__LOGDIR__', real_logdir)
        else:
            output_path = args.output
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(metrics, f, ensure_ascii=False)
        print('Wrote eval results to {}'.format(output_path))
    else:
        print(metrics)

def main2(args, val_data_path):
    real_logdir, metrics = evaluation.compute_metrics2(args.config, args.config_args, args.section, args.inferred, val_data_path, args.logdir)
    #print(f"\nargs.config={args.config}\nargs.config_args={args.config_args}\nargs.section={args.section}\nargs.inferred={args.inferred}\nargs.logdir={args.logdir}\n")#teste para descobrir como fazer evaluation do Spider
    #print(f"\nreal_logdir={real_logdir}\nmetrics={metrics}\n")#teste para descobrir como fazer evaluation do Spider
    if args.output:
        if real_logdir:
            output_path = args.output.replace('__LOGDIR__', real_logdir)
        else:
            output_path = args.output
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(metrics, f, ensure_ascii=False)
        print('Wrote eval results to {}'.format(output_path))
    else:
        print(metrics)

if __name__ == '__main__':
    args = add_parser()
    main(args)
