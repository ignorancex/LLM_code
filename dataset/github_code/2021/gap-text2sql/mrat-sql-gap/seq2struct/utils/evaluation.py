import json
import os

import _jsonnet

from seq2struct import datasets
from seq2struct.utils import registry

def compute_metrics(config_path, config_args, section, inferred_path,logdir=None):
    if config_args:
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))

    print(f"Eval Dataset val(data val paths):{config['data']['val']['paths']}")
    print(f"Eval Dataset val(data val tables_paths):{config['data']['val']['tables_paths']}\n")

    if 'model_name' in config and logdir:
        logdir = os.path.join(logdir, config['model_name'])
    if logdir:
        inferred_path = inferred_path.replace('__LOGDIR__', logdir)

    inferred = open(inferred_path, encoding='utf8')
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data)

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        raise Exception('Not enough inferred: {} vs {}'.format(len(inferred_lines),
          len(data)))


    for line in inferred_lines:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        if 'index' in infer_results:
            metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    return logdir, metrics.finalize()
    
def compute_metrics2(config_path, config_args, section, inferred_path, val_data_path, logdir=None):
    if config_args:
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))
    #use the command line validation data path    
    config['data']['val']['paths'][0] = val_data_path + "dev.json"
    config['data']['val']['tables_paths'][0] = val_data_path + "tables.json"
    
    print(f"Eval Dataset val(data val paths):{config['data']['val']['paths']}")
    print(f"Eval Dataset val(data val tables_paths):{config['data']['val']['tables_paths']}\n")

    if 'model_name' in config and logdir:
        logdir = os.path.join(logdir, config['model_name'])
    if logdir:
        inferred_path = inferred_path.replace('__LOGDIR__', logdir)

    inferred = open(inferred_path, encoding='utf8')
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data)

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        raise Exception('Not enough inferred: {} vs {}'.format(len(inferred_lines),
          len(data)))


    for line in inferred_lines:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        if 'index' in infer_results:
            metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    return logdir, metrics.finalize()