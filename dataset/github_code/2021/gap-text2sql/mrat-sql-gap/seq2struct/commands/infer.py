import argparse
import ast
import itertools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm

from seq2struct import beam_search
from seq2struct import datasets
from seq2struct import models
from seq2struct import optimizers
from seq2struct.utils import registry
from seq2struct.utils import saver as saver_mod

from seq2struct.models.spider import spider_beam_search

class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)
#        self.device = torch.device('cpu')
#        torch.set_num_threads(1)


        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])

        if not last_step:
            raise Exception('Attempting to infer on untrained model')
        return model

    def infer(self, model, output_path, args):
        output = open(f"{output_path}.infer", 'w', encoding='utf8')
        output_clean = open(f"{output_path}.json", 'w', encoding='utf8')
        predicted_txt = open(f"{output_path}.txt", "w", encoding='utf8')
        with torch.no_grad():
            if args.mode == 'infer':
                orig_data = registry.construct('dataset', self.config['data'][args.section])
                preproc_data = self.model_preproc.dataset(args.section)
                #print(f"args.section={args.section}")
                if args.limit:
                    sliced_orig_data = itertools.islice(orig_data, args.limit)
                    sliced_preproc_data = itertools.islice(preproc_data, args.limit)
                    #print(f"args.limit preproc_data={preproc_data}\n\nargs.limit={args.limit}")
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                    #print(f"no args.limit preproc_data={preproc_data}\n\n")
                print(f"Orig_data={len(orig_data)} Preproc_data={len(preproc_data)} Beam_size={args.beam_size} Use_heuristic={args.use_heuristic}")
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data, output, output_clean, predicted_txt,args.use_heuristic)
            elif args.mode == 'debug':
                data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._debug(model, sliced_data, output)
            elif args.mode == 'visualize_attention':
                model.visualize_flag = True
                model.decoder.visualize_flag = True
                data = registry.construct('dataset', self.config['data'][args.section])
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._visualize_attention(model, args.beam_size, args.output_history, sliced_data, args.res1, args.res2, args.res3, output)
        output_clean.close()
        predicted_txt.close()        

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, use_heuristic=True):
        if use_heuristic:
            # TODO: from_cond should be true from non-bert model
            #print(f"_infer_one\nmodel={model}\ndata_item={data_item}\npreproc_item={preproc_item}\nbeam_size={beam_size}");
            #print(f"\npreproc_item={preproc_item}\n");
            beams = spider_beam_search.beam_search_with_heuristics(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
        else:
            beams = beam_search.beam_search(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000)
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'orig_question': data_item.orig["question"],
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output, output_clean, predicted_txt, use_heuristic=False):
        decoded_clean = []
        for i, (orig_item, preproc_item) in enumerate(
                tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data),
                          total=len(sliced_orig_data))):
            if use_heuristic:
                #TODO: from_cond should be true from non-bert model
                #if i>=0 and i<3:
                #    print(f"\npreproc_item={preproc_item}\n");
                #    print(f"_inner_infer\nmodel={model}\ndata_item={orig_item}\npreproc_item={preproc_item}\nbeam_size={beam_size}");
                beams = spider_beam_search.beam_search_with_heuristics(
                    model, orig_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
            else:
                beams = beam_search.beam_search(
                   model, orig_item, preproc_item, beam_size=beam_size, max_steps=1000)
            
            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()
                
                decoded.append({
                    'orig_question': orig_item.orig["question"], 
                    'model_output': model_output,
                    'inferred_code': inferred_code,
                    'score': beam.score,
                    **({
                        'choice_history': beam.choice_history,
                        'score_history': beam.score_history,
                    } if output_history else {})})
                                                

            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }, ensure_ascii=False) + '\n')
            output.flush()
            
            #output_clean.write(json.dumps(decoded_clean, ensure_ascii=False) + '\n')
            #output_clean.flush()
            decoded_clean.append ({'orig_question': orig_item.orig["question"],'predicted': inferred_code})
            predicted_txt.write(f"{inferred_code}\n")
        json.dump(decoded_clean, output_clean, indent=4, ensure_ascii=False)


    def _debug(self, model, sliced_data, output):
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
            (_, history), = model.compute_loss([item], debug=True)
            output.write(
                    json.dumps({
                        'index': i,
                        'history': history,
                    }, ensure_ascii=False) + '\n')
            output.flush()

    def _visualize_attention(self, model, beam_size, output_history, sliced_data, res1file, res2file, res3file, output):
        res1 = json.load(open(res1file, 'r', encoding='utf8'))
        res1 = res1['per_item']
        res2 = json.load(open(res2file, 'r', encoding='utf8'))
        res2 = res2['per_item']
        res3 = json.load(open(res3file, 'r', encoding='utf8'))
        res3 = res3['per_item']
        interest_cnt = 0
        cnt = 0
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
        
            if res1[i]['hardness'] != 'extra':
                continue
        
            cnt += 1
            if (res1[i]['exact'] == 0) and (res2[i]['exact'] == 0) and (res3[i]['exact'] == 0):
                continue
            interest_cnt += 1
            '''
            print('sample index: ')
            print(i)
            beams = beam_search.beam_search(
                model, item, beam_size=beam_size, max_steps=1000, visualize_flag=True)
            entry = item.orig
            print('ground truth SQL:')
            print(entry['query_toks'])
            print('prediction:')
            print(res2[i])
            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()

                decoded.append({
                    'model_output': model_output,
                    'inferred_code': inferred_code,
                    'score': beam.score,
                    **({
                        'choice_history': beam.choice_history,
                        'score_history': beam.score_history,
                    } if output_history else {})})

            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()
            '''
        print(interest_cnt * 1.0 / cnt)


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--step', type=int)
    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug', 'visualize_attention'])
    parser.add_argument('--use_heuristic', action='store_true')
    parser.add_argument('--res1', default='outputs/glove-sup-att-1h-0/outputs.json')
    parser.add_argument('--res2', default='outputs/glove-sup-att-1h-1/outputs.json')
    parser.add_argument('--res3', default='outputs/glove-sup-att-1h-2/outputs.json')
    args = parser.parse_args()
    return args

def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])
    
    output_path = args.output.replace('__LOGDIR__', args.logdir)
    output_path_teste = f"{output_path}.infer"
    if os.path.exists(output_path_teste):
        print('Output file {} already exists'.format(output_path_teste))
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)
    
    
def main2(args, val_data_path):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    #Pode sobrepor o anterior
    #if os.path.exists(output_path):
    #    print('Output file {} already exists'.format(output_path))
    #    sys.exit(1)

    #use the command line validation data path
    config['data']['val']['paths'][0] = val_data_path + "dev.json"
    config['data']['val']['tables_paths'][0] = val_data_path + "tables.json"
    
    print(f"Infer Dataset val(data val paths):{config['data']['val']['paths']}")
    print(f"Infer Dataset val(data val tables_paths):{config['data']['val']['tables_paths']}\n")

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == '__main__':
    args = add_parser()
    main(args)
