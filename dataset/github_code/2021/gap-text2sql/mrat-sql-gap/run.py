#!/usr/bin/env python

import _jsonnet
import json
import argparse
import collections
import attr
from seq2struct.commands import preprocess, train, infer, eval
import crash_on_ipy

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()

@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()

@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)

@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval")
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    args = parser.parse_args()
        
    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None
    other_config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
    
    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, \
            model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, 
            model_config_args, exp_config["logdir"]) 
        train.main(train_config)
    elif args.mode == "eval":
        result = open(f"{exp_config['eval_output']}/eval-results.csv", "a", encoding='utf8')
        result.write(f"checkpoint;type;easy;medium;hard;extra;all\n") 
        result.close()
        first_loop = True

        #File with gold queries from dev.json
        gold = open(f"{exp_config['eval_output']}/gold.txt", "w", encoding='utf8')
        print(f"Open file {other_config['data']['val']['paths'][0]}")
        with open(f"{other_config['data']['val']['paths'][0]}", encoding='utf8') as json_data_file:
            data = json.load(json_data_file)
            length = len(data) #tive que fazer pelo tamanho porque o arquivo .json começa com [ em branco ]    
            for i in range(length):
                gold.write(f"{data[i]['query']}\t{data[i]['db_id']}\n")
        json_data_file.close()
        gold.close()        
      
        for step in exp_config["eval_steps"]:
            infer_output_path = "{}/{}-step{}".format( #infer_output_path = "{}/{}-step{}.infer".format(
                exp_config["eval_output"], 
                exp_config["eval_name"], 
                step)
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                exp_config["logdir"],
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            infer.main(infer_config)

            eval_output_path = "{}/{}-step{}.eval".format(
                exp_config["eval_output"], 
                exp_config["eval_name"], 
                step)
            eval_config = EvalConfig(
                model_config_file,
                model_config_args,
                exp_config["logdir"],
                exp_config["eval_section"],
                f"{infer_output_path}.infer",
                eval_output_path
            )
            eval.main(eval_config)

            res_json = json.load(open(eval_output_path))
            print(step, res_json['total_scores']['all']['exact'])
            print(f"*;count;{res_json['total_scores']['easy']['count']};{res_json['total_scores']['medium']['count']};{res_json['total_scores']['hard']['count']};{res_json['total_scores']['extra']['count']};{res_json['total_scores']['all']['count']}") 
            print(f"checkpoint;type;easy;medium;hard;extra;all") 
            print(f"{step};exact match;{res_json['total_scores']['easy']['exact']:.3f};{res_json['total_scores']['medium']['exact']:.3f};{res_json['total_scores']['hard']['exact']:.3f};{res_json['total_scores']['extra']['exact']:.3f};{res_json['total_scores']['all']['exact']:.3f}") 
            
            #Open, write, close - to leave memory free
            result = open(f"{exp_config['eval_output']}/eval-results.csv", "a", encoding='utf8')            
            if first_loop == True: 
                result.write(f"*;count;{res_json['total_scores']['easy']['count']};{res_json['total_scores']['medium']['count']};{res_json['total_scores']['hard']['count']};{res_json['total_scores']['extra']['count']};{res_json['total_scores']['all']['count']}\n") 
            first_loop = False   
            result.write(f"{step};exact match;{res_json['total_scores']['easy']['exact']:.3f};{res_json['total_scores']['medium']['exact']:.3f};{res_json['total_scores']['hard']['exact']:.3f};{res_json['total_scores']['extra']['exact']:.3f};{res_json['total_scores']['all']['exact']:.3f}\n") 
            result.close()
            
            #Clean version of original .eval file
            eval_clean = open(f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.csv", "w", encoding='utf8')
            for per_item in res_json['per_item']:  
                if per_item['exact'] == 0 or per_item['exact'] == "false": exact = "false" #in original .eval file some appear as 0 others as "false"
                if per_item['exact'] == 1 or per_item['exact'] == "true": exact = "true" #in original .eval fiel all appear as "true", but I did the same to be standard 
                eval_clean.write(f"{exact};{per_item['hardness']};{per_item['gold']};{per_item['predicted']}\n")                                   
            eval_clean.close() 

if __name__ == "__main__":
    main()