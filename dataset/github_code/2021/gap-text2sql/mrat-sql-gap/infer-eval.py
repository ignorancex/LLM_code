#!/usr/bin/env python

import _jsonnet
import json
import argparse
import collections
import attr
from seq2struct.commands import preprocess, train, infer, eval
import crash_on_ipy
import os
import sys

print(f"python verson{sys.version}")


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
    parser.add_argument('data_file', help="Input data json file, ex.: data/")
    parser.add_argument('path_output_file', help="Path to output json file with prediction")
    args = parser.parse_args()
    
    #print(args)
    #print(args.data_file)
    #print(args.path_output_file)
    args.path_output_file = os.path.splitext(args.path_output_file)[0]#remove the extension, for further inclusion (.infer, .txt, .json)
    #print(args.path_output_file)
    
    if not os.path.exists(args.data_file + "dev.json"):
        print(f"\n{args.data_file}dev.json not found.")
        sys.exit(1)
        
    if not os.path.exists(args.data_file + "tables.json"):
        print(f"\n{args.data_file}tables.json not found.")
        sys.exit(1)
 
    
    #Codalab Run-time directory structure
    os.makedirs("preprocess", exist_ok=True)
    os.makedirs("preprocess/spider-en", exist_ok=True)
    os.makedirs("preprocess/spider-en/mT5-large-NoGAP-nl2code-1115,output_from=true,fs=2,emb=t5,cvlink", exist_ok=True)
    os.makedirs("preprocess/spider-en/mT5-large-NoGAP-nl2code-1115,output_from=true,fs=2,emb=t5,cvlink/dec", exist_ok=True)
    os.makedirs("preprocess/spider-en/mT5-large-NoGAP-nl2code-1115,output_from=true,fs=2,emb=t5,cvlink/enc", exist_ok=True)
    print("Run-time directory structure created")
        
    #infer_config["model"]["encoder_preproc"]["db_path"] = "spider/database"
    #print(infer_config)
    #print(model_config_args)
    

    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('mode', help="preprocess/train/eval")
#    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
#    args = parser.parse_args()
    
#    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    #run.py
    exp_config = json.loads(_jsonnet.evaluate_file("experiments/spider-configs/spider-mT5-large-NoGAP-FIT-en-pt-es-fr-train_en-eval.jsonnet"))
    model_config_file = exp_config["model_config"]
    model_config_args = json.dumps(exp_config["model_config_args"])
    infer_config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
#    if "model_config_args" in exp_config:
#        model_config_args = json.dumps(exp_config["model_config_args"])
#    else:
#        model_config_args = None
    #meu prediction.py
#    exp_config = json.loads(_jsonnet.evaluate_file("experiments/spider-configs/spider-mT5-large-NoGAP-FIT-en-pt-es-fr-train_en-eval.jsonnet"))   
#    model_config_file = exp_config["model_config"]
#    model_config_args = exp_config.get("model_config_args")
#    infer_config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': json.dumps(model_config_args)}))


    print(f"\nDatabase(model encoder_preproc db_path):{infer_config['model']['encoder_preproc']['db_path']}")
    print(type(infer_config['model']['encoder_preproc']['db_path']))
    print(f"Preprocess folder(model encoder_preproc save_path):{infer_config['model']['encoder_preproc']['save_path']}")
    print(f"Preprocess folder(model decoder_preproc save_path):{infer_config['model']['decoder_preproc']['save_path']}\n")
    
    #print(f"Dataset train(data train paths):{infer_config['data']['train']['paths']}")
    #print(f"Dataset train(data train tables_paths):{infer_config['data']['train']['tables_paths']}")
    #print(f"Database(data train db_path):{infer_config['data']['train']['db_path']}\n")
    
    print(f"Dataset val(data val paths):{infer_config['data']['val']['paths']}")
    print(f"Dataset val(data val tables_paths):{infer_config['data']['val']['tables_paths']}")
    print(f"Database(data val db_path):{infer_config['data']['val']['db_path']}\n")   
   
    #print("Update with command line arguments.")
    #infer_config['data']['val']['paths'] = f"['{args.data_file}dev.json']"
    #infer_config['data']['val']['tables_paths'] = f"['{args.data_file}tables.json']"
    
    #print(f"Dataset val(data val paths):{infer_config['data']['val']['paths']}")
    #print(f"Dataset val (data val tables_paths):{infer_config['data']['val']['tables_paths']}")


   
    #"preprocess":
    print("\nPreprocess\n")
    print(f"\n*********************************************\nmodel_config_file={model_config_file}\nmodel_config_args={model_config_args}\n*********************************************\n")
    preprocess_config = PreprocessConfig(model_config_file, model_config_args)
    print(f"\n*********************************************\npreprocess_config={preprocess_config}\n*********************************************\n")
    preprocess.main2(preprocess_config, args.data_file) 

   
    #File with gold queries 
    gold = open(f"gold_for_{args.path_output_file}.txt", "w", encoding='utf8')
    with open(f"{args.data_file}dev.json", encoding='utf8') as json_data_file:
        data = json.load(json_data_file)
        length = len(data) #tive que fazer pelo tamanho porque o arquivo .json começa com [ em branco ]    
        for i in range(length):
            gold.write(f"{data[i]['query']}\t{data[i]['db_id']}\n")
    json_data_file.close()
    gold.close()
    
   #"Infer and Eval"
    #result = open(f"results_for_{args.path_output_file}.csv", "w", encoding='utf8')
    #result.write(f"checkpoint;type;easy;medium;hard;extra;all\n")
    for step in exp_config["eval_steps"]:
        print("\nInfer\n")
        infer_output_path = f"{args.path_output_file}"
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
        infer.main2(infer_config, args.data_file)

        # print("\nEval\n")
        # eval_output_path = f"{args.path_output_file}.eval"
        
        # eval_config = EvalConfig(
            # model_config_file,
            # model_config_args,
            # exp_config["logdir"],
            # exp_config["eval_section"],
            # f"{infer_output_path}.infer",
            # eval_output_path
        # )
        # eval.main2(eval_config, args.data_file)

        # res_json = json.load(open(eval_output_path))
        # #print(step, res_json['total_scores']['all']['exact'])
        # #print("res_json ******************************")
        # #print(step, res_json)
        # #print("Total Scores******************************")
        # #print(step, res_json['total_scores'])
        # #print("Easy ******************************")
        # #print(step, {res_json['total_scores']['easy']['count']})
        # print(f"\nResults for dataset {args.data_file}\nPrediction saved in {args.path_output_file}.json and {args.path_output_file}.txt\nGold file is gold_for_{args.path_output_file}.txt\nEvaluation results saved in results_for_{args.path_output_file}.csv\n")
        # print(f"*;count;{res_json['total_scores']['easy']['count']};{res_json['total_scores']['medium']['count']};{res_json['total_scores']['hard']['count']};{res_json['total_scores']['extra']['count']};{res_json['total_scores']['all']['count']}") 
        # print(f"checkpoint;type;easy;medium;hard;extra;all") 
        # print(f"{step};exact match;{res_json['total_scores']['easy']['exact']:.3f};{res_json['total_scores']['medium']['exact']:.3f};{res_json['total_scores']['hard']['exact']:.3f};{res_json['total_scores']['extra']['exact']:.3f};{res_json['total_scores']['all']['exact']:.3f}") 

        # result.write(f"{step};count;{res_json['total_scores']['easy']['count']};{res_json['total_scores']['medium']['count']};{res_json['total_scores']['hard']['count']};{res_json['total_scores']['extra']['count']};{res_json['total_scores']['all']['count']}\n") 
        # result.write(f"{step};exact match;{res_json['total_scores']['easy']['exact']:.3f};{res_json['total_scores']['medium']['exact']:.3f};{res_json['total_scores']['hard']['exact']:.3f};{res_json['total_scores']['extra']['exact']:.3f};{res_json['total_scores']['all']['exact']:.3f}") 
        
        # #Clean version of original .eval file        
        # eval_clean = open(f"clean_eval_for_{args.path_output_file}.csv", "w", encoding='utf8')
        # for per_item in res_json['per_item']:  
            # if per_item['exact'] == 0 or per_item['exact'] == "false": exact = "false" #in original .eval file some appear as 0 others as "false"
            # if per_item['exact'] == 1 or per_item['exact'] == "true": exact = "true" #in original .eval fiel all appear as "true", but I did the same to be standard 
            # eval_clean.write(f"{exact};{per_item['hardness']};{per_item['gold']};{per_item['predicted']}\n")                                   
        # eval_clean.close()
    #result.close()
    print(f"\nDataset {args.data_file}\nPrediction saved in {args.path_output_file}.json (key 'predicted') and {args.path_output_file}.txt (just text)\nGold file is gold_for_{args.path_output_file}.txt\n")
    
    



if __name__ == "__main__":
    main()