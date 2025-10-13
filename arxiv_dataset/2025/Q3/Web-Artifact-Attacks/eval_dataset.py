
import sys
sys.path.append("./")

import os 
import argparse 
from utils import str2bool
from lib.eval.engine import Engine
from lib.model.clip_zs import ClipZeroShot

from lib.dataset.dataset_fairface import FairFace
from lib.dataset.dataset_country211 import Country211
from lib.dataset.dataset_aircraft import Aircraft
from lib.dataset.dataset_celeba import CelebA

from utils import worker_init_fn
from utils import get_run_type_name

import torch
from tqdm import tqdm
import pandas as pd

import random
random.seed(0)

def main(): 

	#add parser
	parser = argparse.ArgumentParser(description='Get logo scores')
	parser.add_argument('--dataset', type=str, default="fairface_age")
	parser.add_argument('--num_subjects', type=int, default=32)

	parser.add_argument('--pretrained', type=str, default="laion400m_e32")
	parser.add_argument('--model_name', type=str, default="ViT-B-32")

	parser.add_argument('--logos_mode', type=str, default="None")
	parser.add_argument('--batch_size', type=int, default=128)

	parser.add_argument('--factor_shrink', type=int, default=10)
	parser.add_argument('--transparency', type=float, default=1.0)
	parser.add_argument('--logos_type', type=str, default="logos")
	parser.add_argument('--add_caption', type=str2bool, default=False)	

	args = parser.parse_args()


	total_test_logos = 4
	
	if args.logos_mode == "None": 
		data_loop = [[(None, None, None)]]

	elif args.logos_mode == "Blank":
		logo_file = "output/artifacts/blank.jpg"
		data_loop = []
		for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:
			data_loop.append([(logo_file, location, "logos")])

	elif args.logos_mode == "Concept":
		logos_dir_base = f"output/artifacts_typo/{args.dataset}" if args.logos_type == "typo" else f"output/artifacts/{args.dataset}/{args.model_name}_{args.pretrained}/{args.num_subjects}_10_1.0/"
		logos_dir = f"{logos_dir_base}/"

		data_loop = [] 
		logo_to_caption = {} 
		for pair in os.listdir(logos_dir):
			if args.logos_type == "typo":

				logo_files = os.listdir(f"{logos_dir}/{pair}")
				logo_files = [f"{logos_dir}/{pair}/{file}" for file in logo_files]

				for logo_num, logo_file in enumerate(logo_files):
					for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:

						if args.add_caption:
							caption = f"{pair} written on it"
							logo_to_caption[logo_file] = caption
						
						data_loop.append([(logo_file, location, args.logos_type)])

			else: 
				logo_files = os.listdir(f"{logos_dir}/{pair}/{args.logos_type}")
				logo_files = sorted(logo_files, key=lambda x: int(x.split("_")[0]))[:total_test_logos]

				logo_files = [f"{logos_dir}/{pair}/{args.logos_type}/{file}" for file in logo_files]
				for logo_num, logo_file in enumerate(logo_files):
					for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:

						if args.add_caption:
							caption_file_link = logo_file.replace("artifacts", "captions")    
							caption_file_link = caption_file_link.replace(".jpg", ".txt")
							caption_file_link = caption_file_link.replace(".png", ".txt")	
							with open(caption_file_link, "r") as f:
								caption = f.readlines()[0].strip()
							
							logo_to_caption[logo_file] = caption
						
						
						data_loop.append([(logo_file, location, args.logos_type)])

	else:
		raise ValueError(f"Invalid ({args.logos_mode}")
	

	for train_mode in ["train", "test"]:
		if args.dataset == "fairface_age":
			args.concept = "age"
			train_val = "train" if train_mode == "train" else "val"
			dataset = FairFace(args, split=train_val)
			dataset.get_rep_samples(32) if train_mode == "train" else None
			
		elif args.dataset == "fairface_gender":
			train_val = "train" if train_mode == "train" else "val"
			args.concept = "gender"
			dataset = FairFace(args, split=train_val)
			dataset.get_rep_samples(32) if train_mode == "train" else None

		elif args.dataset == "celeba_smiling":	
			train_val = "val" if train_mode == "train" else "test"
			dataset = CelebA(args, split=train_val, concept="Smiling")	
			dataset.get_rep_samples(32) if train_mode == "train" else None

		elif args.dataset == "country211":
			train_val = "train" if train_mode == "train" else "test"
			dataset = Country211(args, split=train_val)
			dataset.get_rep_samples(32) if train_mode == "train" else None
			
		elif args.dataset == "aircraft":
			train_val = "train" if train_mode == "train" else "test" 	
			dataset = Aircraft(args, split=train_val)
			dataset.get_rep_samples(32) if train_mode == "train"else None
			
		else: 
			raise ValueError(f"Invalid dataset {args.dataset}")

		for data in tqdm(data_loop):

			dataset.set_logos_filenames([data])
			loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=3, worker_init_fn=worker_init_fn)


			prompts, pair = dataset.get_prompts()
			if args.add_caption: 
				caption = logo_to_caption[data[0][0]]
				for prompt_set in prompts: 
					for idx, prompt in enumerate(prompt_set):	
						prompt_set[idx] = f"{prompt} and a {caption}"	
				
			model = ClipZeroShot(args, prompts)

			engine = Engine(args)
			output = engine.get_output(model, loader, ["label", "img_path"])
			
			df = {}
			df["img_path"] = []
			for p in pair: 
				df[f"{p}"] = []
			df["label"] = []

			for logits, label, img_path in zip(output["model_output"], output["label"], output["img_path"]):
				df["label"].append(label)
				df["img_path"].append(img_path)
				for p, logit in zip(pair, logits):
					df[f"{p}"].append(logit.item())
		
			df = pd.DataFrame(df)

			name_run_dir = get_run_type_name(args)
			model_folder = f"{args.model_name}_{args.pretrained}/{args.model_name}_{args.pretrained}" if "llava" not in args.model_name else "llava"

			logo_file, location, _ = data[0]

			if args.logos_mode == "Concept":
				pair =  logo_file.split("/")[-2] if args.logos_type == "typo" else logo_file.split("/")[-3]	
				dir_results = f"output/results/{train_mode}/individual_results/results_{args.factor_shrink}_{args.transparency}_{args.num_subjects}/{args.dataset}/{args.logos_mode}{name_run_dir}/{args.logos_type}/{pair}/{model_folder}/"
			else: 
				dir_results = f"output/results/{train_mode}/individual_results/results_{args.factor_shrink}_{args.transparency}_{args.num_subjects}/{args.dataset}/{args.logos_mode}{name_run_dir}/{model_folder}/"

			results_file_name = f"{logo_file.split('/')[-1]}_{location}.csv" if logo_file else "results.csv"

			os.makedirs(dir_results, exist_ok=True)
			df.to_csv(f"{dir_results}/{results_file_name}", index=False)
				

if __name__ == "__main__":
	main()