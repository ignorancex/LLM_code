
import sys
sys.path.append("./")

import os 
import argparse 
from utils import str2bool
from lib.eval.engine import Engine
from lib.model.llava import LLaVAModel	

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
	parser.add_argument('--dataset', type=str, default="celeba_smiling")
	parser.add_argument('--num_subjects', type=int, default=10)

	parser.add_argument('--add_caption', type=str2bool, default=False)	

	parser.add_argument('--logos_mode', type=str, default="Blank")
	parser.add_argument('--batch_size', type=int, default=128)

	parser.add_argument('--factor_shrink', type=int, default=10)
	parser.add_argument('--transparency', type=float, default=1.0)
	parser.add_argument('--logos_type', type=str, default="logos_graphics")

	args = parser.parse_args()

	total_test_logos = 4
	args.batch_size = 8

	if args.logos_mode == "None": 
		data_loop = [[(None, None, None)]]

	elif args.logos_mode == "Generic":
		logos_dir =  f"/usr4/cs505/mqraitem/ivc-ml/icon_attacks/cc12m_logos_dataset/logos"
		logo_files = os.listdir(logos_dir)
		logo_files = random.sample(logo_files, total_test_logos)	

		data_loop = [] 
		for logo_num in range(total_test_logos):
			for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:
				logo_file = logo_files[logo_num]
				logo_file_path = f"{logos_dir}/{logo_file}"
				data_loop.append([(logo_file_path, location, "logos")])

	elif args.logos_mode == "Blank":
		logo_file = "output/artifacts/blank.jpg"
		data_loop = []
		for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:
			data_loop.append([(logo_file, location, "logos")])

	elif args.logos_mode == "Concept":
		logos_dir_base =  f"output/artifacts_typo/{args.dataset}" if args.logos_type == "typo" else f"output/artifacts/{args.dataset}/ViT-L-14-336_openai/{args.num_subjects}_{args.factor_shrink}_1.0/"
		logos_dir = f"{logos_dir_base}/"

		data_loop = [] 

		for pair in os.listdir(logos_dir):

			if args.logos_type == "typo":
				logo_files = os.listdir(f"{logos_dir}/{pair}")
				logo_files = [f"{logos_dir}/{pair}/{file}" for file in logo_files]

				for logo_num, logo_file in enumerate(logo_files):
					for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:
						data_loop.append([(logo_file, location, args.logos_type)])

			else:
				logo_files = os.listdir(f"{logos_dir}/{pair}/{args.logos_type}")
				logo_files = sorted(logo_files, key=lambda x: int(x.split("_")[0]))[:total_test_logos]

				logo_files = [f"{logos_dir}/{pair}/{args.logos_type}/{file}" for file in logo_files]

				for logo_num, logo_file in enumerate(logo_files):
					for location in ["top_left", "top_right", "bottom_left", "bottom_right", "top_middle", "bottom_middle", "left_middle", "right_middle"]:
						data_loop.append([(logo_file, location, args.logos_type)])
	
	else:
		raise ValueError(f"Invalid ({args.logos_mode}")
	

	model = LLaVAModel(args)

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
			dataset.get_rep_samples(32) if train_mode == "train" else None
			
		else: 
			raise ValueError(f"Invalid dataset {args.dataset}")

		for data in tqdm(data_loop):


			name_run_dir = get_run_type_name(args)
			model_folder = "llava"

			logo_file, location, _ = data[0]
			if args.logos_mode == "Concept":
				pair =  logo_file.split("/")[-2] if args.logos_type == "typo" else logo_file.split("/")[-3]	
				dir_results = f"output/results/{train_mode}/individual_results/results_{args.factor_shrink}_{args.transparency}_{args.num_subjects}/{args.dataset}/{args.logos_mode}{name_run_dir}/{args.logos_type}/{pair}/{model_folder}/"
			else: 
				dir_results = f"output/results/{train_mode}/individual_results/results_{args.factor_shrink}_{args.transparency}_{args.num_subjects}/{args.dataset}/{args.logos_mode}{name_run_dir}/{model_folder}/"

			results_file_name = f"{logo_file.split('/')[-1]}_{location}.csv" if logo_file else "results.csv"
			if os.path.exists(f"{dir_results}/{results_file_name}"):
				print(f"Skipping {results_file_name}")
				continue

			dataset.set_logos_filenames([data])
			lvlm_prompt = dataset.get_lvlm_prompt()
			
			args.lvlm_prompt = lvlm_prompt	
			args.topic = dataset.topic + " , The options are" + ", ".join(dataset.unique_labels)

			loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=3, worker_init_fn=worker_init_fn)


			prompts, pair = dataset.get_prompts()
			engine = Engine(args)

			output = engine.get_output(model, loader, ["label", "img_path"])
			
			df = {}
			df["img_path"] = []
			df["choice"] = [] 
			df["label"] = []

			for out, label, img_path in zip(output["model_output"], output["label"], output["img_path"]):

				try: 
					choice_num = int(out[0]) - 1 
					df["choice"].append(dataset.unique_labels[choice_num])	
				except: 
					print(f"Error in output {out} for {img_path}, skipping.")
					continue

				df["label"].append(label)
				df["img_path"].append(img_path)

			df = pd.DataFrame(df)
			os.makedirs(dir_results, exist_ok=True)
			df.to_csv(f"{dir_results}/{results_file_name}", index=False)
		

if __name__ == "__main__":
	main()


