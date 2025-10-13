
import sys
sys.path.append("./")

import os 
import argparse 
from lib.eval.engine import Engine
from lib.model.clip_zs import ClipZeroShot
from lib.dataset.dataset_fairface import FairFace 
from lib.dataset.dataset_country211 import Country211
from lib.dataset.dataset_aircraft import Aircraft
from lib.dataset.dataset_celeba import CelebA
from utils import worker_init_fn
import torch
import pickle

def main(): 

	#add parser
	parser = argparse.ArgumentParser(description='Get logo scores')
	parser.add_argument('--dataset', type=str, default="fairface_age")
	parser.add_argument('--pretrained', type=str, default="laion2b_s34b_b79k")
	parser.add_argument('--model_name', type=str, default="ViT-B-32")
	parser.add_argument('--num_subjects', type=int, default=32)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--logos_type', type=str, default="logos")

	args = parser.parse_args()

	args.transparency = 1.0
	args.factor_shrink = 10
	args.attack_location = "random"
	args.logos_mode = "find"
	
	logo_dir = f"cc12m_artifacts_dataset/{args.logos_type}"
	logo_files = os.listdir(logo_dir)
	logo_files = [os.path.join(logo_dir, x) for x in logo_files]

	start = 0
	end = len(logo_files)
	logo_files = logo_files[start:end]
	logo_files = [[(x, "random", args.logos_type)] for x in logo_files]

	if args.dataset == "fairface_age":
		args.concept = "age"
		dataset = FairFace(args, split="train")
		dataset.get_rep_samples(args.num_subjects)
		print(dataset.get_distro())
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	elif args.dataset == "fairface_gender":
		args.concept = "gender"
		dataset = FairFace(args, split="train")
		dataset.get_rep_samples(args.num_subjects)
		print(dataset.get_distro())
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	elif args.dataset == "celeba_blonde":
		dataset = CelebA(args, split="val", concept="Blond_Hair")
		dataset.get_rep_samples(args.num_subjects)
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	elif args.dataset == "celeba_smiling":
		dataset = CelebA(args, split="val", concept="Smiling")
		dataset.get_rep_samples(args.num_subjects)
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	elif args.dataset == "aircraft":
		dataset = Aircraft(args, split="train")
		dataset.get_rep_samples(args.num_subjects)
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	elif args.dataset == "country211": 
		dataset = Country211(args, split="train")	
		dataset.get_rep_samples(args.num_subjects)
		dataset.set_logos_filenames(logo_files)
		selected_classes = dataset.unique_labels

	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=3, worker_init_fn=worker_init_fn, shuffle=False)

	prompts, _, = dataset.get_prompts() 	
	model = ClipZeroShot(args, prompts)

	engine = Engine(args)
	output = engine.get_output(model, loader, ["logo_path", "label"])

	scores_final = {}
	scores_final["scores"] = output["model_output"].cpu().numpy()
	scores_final["filenames"] = output["logo_path"]
	scores_final["label"] = output["label"]

	assert len(scores_final["scores"]) == len(scores_final["filenames"])

	#save scores
	num_classes = len(selected_classes)
	dir_results = f"output/scores/{args.dataset}/{args.logos_type}/{args.model_name}_{args.pretrained}/{args.num_subjects}_{num_classes}_{args.factor_shrink}_{args.transparency}/"
	os.makedirs(dir_results, exist_ok=True)
	file_name = f"{args.run_index}.pkl"
	file_path = os.path.join(dir_results, file_name)
	with open(file_path, 'wb') as f:
		pickle.dump(scores_final, f)

if __name__ == "__main__":
	main()