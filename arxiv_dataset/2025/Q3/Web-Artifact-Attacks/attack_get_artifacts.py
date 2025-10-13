import os 
from tqdm import tqdm 
import pickle
import numpy as np
import argparse 
from PIL import Image
from lib.dataset.dataset_fairface import FairFace
from lib.dataset.dataset_country211 import Country211
from lib.dataset.dataset_aircraft import Aircraft
from lib.dataset.dataset_celeba import CelebA
import easyocr


def check_history(imgs_history, img): 
    for past_image in imgs_history: 
        if np.array(img).shape == np.array(past_image).shape: 
            if np.abs(np.array(img).astype(np.float32) - np.array(past_image).astype(np.float32)).mean() < 20:
                return True
    return False


def save_texts(dir, filenames_sorted, acc_not_sorted): 


    os.makedirs(f"{dir}/texts", exist_ok=True)
    dir_captions = dir.replace("artifacts", "captions")
    os.makedirs(f"{dir_captions}/texts", exist_ok=True)

    imgs_history = []
        
    for idx, filename in enumerate(filenames_sorted):
        img = Image.open(filename).convert("RGBA")
        fn_last = filename.split("/")[-1]
        fn_last = int(fn_last.split(".")[0])

        txt = index_to_text[fn_last]
        img.save(f"{dir}/texts/{len(imgs_history)}_{acc_not_sorted[idx]}.png")
        
        with open(f"{dir_captions}/texts/{len(imgs_history)}_{acc_not_sorted[idx]}.txt", "w") as f:
            f.write(f"{txt} written on the image")

        imgs_history.append(img)
        if len(imgs_history) > 8: 
            break

def save_logos(dir, filenames_sorted, acc_not_sorted): 


    os.makedirs(f"{dir}/logos_text", exist_ok=True)
    os.makedirs(f"{dir}/logos_graphics", exist_ok=True)

    imgs_history_graphics = []  
    imgs_history_text = []

    reader = easyocr.Reader(["en"], gpu=True)

    for idx, filename in enumerate(filenames_sorted):
        extracted_text = reader.readtext(filename, detail=0)
        img = Image.open(filename).convert("RGB")

        if check_history(imgs_history_graphics, img) or check_history(imgs_history_text, img):
            continue

        if len(extracted_text) == 0 and len(imgs_history_graphics) < 8:
            img.save(f"{dir}/logos_graphics/{len(imgs_history_graphics)}_{acc_not_sorted[idx]}.jpg")
            imgs_history_graphics.append(img)
        elif len(imgs_history_text) < 8:
            img.save(f"{dir}/logos_text/{len(imgs_history_text)}_{acc_not_sorted[idx]}.png")
            imgs_history_text.append(img)

        if len(imgs_history_graphics) >= 8 and len(imgs_history_text) >= 8: 
            break

def filter_obv_text(text, pair): 
    text = text.lower()
    pair = pair.lower()

    if pair in text: 
        return False

    return True

parser = argparse.ArgumentParser(description='Get logo scores')
parser.add_argument('--dataset', type=str, default="fairface_age", help='args.pretrained')
args = parser.parse_args()

with open("cc12m_artifacts_dataset/index_to_text.pkl", "rb") as f:
    index_to_text = pickle.load(f)


for dataset in ["aircraft", "country211", "celeba_smiling", "fairface_age", "fairface_gender"]:
    args.dataset = dataset

    for logo_type in ["texts", "logos"]:
        scores_dir = f"output/scores/{args.dataset}/{logo_type}"

        if not os.path.exists(scores_dir):
            print(f"Scores directory {scores_dir} does not exist. Skipping...")
            continue

        for model_pretrained in os.listdir(scores_dir):
                
            for num_subjects_factorsrhink_transparency in os.listdir(f"{scores_dir}/{model_pretrained}"):   

                if not os.path.isdir(f"{scores_dir}/{model_pretrained}/{num_subjects_factorsrhink_transparency}"):
                    continue

                scores_files = os.listdir(f"{scores_dir}/{model_pretrained}/{num_subjects_factorsrhink_transparency}")
                scores_files = [f"{scores_dir}/{model_pretrained}/{num_subjects_factorsrhink_transparency}/{file}" for file in scores_files]

                num_subjects = int(num_subjects_factorsrhink_transparency.split("_")[0])
                num_classes = int(num_subjects_factorsrhink_transparency.split("_")[1])

                scores_all = []
                filenames_all = [] 
                labels = []
                
                for scores_file in tqdm(scores_files):
                    with open(scores_file, "rb") as f:
                        scores_data = pickle.load(f)
                        scores_all.append(scores_data["scores"])
                        filenames_all.extend(scores_data["filenames"])
                        labels.extend(scores_data["label"])

                scores_all = np.concatenate(scores_all, axis=0)
                filenames_all = list(dict.fromkeys(filenames_all))
                labels = np.array(labels)

                args.transparency = 1.0
                args.factor_shrink = 4
                args.owlv2 = False

                if args.dataset == "fairface_age":
                    args.concept = "age"
                    dataset = FairFace(args, split="train")
                    _, pairs = dataset.get_prompts()

                elif args.dataset == "fairface_gender":
                    args.concept = "gender"
                    dataset = FairFace(args, split="train")
                    _, pairs = dataset.get_prompts()

                elif args.dataset == "celeba_blonde":
                    dataset = CelebA(args, split="train", concept="Blond_Hair")
                    _, pairs = dataset.get_prompts()

                elif args.dataset == "celeba_smiling":
                    dataset = CelebA(args, split="train", concept="Smiling")
                    _, pairs = dataset.get_prompts()
                    
                elif args.dataset == "country211":
                    dataset = Country211(args, split="train")
                    _, pairs = dataset.get_prompts()
                
                elif args.dataset == "aircraft":
                    dataset = Aircraft(args, split="train")
                    _, pairs = dataset.get_prompts()


                scores_all = scores_all.reshape(len(set(filenames_all)), num_subjects*num_classes, num_classes)
                labels = labels.reshape(len(set(filenames_all)), -1)[0] #take the first one cause they are all the same


                #get how many pair in each labels
                pairs_count = {pair:np.sum(labels == pair) for pair in pairs}
                for pair in pairs: 

                    indices_not = [i for i, x in enumerate(labels) if x != pair]
                    labels_not = np.array([pairs.index(label) for label in labels[indices_not]])
                    labels_not = labels_not[np.newaxis, :]

                    labels_wrong = np.ones_like(labels_not) * pairs.index(pair)

                    scores_not = scores_all[:, indices_not, :]
                    scores_not = np.argmax(scores_not, axis=2) 
                    
                    acc_not = np.mean(scores_not == labels_wrong, axis=1)
                    idxs = np.argsort(acc_not)[::-1]

                    acc_not_sorted = acc_not[idxs]
                    filenames_sorted = [filenames_all[idx] for idx in idxs]

                    if logo_type == "texts":
                        filenames_sorted = [filename for filename in filenames_sorted if filter_obv_text(index_to_text[int(filename.split(".")[0].split("/")[-1])], pair)]

                    main_dest_dir = f"output/artifacts/{args.dataset}/"

                    #remove the num_class_part 
                    factor_shrink = num_subjects_factorsrhink_transparency.split("_")[-2]
                    transparency = num_subjects_factorsrhink_transparency.split("_")[-1]
                    to_write = f"{num_subjects}_{factor_shrink}_{transparency}"
                    target_dir = f"{main_dest_dir}/{model_pretrained}/{to_write}/{pair}"

                    # print(target_dir)
                    save_texts(target_dir, filenames_sorted, acc_not_sorted) if logo_type == "texts" else save_logos(target_dir, filenames_sorted, acc_not_sorted)

    
