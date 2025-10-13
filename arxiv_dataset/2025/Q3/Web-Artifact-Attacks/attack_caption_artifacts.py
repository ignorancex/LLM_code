
import sys
sys.path.append("./")
import os 
from lib.model.llava import LLaVAModel	
from PIL import Image
import random
random.seed(0)

def main(): 

	#add parser
    class Args(): 
        def __init__(self): 
            self.lvlm_prompt = "Caption this artifact in one sentence."
     
    args = Args()   
    args.add_caption = False
    model = LLaVAModel(args)

    artifact_dir = "output/artifacts"
    for dataset in os.listdir(artifact_dir):
        dataset_dir = os.path.join(artifact_dir, dataset)   
        for model_pretrained in os.listdir(dataset_dir):
            model_dir = os.path.join(dataset_dir, model_pretrained)
            for transp_dir in os.listdir(model_dir):
                
                num_subjects = int(transp_dir.split("_")[0])
                if num_subjects not in [32, 10]:
                    continue

                transp_dir = os.path.join(model_dir, transp_dir)
                for pair in os.listdir(transp_dir):
                    pair_dir = os.path.join(transp_dir, pair)
                    artifact_types = ["logos_graphics", "logos_text"] 

                    batch = {"images":[]}
                    art_links = [] 
                    for artifact_type in artifact_types: 
                        artifact_type_dir = os.path.join(pair_dir, artifact_type)
                        for artifact in os.listdir(artifact_type_dir):
                            art_link = os.path.join(artifact_type_dir, artifact)
                            batch["images"].append([Image.open(art_link)])
                            art_links.append(art_link)

                    check_if_done = [False for _ in range(len(art_links))]
                    for link in art_links:
                        new_link = link.replace("artifacts", "captions")    
                        new_link = new_link.replace(".jpg", ".txt") 
                        new_link = new_link.replace(".png", ".txt")
                        if os.path.exists(new_link):
                            check_if_done[art_links.index(link)] = True
                        
                    if all(check_if_done):
                        print(f"All captions already exist for {pair_dir}, skipping.")
                        continue

                    results = model(batch)
                    for link, result in zip(art_links, results): 
                        new_link = link.replace("artifacts", "captions")    
                        new_link = new_link.replace(".jpg", ".txt") 
                        new_link = new_link.replace(".png", ".txt")
                        os.makedirs(os.path.dirname(new_link), exist_ok=True)

                        if os.path.exists(new_link):
                            print(f"Caption already exists for {link}, skipping.")
                            continue

                        with open(new_link, "w") as f: 
                            f.write(result)
                            print(f"Writing caption for {link} to {new_link}")


if __name__ == "__main__":
    main()