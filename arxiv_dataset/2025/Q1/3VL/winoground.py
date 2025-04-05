from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy
import torch

#import clip with LoRA
#from lora.lib.CLIP.clip import *

#import clip without LoRA
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"



def get_token():
    with open("../token.txt") as f:
        return f.read()


def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
    
    clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
    clip_model = clip_model.to(device)
    clip_model = clip_model.float()
    
    for param in clip_model.parameters():
        param.data = param.data.float()
        if param.grad:
            param.grad.data = param.grad.data.float() 

    if lora_r <= 0:
        #clip.model.convert_weights(self.clip_model)
        model.convert_weights(clip_model)

    if weight_name:
        clip_model.load_state_dict(torch.load(weight_name)['state_dict'])
        
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    print("=========")
    print("Model parameters:", f"{numpy.sum([int(numpy.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Image Preprocessing:", preprocess)
    print("=========")
    return clip_model, preprocess


# Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
#auth_token = get_token()
auth_token = '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'

winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]


# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
#filename = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

#checkpoint trained with LR_3e-6 LoRA rank = 1, get_caption_tree6_shuffle_all
#filename = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption6_shuffle_all_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0008.pt.tar'

#checkpoint trained with LR_3e-6 LoRA rank = 1, get_caption_tree6_shuffled_nouns_adjectives
filename = '/mnt5/nir/CLIP/interpret/DT_0.5_tree6_shuffle_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0008.pt.tar'
        



model_name = "ViT-B/32"

#load model with LoRA
lora = 1
#lora = 4
#lora = 2
#clip_model, preprocess = load_model(model_name, lora)


#load WITHOUT LoRA 
clip_model, preprocess = clip.load(model_name, device=device, jit=False)


clip_model = clip_model.float()

#print(f'loading model state from filename: {filename}')

print(f'run with pretrained CLIP model')


clip_model = clip_model.float()

#print(f'loading model state from filename: {filename}')
checkpoint = torch.load(filename)

#clip_model.load_state_dict(checkpoint['state_dict'])



# clip_model = CLIPModel.from_pretrained(filename).to("cuda")
# clip_model.load_state_dict(checkpoint['state_dict'])

# clip_processor = CLIPProcessor.from_pretrained(filename)


def get_clip_scores(id):
    # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
    # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
    # input_c0_i0 = clip_processor(text=[winoground[id]["caption_0"]], images=[winoground[id]["image_0"].convert("RGB")], return_tensors="pt").to("cuda")
    # input_c1_i0 = clip_processor(text=[winoground[id]["caption_1"]], images=[winoground[id]["image_0"].convert("RGB")], return_tensors="pt").to("cuda")
    # input_c0_i1 = clip_processor(text=[winoground[id]["caption_0"]], images=[winoground[id]["image_1"].convert("RGB")], return_tensors="pt").to("cuda")
    # input_c1_i1 = clip_processor(text=[winoground[id]["caption_1"]], images=[winoground[id]["image_1"].convert("RGB")], return_tensors="pt").to("cuda")
    # output_c0_i0 = clip_model(**input_c0_i0)
    # output_c1_i0 = clip_model(**input_c1_i0)
    # output_c0_i1 = clip_model(**input_c0_i1)
    # output_c1_i1 = clip_model(**input_c1_i1)
    # clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
    # clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
    # clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
    # clip_score_c1_i1 = output_c1_i1.logits_per_image.item()

    
    texts = [winoground[id]["caption_0"], winoground[id]["caption_1"]]
    tokenized_texts = clip.tokenize(texts).cuda() #tokenize
    text_features = clip_model.encode_text(tokenized_texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)  
    
    #preprocess(Image.open(chunk_i[j])).unsqueeze(0).to(self.device)
    images = [preprocess(winoground[id]["image_0"]).to(device), preprocess(winoground[id]["image_1"]).to(device)]

    images = torch.stack(images).to(device)

    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    scores = 100. * text_features @ image_features.T

    #clip_scores = {"id": id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}
    clip_scores = {"id": id, "c0_i0": scores[0][0].item(), "c0_i1": scores[0][1].item(), "c1_i0": scores[1][0].item(), "c1_i1": scores[1][1].item()}
    return clip_scores


def print_scores(scores):
    print("image_0, caption_0:", scores[0][0].item())
    print("image_0, caption_1:", scores[1][0].item())
    print("image_1, caption_0:", scores[0][1].item())
    print("image_1, caption_1:", scores[1][1].item())



def get_all_clip_scores():  
    winoground_clip_scores = []
    for id in tqdm(range(len(winoground))):
        clip_scores = get_clip_scores(id=id)
        winoground_clip_scores.append(clip_scores)
    return winoground_clip_scores




def save_results(filename, winoground_scores):
    with open(filename, 'w') as f:
        for scores in winoground_scores:
            f.write(f'{{"label": "{scores["id"]}_c0_i0", "score": {scores["c0_i0"]}}}\n')
            f.write(f'{{"label": "{scores["id"]}_c1_i0", "score": {scores["c1_i0"]}}}\n')
            f.write(f'{{"label": "{scores["id"]}_c0_i1", "score": {scores["c0_i1"]}}}\n')
            f.write(f'{{"label": "{scores["id"]}_c1_i1", "score": {scores["c1_i1"]}}}\n')


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def print_performance(winoground_scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_scores)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)


with torch.no_grad():
    winoground_clip_scores = get_all_clip_scores()
    print("CLIP scores:")
    print_performance(winoground_clip_scores)
    save_results("winoground_3VL_scores.jsonl", winoground_clip_scores)

