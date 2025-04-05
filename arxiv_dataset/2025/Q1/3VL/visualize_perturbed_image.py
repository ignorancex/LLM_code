import sys
import argparse
import os
import random
import numpy as np
import torch
from torch import autograd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from hilaCAM_lora.lib.CLIP.clip import *
from gradcam.CLIP_explainability import interpret, show_image_relevance, get_image_text_relevance


#sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'clip_ancor', 'VL-CheckList'))
from vl_checklist.utils import chunks, add_caption
from vl_checklist.data_loader import DataLoader
from vl_checklist.ancor_data_loader import AncorDataLoader


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_img_patches(img, num_patches):

    patches_dim = num_patches ** 0.5
    patch_size = int(img.shape[-1] / patches_dim)

    kc = img.shape[1] #channels kernel size
    kh, kw = patch_size, patch_size  # height, width kernel size
    dc = kc #channels stride
    dh, dw = patch_size, patch_size  # height, width stride
    
    img_patches = img.clone()

    # Pad to multiples of 32
    #x = F.pad(x, (x.size(2)%kw // 2, x.size(2)%kw // 2,
    #          x.size(1)%kh // 2, x.size(1)%kh // 2,
    #          x.size(0)%kc // 2, x.size(0)%kc // 2))

    img_patches = img_patches.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = img_patches.size()
    img_patches = img_patches.contiguous().view(-1, kc, kh, kw)
    #print(f'img_patches.shape: {img_patches.shape}, unfold_shape: {unfold_shape}')
    #img_patches.shape: torch.Size([49, 3, 32, 32]), unfold_shape: torch.Size([1, 1, 7, 7, 3, 32, 32])
    #patches_orig.shape: torch.Size([1, 3, 224, 224])

    return img_patches, unfold_shape


def reshape_patches(img_patches, unfold_shape):

    # Reshape back
    patches_orig = img_patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    #print(f'patches_orig.shape: {patches_orig.shape}')
    #patches_orig.shape: torch.Size([1, 3, 224, 224])

    return patches_orig

def get_relative_relevancy_image(img, cam_image):
    total_num_patches = cam_image.shape[0]
    img_patches, unfold_shape = get_img_patches(img, total_num_patches)

    if (img_patches.max() - img_patches.min() != 0):
        img_patches = (img_patches - img_patches.min()) / (img_patches.max() - img_patches.min())
    

    #print(f'cam_image.min(): {cam_image.min()}, cam_image.max(): {cam_image.max()}\n')
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min()) 
    #print(f'cam_image.min(): {cam_image.min()}, cam_image.max(): {cam_image.max()}\n')
               
    #print(f'before scale img_patches.min(): {img_patches.min()}, img_patches.max(): {img_patches.max()}\n')


    #patch_indices = list(range(len(cam_image)))
    #img_patches[patch_indices, : , : , : ] = img_patches[patch_indices, : , : , : ]*cam_image[patch_indices]
    #[top_patches_indices, : , : , : ] = img_patches[top_patches_indices, : , : , : ]

    #print(f'img_patches.shape: {img_patches.shape},  cam_image.shape: {cam_image.shape}before torch.mul')
    #print(f'\nimg_patches before torch.mul: \n{img_patches[0]}')
    for i, scale in enumerate(cam_image):
        #img_patches[i] = img_patches[i]*scale
        img_patches[i] = img_patches[i]*scale
    #print(f'\n\nimg_patches after torch.mul: \n{img_patches[0]}')

    #print(f'after scale img_patches.min(): {img_patches.min()}, img_patches.max(): {img_patches.max()}\n')


    restored_img = reshape_patches(img_patches, unfold_shape)
    #print(f'restored_img.min(): {restored_img.min()}, restored_img.max(): {restored_img.max()}\n')


    restored_img = restored_img.squeeze(0)

    if (restored_img.max() - restored_img.min() != 0):
        restored_img = (restored_img - restored_img.min()) / (restored_img.max() - restored_img.min())
    
    #print(f'restored_img.min(): {restored_img.min()}, restored_img.max(): {restored_img.max()}\n')

    restored_img = restored_img.cpu().permute(1, 2, 0)

    return restored_img



def perturbation_image(clip_model, img, text_inputs, cam_image, pert_steps, is_positive_pert=False, folder_path="" , model_str=""):
    with torch.no_grad():
        
        cam_shaded_img_vis = get_relative_relevancy_image(img, cam_image)
        fig, axs = plt.subplots()
        axs.imshow(cam_shaded_img_vis)
        axs.axis('off')
        axs.set_title('Anchor shaded relevancy', fontsize=8)
        plt.savefig(f'{folder_path}VLC_VG{model_str}_Anchor_shaded_relevancy.png')
        plt.close()

        if is_positive_pert:
            cam_image = cam_image * (-1)
        
        total_num_patches = cam_image.shape[0]
        #print(f'total_num_patches: {total_num_patches}')

        ##############################################
        ##############################################

        #TODO: visualize image with masked patches
        #img.shape: torch.Size([1, 3, 224, 224])
        #print(f'img.shape: {img.shape}')
        #masked_img = torch.zeros_like(img)
        
        img_patches, unfold_shape = get_img_patches(img, total_num_patches)
        
        if (img_patches.max() - img_patches.min() != 0):
            img_patches = (img_patches - img_patches.min()) / (img_patches.max() - img_patches.min())
    

        #img_patches = img.clone()
        #img_patches = img_patches.squeeze(0)
        #tot_patches_dim = total_num_patches ** 0.5
        #patch_size = int(img_patches.shape[-1] / tot_patches_dim)
        #image dim: 224, total_num_patches: 49, tot_patches_dim: 7.0, patch_size: 32
        #print(f'image dim: {img_patches.shape[-1]}, total_num_patches: {total_num_patches}, tot_patches_dim: {tot_patches_dim}, patch_size: {patch_size}')
        #exit(0)
        #reshape to have total_num_patches patches
        #img_patches = img_patches.reshape(img.shape[1], total_num_patches, -1)
        #img_patches = img_patches.reshape(img.shape[1], total_num_patches, patch_size, patch_size)


        positive_text_fig = plt.figure(figsize=(9, 9), dpi=200)
        negative_text_fig = plt.figure(figsize=(9, 9), dpi=200)
        masked_img_fig = plt.figure(figsize=(9, 9), dpi=200)
        figures = [("pos_text", positive_text_fig), ("neg_text", negative_text_fig)]
        #for step_idx, num_top_patches in enumerate(num_tokens):
        for step_idx, step in enumerate(pert_steps):
            # find top step boxes
            num_top_patches = int((1 - step) * total_num_patches)
            #print(f'step_idx: {step_idx}, step: {step}, num_top_patches: {num_top_patches}')
            _, top_patches_indices = cam_image.topk(k=num_top_patches, dim=-1)
            #print(f'top_patches_indices: {top_patches_indices}')
            top_patches_indices = top_patches_indices.cpu().data.numpy()
            top_patches_indices.sort()

            #TODO: visualize image with masked patches
            #img.shape: torch.Size([1, 3, 224, 224])
            #print(f'img.shape: {img.shape}')
            #masked_img = torch.zeros_like(img)
            #img_patches = img.copy()
            
            #reshape to have total_num_patches patches
            #img_patches = img_patches.reshape(img.shape[0], img.shape[1], total_num_patches, -1)

            masked_img = torch.zeros_like(img_patches)

            #masked_img = masked_img.reshape_as(img_patches)
            #img.shape: torch.Size([1, 3, 224, 224]), img_patches.shape: torch.Size([3, 49, 1024]), masked_img.shape: torch.Size([3, 49, 1024])
            #print(f'img.shape: {img.shape}, img_patches.shape: {img_patches.shape}, masked_img.shape: {masked_img.shape}')
            
            masked_img[top_patches_indices, : , : , : ] = img_patches[top_patches_indices, : , : , : ]
            masked_img = reshape_patches(masked_img, unfold_shape)

            #masked_img = masked_img.reshape_as(img)
            masked_img = masked_img.squeeze(0)
            if (masked_img.max() - masked_img.min() != 0):
                masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())
            #img.shape: torch.Size([1, 3, 224, 224]), img_patches.shape: torch.Size([3, 49, 1024]), masked_img.shape: torch.Size([3, 224, 224])
            #print(f'img.shape: {img.shape}, img_patches.shape: {img_patches.shape}, masked_img.shape: {masked_img.shape}')
            masked_img = masked_img.cpu().permute(1, 2, 0)
            #masked_img = np.uint8(255 * masked_img)


            ax_mask = masked_img_fig.add_subplot(3, 3, step_idx+1)
            ax_mask.axis('off')
            #ax.set_title(f"{pos_neg_str}_{int(100 - 100*num_top_patches/num_tokens[0])}% removed")
            ax_mask.set_title(f"Image with {100*step}% patches removed")
            ax_mask.title.set_fontsize(8)
            #pil_img = Image.open(img_file_name)
            ax_mask.imshow(masked_img)


            

            #total_num_patches=49
            #img_mask = img.reshape(1,3,49,224/49,224/49 )
            #patched_img = torch.zeros_like(img_mask)
            #for ind in top_patches_indices:
            #    masked_img[:,:,ind,:,:] = patched_img[:,:,ind,:,:]

            with torch.set_grad_enabled(True):
                R_text, R_image = interpret(model=clip_model, image=img, texts=text_inputs, device=device, top_patches_indices=top_patches_indices)
            
            batch_size = text_inputs.shape[0]
            for i, (pos_neg_str, fig) in enumerate(figures):
                
                image_relevance = torch.zeros_like(cam_image)
                
                for rel_index, patch_index in enumerate(top_patches_indices):
                    image_relevance[patch_index] = R_image[i][rel_index]
                
                #img_file_name = show_image_relevance(R_image[i], img, img, idx=step_idx, file_name_prefix=f"VL_checklist_VG_{pos_neg_str}")
                img_file_name = show_image_relevance(image_relevance, img, img, idx=step_idx, file_name_prefix=f"VL_checklist_VG_{pos_neg_str}")


                ax = fig.add_subplot(3, 3, step_idx+1)
                ax.axis('off')
                #ax.set_title(f"{pos_neg_str}_{int(100 - 100*num_top_patches/num_tokens[0])}% removed")
                ax.set_title(f"{pos_neg_str} {100*step}% removed")
                ax.title.set_fontsize(8)
                pil_img = Image.open(img_file_name)
                ax.imshow(pil_img)

                
        for pos_neg_str, fig in figures: 
            fig.savefig(f"{folder_path}VLC_VG{model_str}_{pos_neg_str}_pert_image_text_relevance.jpg", dpi=200, bbox_inches='tight')
        
        masked_img_fig.savefig(f"{folder_path}VLC_VG{model_str}_masked_image.jpg", dpi=200, bbox_inches='tight')
        plt.close()


    

def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
    clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
    #clip_model = clip_model.cuda()
    clip_model = clip_model.to(device=device)
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
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Image Preprocessing:", preprocess)
    print("=========")

    return clip_model, preprocess


def seed_everything(seed: int = 42):
  
  print(f'seed_everythin with seed: {seed}')        
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def get_text_features(clip_model, text_inputs):
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).to(device)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

def get_positve_negative_scores(text_features, image_features):
    
    #normalize image_features and text_features
    image_features /= image_features.norm(dim=-1, keepdim=True).to(device)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    #print(f'similarity.shape: {similarity.shape}, similarity: {similarity}')

    pos_score = similarity[0][0]
    neg_score = similarity[0][1]
    
    return pos_score, neg_score


#def main(args):
def main():
    
    test_type = "positive"
    #modality = "text" if args.is_text_pert else "image"
    #batch_size = args.batch_size
    batch_size = 1
    print(f"running visualize pert image")

    #################################
        
    ####
    vit_name = 'ViT-B/32'
    #self.load_model(base_name=vit_name, weight_name=checkpoint_file)
    #load with lora lib
    lora=1
    clip_3VL_model, preprocess = load_model(base_name=vit_name, lora_r=lora)
    clip_base_model, preprocess = load_model(base_name=vit_name, lora_r=lora)

    
    #checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
    filename = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

    #print(f'testing with no training')

    print(f'loading clip_3VL_model state from filename: {filename}')
    checkpoint = torch.load(filename) 
    clip_3VL_model.load_state_dict(checkpoint['state_dict'])

    clip_3VL_model = clip_3VL_model.to(device=device)
    clip_3VL_model = clip_3VL_model.float()

    clip_base_model = clip_base_model.to(device=device)
    clip_base_model = clip_base_model.float()

    #VL-Checklist dataloader
    #data_types
    #TYPES: ["Relation/action", "Relation/spatial", "Attribute/color", "Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]
    #TYPES: ["Object/Location/center", "Object/Location/margin", "Object/Location/mid", "Object/Size/large", "Object/Size/medium", "Object/Size/small"]
    #TEST_DATA: ["vg"]
    data_names = ["vg"]
    #data_type = "Relation/action"
    #data_type = "Relation/spatial"
    #data_type = "Attribute/color"
    #TYPES = ["Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]
    #TYPES = ["Relation/action", "Relation/spatial", "Attribute/color", "Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]
    
    # TYPES = ["Attribute/action"]#next image_num: 5
    # TYPES = ["Attribute/color"] #next image_num: 677
    # TYPES = ["Attribute/material"] #next image_num: 84
    TYPES = ["Attribute/size"] #next image_num: 37
    # TYPES = ["Attribute/state"] #next image_num: 53
    # TYPES = ["Relation/action"] #next image_num: 29
    # TYPES = ["Relation/spatial"] #next image_num: 14
    # dataset_folder_name = "/attribute_action/"
    # dataset_folder_name = "/attribute_color/"
    # dataset_folder_name = "/attribute_material/"
    dataset_folder_name = "/attribute_size/"
    # dataset_folder_name = "/attribute_state/"
    # dataset_folder_name = "/relation_action/"
    # dataset_folder_name = "/relation_spatial/"

    #positive_negative_pert_list = [True, False]
    positive_negative_pert_list = [False]
    task = 'itc'
    #positive_cam = True
    #positive_cam = False
    #relevancy_map_str = "positive" if positive_cam else "negative"
    print(f'use diff relevancy map positive - negative')
    # folder_name = "VL_checklist_visualizations/Anchor/"
    folder_name = "VL_checklist_visualizations/DiRe/"
    
    for data_type in TYPES:
        for is_positive_pert in positive_negative_pert_list:
        
            print(f'starting perturbation_image visualization for data_type: {data_type} with is_positive_pert: {is_positive_pert}')
            
            #d = DataLoader(data_names, data_type, task)
            d = AncorDataLoader(data_names, data_type, task)
            for name in d.data:
                #print(f'name: {name}')

                iterator = tqdm(chunks(d.data[name], batch_size), desc="Progress", ncols=100,
                                            total=int(len(d.data[name]) / batch_size))
                #for batch in tqdm(chunks(d.data[name], self.batch_size), desc="Progress", ncols=100,
                #                              total=int(len(d.data[name]) / self.batch_size)):
                            #for batch in tqdm(chunks(d.test_data[name], self.batch_size), desc="Progress", ncols=100,
                            #                total=int(len(d.test_data[name]) / self.batch_size)):
                                

                pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                #num_tokens = [49, 36, 25, 16, 9, 4]
                
                image_num = 37
                image_cnt = 0
                #remove counters_dict. use one counter to keep track of the image index to be able to compare to other methods (Anchor to DiRe)
                #counters_dict = dict()
                # counters_dict["CLIP_correct_3VL_correct"] = 0
                # counters_dict["CLIP_correct_3VL_incorrect"] = 0
                # counters_dict["CLIP_incorrect_3VL_correct"] = 0
                # counters_dict["CLIP_incorrect_3VL_incorrect"] = 0
                 
                for index, batch in enumerate(iterator):

                    # """
                    if image_cnt < image_num:
                        print(f'image_cnt: {image_cnt}, image_num: {image_num}')
                        image_cnt += 1
                        continue
                    # """

                    image_paths = [z["path"] for z in batch]
                    texts_pos = [z['texts_pos'][0] for z in batch]
                    texts_neg = [z['texts_neg'][0] for z in batch]
                    texts_anc = [z['texts_anc'][0] for z in batch]

                    #print(f'images.shape: {images.shape}, texts_pos.shape: {texts_pos.shape}, texts_neg.shape: {texts_neg.shape}')
                    #print(f'\n\nimage_paths: {image_paths}, \ntexts_pos: {texts_pos}, \ntexts_neg: {texts_neg}')

                    texts = [texts_pos[0], texts_neg[0], texts_anc[0]]
                    orig_image = Image.open(image_paths[0])
                    img = preprocess(orig_image).unsqueeze(0).to(device)
                    #img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    #texts = ["a man with eyeglasses"]
                    tokenized_text = clip.tokenize(texts).to(device)

                    with torch.no_grad():
                        text_features = get_text_features(clip_3VL_model, tokenized_text)
                        image_features = clip_3VL_model.encode_image(img)

                        clip_3VL_pos_score, clip_3VL_neg_score = get_positve_negative_scores(text_features, image_features)
                        clip_3VL_correct = clip_3VL_pos_score > clip_3VL_neg_score

                        text_features = get_text_features(clip_base_model, tokenized_text)
                        image_features = clip_base_model.encode_image(img)

                        clip_base_pos_score, clip_base_neg_score = get_positve_negative_scores(text_features, image_features)
                        clip_base_correct = clip_base_pos_score > clip_base_neg_score


                    if clip_base_correct:
                        correct_path_str = "CLIP_correct_"
                    else:
                        correct_path_str = "CLIP_incorrect_"

                    if clip_3VL_correct:
                        correct_path_str = correct_path_str + "3VL_correct"
                    else:
                        correct_path_str = correct_path_str + "3VL_incorrect"

                    save_folder_path = folder_name + correct_path_str + dataset_folder_name

                    #img_cnt = counters_dict[correct_path_str]
                    
                    
                    #image_dir = save_folder_path + f'Img_{img_cnt}'
                    image_dir = save_folder_path + f'Img_{image_cnt}'

                    image_cnt += 1
                    #counters_dict[correct_path_str] += 1
                    print(f'image_dir: {image_dir}')
                    
                    #if not os.path.exists(image_dir):
                    #    os.mkdir(image_dir)
                    
                    os.makedirs(image_dir, exist_ok=True)

                    merged_path = image_dir + "/"

    
                    #if pos_score > neg_score:
                    #    image_cnt += 1
                    #    print(f'got correct result - continue to image_cnt: {image_cnt}')
                    #    continue


                    #visualize positive and negative relevance
                    clip_3VL_R_text, clip_3VL_R_image = get_image_text_relevance(image_paths[0], texts, clip_3VL_model, preprocess, file_name_prefix="VLC_VG")
                    clip_base_R_text, clip_base_R_image = get_image_text_relevance(image_paths[0], texts, clip_base_model, preprocess, file_name_prefix="VLC_VG")

                    #R_text, R_image = interpret(model=clip_model, image=img, texts=tokenized_text, device=device)
                    
                    #cam_image_idx = 0 if positive_cam, negative 1 anchor 2
                    clip_3VL_positive_cam_image = clip_3VL_R_image[0]
                    clip_3VL_negative_cam_image = clip_3VL_R_image[1]
                    clip_3VL_ancor_cam_image = clip_3VL_R_image[2]

                    clip_base_positive_cam_image = clip_base_R_image[0]
                    clip_base_negative_cam_image = clip_base_R_image[1]
                    clip_base_ancor_cam_image = clip_base_R_image[2]

                    
                    clip_3VL_DiRe_cam_image = clip_3VL_positive_cam_image - clip_3VL_negative_cam_image
                    clip_3VL_DiRe_cam_image = (clip_3VL_DiRe_cam_image - clip_3VL_DiRe_cam_image.min()) / (clip_3VL_DiRe_cam_image.max() - clip_3VL_DiRe_cam_image.min())


                    
                    clip_base_DiRe_cam_image = clip_base_positive_cam_image - clip_base_negative_cam_image
                    clip_base_DiRe_cam_image = (clip_base_DiRe_cam_image - clip_base_DiRe_cam_image.min()) / (clip_base_DiRe_cam_image.max() - clip_base_DiRe_cam_image.min())


                    clip_3VL_pos_relative_relevancy_vis = get_relative_relevancy_image(img, clip_3VL_positive_cam_image)
                    clip_3VL_neg_relative_relevancy_vis = get_relative_relevancy_image(img, clip_3VL_negative_cam_image)
                    clip_3VL_anc_relative_relevancy_vis = get_relative_relevancy_image(img, clip_3VL_ancor_cam_image)
                    clip_3VL_dire_relative_relevancy_vis = get_relative_relevancy_image(img, clip_3VL_DiRe_cam_image)




                    clip_base_pos_relative_relevancy_vis = get_relative_relevancy_image(img, clip_base_positive_cam_image)
                    clip_base_neg_relative_relevancy_vis = get_relative_relevancy_image(img, clip_base_negative_cam_image)
                    clip_base_anc_relative_relevancy_vis = get_relative_relevancy_image(img, clip_base_ancor_cam_image)
                    clip_base_dire_relative_relevancy_vis = get_relative_relevancy_image(img, clip_base_DiRe_cam_image)



                    clip_3VL_relative_relevancy_fig = plt.figure(figsize=(6, 6), dpi=200)

                    ax1_rel = clip_3VL_relative_relevancy_fig.add_subplot(1, 3, 1)
                    ax1_rel.imshow(clip_3VL_pos_relative_relevancy_vis)
                    ax1_rel.axis('off')
                    ax1_rel.set_title(texts_pos[0], fontsize=8)
                    ax2_rel = clip_3VL_relative_relevancy_fig.add_subplot(1, 3, 2)
                    ax2_rel.imshow(clip_3VL_neg_relative_relevancy_vis)
                    ax2_rel.axis('off')
                    ax2_rel.set_title(texts_neg[0], fontsize=8)
                    ###
                    #anchor
                    ax3_rel = clip_3VL_relative_relevancy_fig.add_subplot(1, 3, 3)
                    ax3_rel.imshow(clip_3VL_dire_relative_relevancy_vis)
                    ax3_rel.axis('off')
                    ax3_rel.set_title('DiRe', fontsize=8)
                    ###
                    plt.tight_layout()
                    plt.savefig(f'{merged_path}VLC_VG_3VL_pos_neg_DiRe_relative_relevancy.png', bbox_inches='tight')
                    plt.close()



                    clip_base_relative_relevancy_fig = plt.figure(figsize=(6, 6), dpi=200)

                    ax1_rel = clip_base_relative_relevancy_fig.add_subplot(1, 3, 1)
                    ax1_rel.imshow(clip_base_pos_relative_relevancy_vis)
                    ax1_rel.axis('off')
                    ax1_rel.set_title(texts_pos[0], fontsize=8)
                    ax2_rel = clip_base_relative_relevancy_fig.add_subplot(1, 3, 2)
                    ax2_rel.imshow(clip_base_neg_relative_relevancy_vis)
                    ax2_rel.axis('off')
                    ax2_rel.set_title(texts_neg[0], fontsize=8)
                    ###
                    #anchor / DiRe
                    ax3_rel = clip_base_relative_relevancy_fig.add_subplot(1, 3, 3)
                    ax3_rel.imshow(clip_base_dire_relative_relevancy_vis)
                    ax3_rel.axis('off')
                    ax3_rel.set_title('DiRe', fontsize=8)
                    ###
                    plt.tight_layout()
                    plt.savefig(f'{merged_path}VLC_VG_base_CLIP_pos_neg_dire_relative_relevancy.png', bbox_inches='tight')
                    plt.close()


                    #visualize 3VL positive negative anchor relevancy maps
                    """
                    clip_3VL_pos_cam_img_file_name = show_image_relevance(clip_3VL_positive_cam_image, img, orig_image=orig_image, idx=2)
                    clip_3VL_neg_cam_img_file_name = show_image_relevance(clip_3VL_negative_cam_image, img, orig_image=orig_image, idx=3)
                    clip_3VL_anc_cam_img_file_name = show_image_relevance(clip_3VL_ancor_cam_image, img, orig_image=orig_image, idx=4)



                    clip_3VL_pos_image = Image.open(clip_3VL_pos_cam_img_file_name)
                    clip_3VL_neg_image = Image.open(clip_3VL_neg_cam_img_file_name)
                    clip_3VL_anc_image = Image.open(clip_3VL_anc_cam_img_file_name)

                    #fig = plt.figure(figsize=(6, 6), dpi=200)
                    fig = plt.figure()
                   
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(clip_3VL_pos_image)
                    ax1.axis('off')
                    ax1.set_title(f'{texts_pos[0]}', fontsize=8)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(clip_3VL_neg_image)
                    ax2.axis('off')
                    ax2.set_title(f'{texts_neg[0]}', fontsize=8)
                    ###
                    #anchor
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(clip_3VL_anc_image)
                    ax3.axis('off')
                    ax3.set_title(f'{texts_anc[0]}', fontsize=8)
                    ###
                    plt.tight_layout()
                    plt.savefig(f'{merged_path}VLC_VG_3VL_pos_neg_anc_images', bbox_inches='tight')
                    plt.close()
                    """

                   
                    #visualize base CLIP positive negative anchor relevancy maps
                    """
                    clip_base_pos_cam_img_file_name = show_image_relevance(clip_base_positive_cam_image, img, orig_image=orig_image, idx=4)
                    clip_base_neg_cam_img_file_name = show_image_relevance(clip_base_negative_cam_image, img, orig_image=orig_image, idx=5)
                    clip_base_anc_cam_img_file_name = show_image_relevance(clip_base_ancor_cam_image, img, orig_image=orig_image, idx=6)



                    clip_base_pos_image = Image.open(clip_base_pos_cam_img_file_name)
                    clip_base_neg_image = Image.open(clip_base_neg_cam_img_file_name)
                    clip_base_anc_image = Image.open(clip_base_anc_cam_img_file_name)

                    #fig = plt.figure(figsize=(6, 6), dpi=200)
                    fig = plt.figure()
                   
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(clip_base_pos_image)
                    ax1.axis('off')
                    ax1.set_title(f'{texts_pos[0]}', fontsize=8)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(clip_base_neg_image)
                    ax2.axis('off')
                    ax2.set_title(f'{texts_neg[0]}', fontsize=8)
                    ###
                    #anchor
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(clip_base_anc_image)
                    ax3.axis('off')
                    ax3.set_title(f'{texts_anc[0]}', fontsize=8)
                    ###
                    plt.tight_layout()
                    plt.savefig(f'{merged_path}VLC_VG_CLIP_base_pos_neg_anc_images', bbox_inches='tight')
                    plt.close()
                    """


                    #visualize base CLIP diff relevance map
                    # clip_base_DiRe_cam_image = clip_base_positive_cam_image - clip_base_negative_cam_image
                    # clip_base_DiRe_cam_image = (clip_base_DiRe_cam_image - clip_base_DiRe_cam_image.min()) / (clip_base_DiRe_cam_image.max() - clip_base_DiRe_cam_image.min())

                    #cam_image = negative_cam_image *(-1)

                    #visualize anchor relevance map
                    # clip_base_cam_image = clip_base_ancor_cam_image
                    
                    # clip_base_cam_image = (clip_base_cam_image - clip_base_cam_image.min()) / (clip_base_cam_image.max() - clip_base_cam_image.min())

                    #visualize anchor and DiRe relevance after norm
                    # clip_base_anc_cam_img_file_name = show_image_relevance(clip_base_cam_image, img, orig_image=orig_image, idx=1)
                    # clip_base_dire_cam_img_file_name = show_image_relevance(clip_base_DiRe_cam_image, img, orig_image=orig_image, idx=1)


                    # clip_base_image_anchor_cam = Image.open(clip_base_anc_cam_img_file_name)
                    # clip_base_image_dire_cam = Image.open(clip_base_dire_cam_img_file_name)
                    

                    ######################
                    # fig = plt.figure()
                   
                    # ax1 = fig.add_subplot(1, 2, 1)
                    # ax1.imshow(clip_base_image_anchor_cam)
                    # ax1.axis('off')
                    # ax1.set_title(f'Anchor', fontsize=8)
                    # ax2 = fig.add_subplot(1, 2, 2)
                    # ax2.imshow(clip_base_image_dire_cam)
                    # ax2.axis('off')
                    # ax2.set_title(f'DiRe', fontsize=8)                    
                    # plt.tight_layout()
                    # plt.savefig(f'{merged_path}VLC_VG_CLIP_base_Anchor_DiRe_CAM_images', bbox_inches='tight')
                    # plt.close()
                    ######################


                    # fig, axs = plt.subplots()
                    # axs.imshow(clip_base_image_diff_cam)
                    # axs.axis('off')
                    # plt.savefig(f'{merged_path}VLC_VG_CLIP_base_image_diff_cam.png')
                    # plt.close()

                    # fig, axs = plt.subplots()
                    # axs.imshow(orig_image)
                    # axs.axis('off')
                    # plt.savefig(f'{merged_path}orig_image.png')
                    # plt.close()
                    


                    #visualize 3VL diff relevance map
                    # clip_3VL_DiRe_cam_image = clip_3VL_positive_cam_image - clip_3VL_negative_cam_image
                    # clip_3VL_DiRe_cam_image = (clip_3VL_DiRe_cam_image - clip_3VL_DiRe_cam_image.min()) / (clip_3VL_DiRe_cam_image.max() - clip_3VL_DiRe_cam_image.min())

                    # clip_3VL_cam_image = clip_3VL_ancor_cam_image
                    # clip_3VL_cam_image = (clip_3VL_cam_image - clip_3VL_cam_image.min()) / (clip_3VL_cam_image.max() - clip_3VL_cam_image.min())

                    #visualize anchor and DiRe relevance after norm
                    # clip_3VL_anc_cam_img_file_name = show_image_relevance(clip_3VL_cam_image, img, orig_image=orig_image, idx=3)
                    # clip_3VL_dire_cam_img_file_name = show_image_relevance(clip_3VL_DiRe_cam_image, img, orig_image=orig_image, idx=4)

                    #######
                    # clip_3VL_image_anchor_cam = Image.open(clip_3VL_anc_cam_img_file_name)
                    # clip_3VL_image_dire_cam = Image.open(clip_3VL_dire_cam_img_file_name)
                    

                    #######
                    #######
                    # fig = plt.figure()
                   
                    # ax1 = fig.add_subplot(1, 2, 1)
                    # ax1.imshow(clip_3VL_image_anchor_cam)
                    # ax1.axis('off')
                    # ax1.set_title(f'Anchor', fontsize=8)
                    # ax2 = fig.add_subplot(1, 2, 2)
                    # ax2.imshow(clip_3VL_image_dire_cam)
                    # ax2.axis('off')
                    # ax2.set_title(f'DiRe', fontsize=8)                    
                    # plt.tight_layout()
                    # plt.savefig(f'{merged_path}VLC_VG_3VL_Anchor_DiRe_CAM_images', bbox_inches='tight')
                    # plt.close()
                    #######
                    #######


                    
                    # fig, axs = plt.subplots()
                    # axs.imshow(clip_3VL_image_diff_cam)
                    # axs.axis('off')
                    # plt.savefig(f'{merged_path}VLC_VG_3VL_image_diff_cam.png')
                    # plt.close()


                    #visualize diff relevance with negative perturbation
                    # perturbation_image(clip_base_model, img, tokenized_text, clip_base_cam_image, pert_steps, is_positive_pert, folder_path=merged_path , model_str="_CLIP_base")   
                    # perturbation_image(clip_3VL_model, img, tokenized_text, clip_3VL_cam_image, pert_steps, is_positive_pert, folder_path=merged_path , model_str="_3VL")   
                    #print(f'exit(0) after the first image visualization')
                    #exit(0)
                    

if __name__ == "__main__":
    #main(args)
    #seed_everything()
    seed_everything(1)
    main()