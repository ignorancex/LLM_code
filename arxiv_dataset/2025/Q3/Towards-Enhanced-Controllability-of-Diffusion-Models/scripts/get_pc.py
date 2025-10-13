import torch
import glob
import os

'''
w schedule (exclusive) : "2022-10-21T18-10-27"
w schedule (sigmoid) : "2022-10-19T16-38-21"
wo schedule : "2022-10-29T05-02-34"
'''
fname = '2022-10-29T05-02-34'
print("===============================")
print("filename: ",fname)
print("===============================")
root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"
num_pc = 30

train_dir = f'{root}/latents/{fname}/semantic_codes_train'
# val_dir = f'{root}/{fname}/semantic_codes_val'
target_dir = f'{root}/latents/{fname}/pc'

os.makedirs(target_dir, exist_ok=True)

train_style_list = glob.glob(os.path.join(train_dir, "stylecode*pth"))
# val_style_list = glob.glob(os.path.join(val_dir, "stylecode*pth"))
train_content_list = glob.glob(os.path.join(train_dir, "contentcode*pth"))
# val_content_list = glob.glob(os.path.join(val_dir, "contentcode*pth"))

temp = ['train_style', 'train_content'] # ['train_style', 'val_style', 'train_content', 'val_content']
# temp = ['train_content', 'val_content']
for i, item_list in enumerate([train_style_list, train_content_list]): # enumerate([train_style_list, val_style_list, train_content_list, val_content_list]):
# for i, item_list in enumerate([train_content_list, val_content_list]):
    conds = []
    if len(item_list) > 0:
        for item in item_list:
            conds.append(torch.load(item))

        conds = torch.stack(conds, dim=0) # N, C
        if conds.size() != (len(item_list), 512):
            conds = conds.reshape(len(item_list), -1)
        U, S, V = torch.pca_lowrank(conds, q=num_pc)
        torch.save(S, os.path.join(target_dir, f'{temp[i]}_singular_values.pth'))
        torch.save(V.T, os.path.join(target_dir, f'{temp[i]}_right_singular_vectors.pth'))

        print(f'Done computing pc for {temp[i]} latents! The # of samples used here is {conds.size(0)}.')
