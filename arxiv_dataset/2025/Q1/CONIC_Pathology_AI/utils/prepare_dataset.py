import os
import cv2
import joblib
import numpy as np

def monusac_process():
    splits = ["Train", "Test"]

    for split in splits:
        img_folder = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/ft_local/img/{split}"
        mask_folder = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/ft_local/mask/{split}"
        type_folder = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/ft_local/type/{split}"

        imgs = []
        masks = []

        img_names = os.listdir(img_folder)

        for img_name in img_names:
            img_path = f"{img_folder}/{img_name}"
            img_this = cv2.imread(img_path)

            mask_path = f"{mask_folder}/{img_name[:-4]}.npy"
            mask_this = np.load(mask_path)

            type_path = f"{type_folder}/{img_name[:-4]}.npy"
            type_this = np.load(type_path)

            imgs.append(cv2.cvtColor(img_this, cv2.COLOR_BGR2RGB))
            
            type_map = np.zeros_like(mask_this)
            for inst_tp in type_this:
                inst_indexes = np.where(mask_this == inst_tp[0])
                type_map[inst_indexes] = inst_tp[1]
            
            mask = np.concatenate((mask_this, type_map), axis=-1)
            masks.append(mask)

        imgs = np.array(imgs)
        masks = np.array(masks)
        os.makedirs("/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/img", exist_ok=True)
        os.makedirs("/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/mask", exist_ok=True)

        dst_img_path = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/img/{split}.npy"
        dst_mask_path = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/monusac/mask/{split}.npy"

        with open(dst_img_path, "wb") as f_op:
            np.save(f_op, imgs)
        
        with open(dst_mask_path, "wb") as  f_op:
            np.save(f_op, masks)

def pannuke_process():
    img_path = "/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/pannuke/ft_local/Fold_3/images/images.npy"
    types_path = "/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/pannuke/ft_local/Fold_3/images/types.npy"

    types_images = np.load(types_path)
    print("types_images shape: ", types_images.shape)
    print(types_images[0])


def lizard_process_openmmlab():
    index_path = "splits_10_fold.dat"
    splits = joblib.load(index_path)
    train_indices = splits[0]["train"]
    test_indices = splits[0]["valid"]
    imgs = np.load("data/images.npy")
    # masks = np.load("data/labels.npy")

    dst_folder = "/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/conic_jbhi/lizard"

    img_folder = f"{dst_folder}/images/train"
    os.makedirs(img_folder, exist_ok=True)
    for index in train_indices:
        img = imgs[index]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_path = f"{img_folder}/{index:04d}.png"
        cv2.imwrite(img_path, img)

    img_folder = f"{dst_folder}/images/valid"
    os.makedirs(img_folder, exist_ok=True)
    for index in test_indices:
        img = imgs[index]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_path = f"{img_folder}/{index:04d}.png"
        cv2.imwrite(img_path, img)



if __name__ == "__main__":
    # monusac_process()
    # pannuke_process()
    lizard_process_openmmlab()