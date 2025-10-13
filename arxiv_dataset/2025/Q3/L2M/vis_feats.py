
import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import argparse
from romatch.models.transformer import vit_base

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- 可视化函数 --------------------
def vis_feat_map_batch(features_list, patch_h, patch_w, resize_hw=(560, 560)):
    all_feats = np.concatenate([f.reshape(patch_h * patch_w, -1) for f in features_list], axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_feats)

    images = []
    for features in features_list:
        f = features.reshape(patch_h * patch_w, -1)
        pca_feats = pca.transform(f)
        pca_feats = (pca_feats - pca_feats.mean(0)) / (pca_feats.std(0) + 1e-5)
        pca_feats = np.clip(pca_feats * 0.5 + 0.5, 0, 1)
        img = pca_feats.reshape(patch_h, patch_w, 3)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).resize(resize_hw, Image.BICUBIC)
        images.append(img)
    return images

def save_combined_visualization(
    feats_dino, feats_fit3d, feats_L2M,
    patch_h, patch_w, base_name, save_dir, original_images
):
    os.makedirs(save_dir, exist_ok=True)

    imgs_dino = vis_feat_map_batch(feats_dino, patch_h, patch_w)
    imgs_fit3d = vis_feat_map_batch(feats_fit3d, patch_h, patch_w)
    imgs_L2M = vis_feat_map_batch(feats_L2M, patch_h, patch_w)

    # 拼图：每行一个图，共两行四列
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    titles = ["Original", "DINOv2", "Fit3D", "L2M (Ours)"]
    for i in range(2):  # row
        row_imgs = [original_images[i], imgs_dino[i], imgs_fit3d[i], imgs_L2M[i]]
        for j in range(4):
            axs[i, j].imshow(row_imgs[j])
            axs[i, j].set_title(titles[j], fontsize=12)
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_compare.png"))
    plt.close()

# -------------------- 特征提取函数 --------------------
def extract_features(model, image_tensor):
    with torch.no_grad():
        return model.forward_features(image_tensor)["x_norm_patchtokens"].squeeze(0).cpu().numpy()

# -------------------- 主脚本 --------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    patch_h, patch_w = 37, 37
    img_size = patch_h * 14

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    # 初始化模型
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0
    )

    def load_model(ckpt_path):
        model = vit_base(**vit_kwargs).eval().to(device)
        raw = torch.load(ckpt_path, map_location="cpu")
        if "model" in raw:
            raw = raw["model"]
        ckpt = {k.replace("model.", ""): v for k, v in raw.items()}
        model.load_state_dict(ckpt, strict=False)
        return model

    dino = load_model(args.ckpt_dino)
    fit3d = load_model(args.ckpt_fit3d)
    L2M = load_model(args.ckpt_L2M)

    feats_dino, feats_fit3d, feats_L2M = [], [], []
    original_images = []

    for img_path in args.img_paths:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        feats_dino.append(extract_features(dino, x))
        feats_fit3d.append(extract_features(fit3d, x))
        feats_L2M.append(extract_features(L2M, x))
        original_images.append(img)

    base_name = "multi" if len(args.img_paths) > 1 else os.path.splitext(os.path.basename(args.img_paths[0]))[0]
    save_combined_visualization(
        feats_dino, feats_fit3d, feats_L2M,
        patch_h, patch_w, base_name, args.save_dir, original_images
    )

    print(f"Saved 2-row comparison to {os.path.join(args.save_dir, f'{base_name}_compare.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_paths",
        nargs="+",
        default=[
            "assets/sacre_coeur_A.jpg",
            "assets/sacre_coeur_B.jpg"
        ],
        help="List of image paths"
    )
    parser.add_argument(
        "--ckpt_fit3d",
        default="ckpts/fit3d.pth",
        help="Original Fit3D checkpoint"
    )
    parser.add_argument(
        "--ckpt_L2M",
        default="ckpts/l2m_vit_base.pth",
        help="L2M Fit3D checkpoint"
    )
    parser.add_argument(
        "--ckpt_dino",
        default="ckpts/dinov2.pth",
        help="dino checkpoint"
    )
    parser.add_argument(
        "--save_dir",
        default="outputs_vis_feat",
        help="Directory to save visualizations"
    )
    args = parser.parse_args()
    main(args)
