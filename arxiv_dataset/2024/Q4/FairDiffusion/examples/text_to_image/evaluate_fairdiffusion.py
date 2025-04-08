import argparse
import os
import shutil
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from data_handler import FairGenMed
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch_fidelity
from train_text_to_image import normalize_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FairDiffusion")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--initial_model", type=str, default=None)
    parser.add_argument("--race_prompt", action="store_true")
    parser.add_argument("--gender_prompt", action="store_true")
    parser.add_argument("--ethnicity_prompt", action="store_true")
    parser.add_argument("--datasets", type=str, default=None, nargs="+")
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--metrics_calculation_idx", type=int, default=None, required=True)
    args = parser.parse_args()
    return args

def evaluate_fairdiffusion(args):
    if os.path.exists(f'metrics_calculation_{args.metrics_calculation_idx}'):
        shutil.rmtree(f'metrics_calculation_{args.metrics_calculation_idx}')

    os.mkdir(f'metrics_calculation_{args.metrics_calculation_idx}')
    os.mkdir(f'metrics_calculation_{args.metrics_calculation_idx}/actual')
    os.mkdir(f'metrics_calculation_{args.metrics_calculation_idx}/generated')
    os.mkdir(f'metrics_calculation_{args.metrics_calculation_idx}/prompts')

    unet = UNet2DConditionModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(args.initial_model, unet=unet, torch_dtype=torch.float16)
    pipe.to("cuda")
    print(pipe.text_encoder.text_model.encoder.layers[0].mlp.fc2.weight[:3,:3].tolist())
    print(unet.mid_block.resnets[0].conv1.weight[:3,:3,0,0].tolist())

    diffusion_datasets = []
    if 'glaucoma' in args.datasets:
        diffusion_datasets.append(FairGenMed(os.path.join(args.dataset_dir, 'test'), modality_type='slo_fundus', task='cls', resolution=200, attribute_type='race', needBalance=False, dataset_proportion=1.0, race_prompt=args.race_prompt, gender_prompt=args.gender_prompt, ethnicity_prompt=args.ethnicity_prompt))

    dataset = torch.utils.data.ConcatDataset(diffusion_datasets)

    # convert PyTorch dataset to HuggingFace dataset
    images_list = []
    texts_list = []
    for item in dataset:
        images_list.append(item['image'])
        texts_list.append(item['text'])
    data_dict = {'image': images_list, 'text': texts_list}
    dataset = Dataset.from_dict(data_dict)

    actual_imgs = []
    generated_imgs = []
    prompts = []
    all_prompts = []

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation((-90, -90)),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ]
    )

    for idx, item in enumerate(dataset):
        actual_img = train_transforms(normalize_image(Image.open(item['image']).convert("RGB")))
        gen_img = pipe(prompt=item["text"], num_inference_steps=75, height=512, width=512, guidance_scale=7.5).images[0]

        actual_img = actual_img.rotate(90)
        gen_img = gen_img.rotate(90)

        if idx < 10:
            actual_imgs.append(actual_img)
            generated_imgs.append(gen_img)
            prompts.append(item["text"])
    
        actual_img.save(os.path.join(f"metrics_calculation_{args.metrics_calculation_idx}/actual", f"{idx}.jpg"))
        gen_img.save(os.path.join(f"metrics_calculation_{args.metrics_calculation_idx}/generated", f"{idx}.jpg"))
        all_prompts.append(item["text"])

    # save all_prompts
    with open(os.path.join(f"metrics_calculation_{args.metrics_calculation_idx}/prompts", "prompts.txt"), "w") as f:
        for prompt in all_prompts:
            f.write(prompt + "\n")
    plot_imgs(actual_imgs, generated_imgs, prompts, args.model_path.split('/')[1])
    metrics_dict = compute_fairness_metrics(dir1=f"metrics_calculation_{args.metrics_calculation_idx}/generated", dir2=f"metrics_calculation_{args.metrics_calculation_idx}/actual")
    print(metrics_dict)
    compute_groupwise_metrics(args)
    shutil.rmtree(f'metrics_calculation_{args.metrics_calculation_idx}')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = sorted(os.listdir(directory))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def compute_fairness_metrics(dir1, dir2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
    ])
    dataset1 = CustomDataset(directory=dir1, transform=transform)
    dataloader1 = DataLoader(dataset1, batch_size=100, shuffle=False)
    dataset2 = CustomDataset(directory=dir2, transform=transform)
    dataloader2 = DataLoader(dataset2, batch_size=100, shuffle=False)

    fid = FrechetInceptionDistance().to(device)
    fid.set_dtype(torch.float64)
    for batch in dataloader1:
        fid.update(batch.to(device), real=False)
    for batch in dataloader2:
        fid.update(batch.to(device), real=True)
    fid_metric = fid.compute().item()

    mifid = MemorizationInformedFrechetInceptionDistance().to(device)
    for batch in dataloader1:
        mifid.update(batch.to(device), real=False)
    for batch in dataloader2:
        mifid.update(batch.to(device), real=True)
    mifid_metric = mifid.compute().item()

    inception = InceptionScore().to(device)
    for batch in dataloader1:
        inception.update(batch.to(device))
    inception_metric = inception.compute()

    return {'fid': fid_metric, 'mifid': mifid_metric, 'is': inception_metric}

def compute_groupwise_metrics(args):
    demographic_groups = [' Asian', ' Black', ' White', ' Female', ' Male', ' Non-Hispanic', ' Hispanic', ' non-glaucoma', ' glaucoma', ' normal vision', ' mild vision loss', ' moderate vision loss', ' severe vision loss', ' normal cup-disc ratio', ' borderline cup-disc ratio', ' abnormal cup-disc ratio', ' positive', ' neutral', ' negative']
    prompts_path = f"metrics_calculation_{args.metrics_calculation_idx}/prompts/prompts.txt"
    
    # Read prompts and categorize images
    with open(prompts_path, 'r') as file:
        prompts = file.readlines()
    
    # For each demographic group, filter images and calculate metrics
    for group in demographic_groups:
        group_prompts = [idx for idx, prompt in enumerate(prompts) if group in prompt]
        print(group)
        print(len(group_prompts))
        tmp_dir_actual = f"metrics_calculation_{args.metrics_calculation_idx}_tmp/actual/{group}"
        tmp_dir_generated = f"metrics_calculation_{args.metrics_calculation_idx}_tmp/generated/{group}"
        os.makedirs(tmp_dir_actual, exist_ok=True)
        os.makedirs(tmp_dir_generated, exist_ok=True)
        
        # Filter and save images for the current group
        for idx in group_prompts:
            shutil.copy(f"metrics_calculation_{args.metrics_calculation_idx}/actual/{idx}.jpg", tmp_dir_actual)
            shutil.copy(f"metrics_calculation_{args.metrics_calculation_idx}/generated/{idx}.jpg", tmp_dir_generated)
        
        # Compute metrics for the current group
        metrics_dict = compute_fairness_metrics(dir1=tmp_dir_generated, dir2=tmp_dir_actual)

        # Print or store metrics for analysis
        print(f"Metrics for {group}: {metrics_dict}")
        
        # Clean up temporary directories after calculation
        shutil.rmtree(tmp_dir_actual)
        shutil.rmtree(tmp_dir_generated)


def plot_imgs(actual_images, generated_images, column_names, filename):
    fig, axs = plt.subplots(2, len(column_names), figsize=(15, 6))

    for i, img in enumerate(actual_images):
        axs[0, i].imshow(img)
        axs[0, i].axis('off')  # Turn off axis
        if i == 0:
            axs[0, i].set_ylabel('Actual')

    for i, img in enumerate(generated_images):
        axs[1, i].imshow(img)
        axs[1, i].axis('off')  # Turn off axis
        if i == 0:
            axs[1, i].set_ylabel('Generated')

    for i in range(len(column_names)):
        axs[0, i].set_title(column_names[i], fontsize=6)

    plt.tight_layout()
    plt.savefig(f'grid_{filename}.png')

if __name__ == "__main__":
    args = parse_args()
    evaluate_fairdiffusion(args)