import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# from dataset import myData
# from metrics import iou_score, dice_score
# from models.ULite import ULite
from models.mamba_ulite import *
from dataloaders import *
import torch.nn.functional as F
import numpy as np
from metric import *



model = ULite()
model.eval()
# Dataset & Data Loader
import argparse

parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--path_checkpoint", type=str, default="./checkpoints", help="Path to save checkpoints")
parser.add_argument("--path_image_test", type=str, required=True, help="Path to testing images .npy file")
parser.add_argument("--path_label_test", type=str, required=True, help="Path to testing labels .npy file")
args = parser.parse_args()
CHECKPOINT_PATH = sorted(args.path_checkpoint)[0]
# Prediction

x_test = np.load(args.path_image_test)
y_test = np.load(args.path_label_test)
test_dataset = DataLoader(ISICLoader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = DiceLoss()(y_pred, y_true)
        print(loss.cpu().numpy(), end = ' ')
        # loss_test.append(loss.item())
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics
    

def compute_dice_for_images(model, dataset):
    model.cuda()
    dice_scores = {}

    for idx in range(len(dataset)):

        x, y_true = dataset[idx]
        x = x.unsqueeze(0).cuda()
        y_true = y_true.cpu().numpy().squeeze()

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred).squeeze(0)


        dice = dice_score(y_pred, torch.tensor(y_true).cuda())
        dice_scores[idx] = dice.item()

    return dice_scores

def dice_score(pred, target, smooth=1e-5):
    """
    Compute Dice score for binary segmentation.

    Args:
    - pred: The predicted mask.
    - target: The ground truth mask.
    - smooth: A small smoothing constant to prevent division by zero.

    Returns:
    - Dice score as a float.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

dice_scores = compute_dice_for_images(model=model, dataset=ISICLoader(x_test, y_test, typeData="test"))


trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)

for idx, score in dice_scores.items():
    print(f"Image {idx}: Dice score = {score:.4f}")




class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}

        # Register hooks to capture forward and backward passes
        for name, layer in target_layers.items():
            layer.register_forward_hook(self.save_activation(name))
            layer.register_backward_hook(self.save_gradient(name))

    def save_activation(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output
        return hook

    def save_gradient(self, name):
        def hook(module, input_grad, output_grad):
            if isinstance(output_grad[0], tuple):
                output_grad = output_grad[0][0]
            self.gradients[name] = output_grad[0]
        return hook

    def compute_cam(self, layer_name):
        activation = self.activations[layer_name]
        gradient = self.gradients[layer_name]

        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activation, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()

        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
        return cam

def preprocess_image(image_path, input_size=(192, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0)
    return img_tensor

def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def visualize_specific_indices(model, dataset, grad_cam, indices, target_layer_name):
    model.eval()

    plt.figure(figsize=(18, 24))

    for idx, i in enumerate(indices):
        # Get a sample from the dataset
        x, y_true = dataset[i]
        x = x.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        y_true = y_true.cpu().numpy().squeeze()

        # Forward pass through the model
        output = model(x)
        y_pred = torch.sigmoid(output).squeeze(0)  # Use sigmoid for binary classification

        # Calculate Dice score
        dice = dice_score(y_pred, torch.tensor(y_true).cuda())

        # Backward pass to compute gradients
        output = output.sum()
        model.zero_grad()
        output.backward(retain_graph=True)

        # Get Grad-CAM for the target layer
        cam = grad_cam.compute_cam(target_layer_name)

        # Visualize original image, ground truth, Grad-CAM, and Dice score
        plt.subplot(6, 4, idx * 4 + 1)
        img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)
        plt.title(f"Original Image {i}")
        plt.axis('off')

        # Ground truth mask (remove channel dimension with squeeze)
        plt.subplot(6, 4, idx * 4 + 2)
        plt.imshow(y_true.squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        # Predicted mask (remove channel dimension with squeeze)
        plt.subplot(6, 4, idx * 4 + 3)
        plt.imshow(y_pred.cpu().detach().numpy().squeeze(), cmap='gray')
        plt.title(f"Prediction (Dice: {dice:.4f})")
        plt.axis('off')

        # Grad-CAM heatmap
        plt.subplot(6, 4, idx * 4 + 4)
        plt.imshow(img_np)
        plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay Grad-CAM heatmap
        plt.title(f"Grad-CAM {target_layer_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Define the layers to capture Grad-CAM from
target_layers = {
    # 'e1': model.e1,
    # 'e2': model.e2,
    # 'e3': model.e3,
    # 'e4': model.e4,
    # 'e5': model.e5,
    # 's1': model.s1,
    # 's2': model.s2,
    # 's3': model.s3,
    # 's4': model.s4,
    # 's5': model.s5,
    # 'bottle': model.b5,
    # 'd5': model.d5,
    'd4': model.d4,
    'd3': model.d3,
    'd2': model.d2,
    'd1': model.d1,
    'pw4': model.pw4,
    'pw3': model.pw3,
    'pw2': model.pw2,
    'pw1': model.pw1,
    'conv_out': model.conv_out,
}

# Initialize GradCAM object
grad_cam = GradCAM(model, target_layers)

# List of image indices to visualize
indices_to_visualize = [1,2,3,4,5,29]

# Example visualization for a dataset and model
visualize_specific_indices(model=model, dataset=ISICLoader(x_test, y_test, typeData="test"), grad_cam=grad_cam, indices=indices_to_visualize, target_layer_name='conv_out')
