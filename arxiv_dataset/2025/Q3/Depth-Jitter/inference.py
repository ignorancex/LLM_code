import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from q2l_labeller.pl_modules.query2label_train_module import Query2LabelTrainModule
from q2l_labeller.data.dataset import SeaThruAugmentation

# Define the transformation pipeline (Resize, ToTensor, Normalize)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path, num_classes):
    """Load the trained Query2Label model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    model = Query2LabelTrainModule(
        data=None,  # Data module is not required for inference
        backbone_desc=checkpoint["hyper_parameters"]["backbone_desc"],
        conv_out_dim=checkpoint["hyper_parameters"]["conv_out_dim"],
        hidden_dim=checkpoint["hyper_parameters"]["hidden_dim"],
        num_encoders=checkpoint["hyper_parameters"]["num_encoders"],
        num_decoders=checkpoint["hyper_parameters"]["num_decoders"],
        num_heads=checkpoint["hyper_parameters"]["num_heads"],
        batch_size=checkpoint["hyper_parameters"]["batch_size"],
        image_dim=checkpoint["hyper_parameters"]["image_dim"],
        learning_rate=checkpoint["hyper_parameters"]["learning_rate"],
        momentum=checkpoint["hyper_parameters"]["momentum"],
        weight_decay=checkpoint["hyper_parameters"]["weight_decay"],
        n_classes=num_classes,  # Dynamically set class count
        thresh=0.4,  # Default threshold, can be changed
        use_cutmix=checkpoint["hyper_parameters"]["use_cutmix"],
        use_pos_encoding=checkpoint["hyper_parameters"]["use_pos_encoding"],
        loss=checkpoint["hyper_parameters"]["loss"],
    )

    model.load_state_dict(checkpoint["state_dict"])  # Load model weights
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def preprocess_image(image_path, seathru_transform=None):
    """Load and preprocess an image for inference."""
    image = Image.open(image_path).convert("RGB")

    if seathru_transform:
        image = seathru_transform(image_path, image)

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, image_tensor, class_labels):
    """Run inference and return predicted labels and probabilities."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # Apply sigmoid to get probabilities

    predictions = [(class_labels[i], probs[i]) for i in range(len(probs)) if probs[i] > 0.4]  # Thresholding
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Inference with Depth-Jitter Query2Label model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes (4 for UTDAC, 290 for FathomNet).")
    parser.add_argument("--seathru", action="store_true", help="Apply SeaThru transformation if available.")
    args = parser.parse_args()

    # Load class labels (You should replace this with your dataset's class names)
    class_labels = [f"Class {i}" for i in range(args.num_classes)]

    # Load the model
    model = load_model(args.checkpoint, args.num_classes)

    # Initialize SeaThru transformation (if enabled)
    seathru_transform = None
    if args.seathru:
        seathru_transform = SeaThruAugmentation(
            image_folder="",  # Not needed for inference
            depth_image_folder="",
            depth_npy_folder="",
            seathru_parameters_path="parameters_train.json",
            depth_variance_path="depth_variance.json",
            threshold=7.5
        )

    # Preprocess the input image
    image_tensor = preprocess_image(args.image, seathru_transform)

    # Run inference
    predictions = predict(model, image_tensor, class_labels)

    # Print results
    print("\n🎯 Predictions:")
    for label, prob in predictions:
        print(f"- {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
