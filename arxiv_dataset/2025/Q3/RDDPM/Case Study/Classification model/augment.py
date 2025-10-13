from torchvision import transforms
import torch
class ToChannelsLast:
    def __call__(self, x):
        return torch.permute(x, (1, 2, 0))

def get_augmentation():
    int_number=torch.randint(0,2,(1,)).item()
    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=(89*int_number,90*int_number)),  # Random rotation
        transforms.RandomResizedCrop(300, scale=(0.98, 1.02)),  # Scale the image by at most 2%
        transforms.RandomRotation(degrees=3),  # Rotation capped at ±3°
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # Translation limited to ±2% of the cell dimensions
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip 
        transforms.ToTensor(),  # Convert PIL image to Tensor
    ])
    return augmentations
def test_augmentation():
    augmentations = transforms.Compose([
        transforms.ToTensor()  # Convert PIL image to Tensor
    ])
    return augmentations
    
if __name__ == "__main__":
    aug = get_augmentation()
    print(aug)