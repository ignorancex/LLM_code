from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
import torchvision.transforms as transforms

def load_wilds_camelyon17(corruption_type,
                   clean_cifar_path,
                   corruption_cifar_path,
                   corruption_severity=0,
                   datatype='test',
                   seed=1):
    dataset = Camelyon17Dataset(download=True, root_dir=f"{clean_cifar_path}/")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if corruption_severity == 0:
        load_set = dataset.get_subset('train', transform=transform)
    else:
        load_set = dataset.get_subset(corruption_type, transform=transform)
    return load_set



