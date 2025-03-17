import torchvision
import torchvision.transforms as transforms


CIFAR10_Train_ROOT = ''
CIFAR10_Test_ROOT = ''
CIFAR100_Train_ROOT = ''
CIFAR100_Test_ROOT = ''
TinyImageNet_Resize_Test_ROOT = ''
TinyImageNet_Crop_Test_ROOT = ''
SVHN_Train_root = ''
SVHN_Test_root = ''
LSUN_Test_Resize_root = ''
LSUN_Test_Crop_root = ''
iSUN_Test_root = ''


#CIFAR10 data set
CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)

# CIFAR100 data set
CIFAR100_train_data = torchvision.datasets.ImageFolder(CIFAR100_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
# SVHN data set
SVHN_train_data = torchvision.datasets.ImageFolder(SVHN_Train_root,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
SVHN_test_data = torchvision.datasets.ImageFolder(SVHN_Test_root,
    transform=transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
TinyImageNet_Resize_Test = torchvision.datasets.ImageFolder(TinyImageNet_Resize_Test_ROOT,
    transform=transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)

TinyImageNet_Crop_Test = torchvision.datasets.ImageFolder(TinyImageNet_Crop_Test_ROOT,
    transform=transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)

LSUN_Test_Resize = torchvision.datasets.ImageFolder(LSUN_Test_Resize_root,
    transform=transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
# )
LSUN_Test_Crop = torchvision.datasets.ImageFolder(LSUN_Test_Crop_root,
    transform=transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
iSUN_test_data = torchvision.datasets.ImageFolder(iSUN_Test_root,
    transform=transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
