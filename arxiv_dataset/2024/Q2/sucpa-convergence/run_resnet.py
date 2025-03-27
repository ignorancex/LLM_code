
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

np.random.seed(3748)
torch.manual_seed(3748)


IMGS_PATH = 'data/dogs-vs-cats/train'
test_idx = 'data/dogs-vs-cats/test_list.txt'

hyperparameters = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 4,
    'print_every': 10,
    'eval_every': 20,
}


class DogsVsCatsDataset(Dataset):

    def __init__(self, paths_list, transform=None):
        self.paths_list = paths_list
        self.transform = transform
        self.class2idx = {'cat': 0, 'dog': 1}

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):
        image_name = self.paths_list[idx]
        image_path = os.path.join(IMGS_PATH, image_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = 0 if image_name.startswith('cat') else 1
        return image, label


def get_data_loaders(batch_size=32, val_size=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    all_images = os.listdir(IMGS_PATH)
    test_images = np.array([all_images[idx] for idx in np.loadtxt(test_idx, dtype=int)])
    all_images = np.array([img for img in all_images if img not in test_images])
    np.random.shuffle(all_images)
    all_images = all_images
    
    split_idx = int(len(all_images) * val_size)
    train_images = all_images[split_idx:]
    val_images = all_images[:split_idx]
    train_dataset = DogsVsCatsDataset(train_images, transform=transform)
    val_dataset = DogsVsCatsDataset(val_images, transform=transform)
    test_dataset = DogsVsCatsDataset(test_images, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 2)
    for param in model.parameters():
        param.requires_grad = True
    return model


def train(model, loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    best_loss = float("inf")
    for epoch in range(hyperparameters["epochs"]):
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % hyperparameters["print_every"] == 0:
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'Epoch: {epoch}, Batch: {i}, Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}')
            if (i + 1) % hyperparameters["eval_every"] == 0:
                eval_loss = eval(model, test_loader, criterion, device)
                if eval_loss < best_loss:
                    torch.save(model.state_dict(), "models/model.pt")
                    best_loss = eval_loss
                torch.save(model.state_dict(), "models/last_model.pt")
                model.train()
    return model, device


def eval(model, loader, criterion=nn.CrossEntropyLoss(), device=None):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        loss = 0
        for images, labels in tqdm(loader, leave=False, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels) * images.size(0)
    loss /= len(loader.dataset)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}')
    return loss


def get_logits(model, device, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc='Getting logits'):
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs)
            all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    return all_logits, all_labels


def main():
    os.makedirs("models", exist_ok=True)

    model = get_model()
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=hyperparameters["batch_size"], val_size=0.2)
    model, device = train(model, train_loader, val_loader)

    for prefix in ["", "last_"]:

        model.load_state_dict(torch.load(f"models/{prefix}model.pt"))
        model = model.to(device)
        logits_path = "/".join(IMGS_PATH.split('/')[:-1])
        logits, labels = get_logits(model, device, train_loader)
        np.save(os.path.join(logits_path,f'{prefix}train_logits.npy'), logits)
        np.save(os.path.join(logits_path,f'{prefix}train_labels.npy'), labels)
        
        logits, labels = get_logits(model, device, val_loader)
        np.save(os.path.join(logits_path,f'{prefix}val_logits.npy'), logits)
        np.save(os.path.join(logits_path,f'{prefix}val_labels.npy'), labels)

        logits, labels = get_logits(model, device, test_loader)
        np.save(os.path.join(logits_path,f'{prefix}test_logits.npy'), logits)
        np.save(os.path.join(logits_path,f'{prefix}test_labels.npy'), labels)



if __name__ == '__main__':
    main()
