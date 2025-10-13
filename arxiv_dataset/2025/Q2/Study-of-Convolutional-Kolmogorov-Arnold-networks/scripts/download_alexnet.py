import torch
import torchvision.models as models
model = models.alexnet(pretrained=True)
torch.save(model.state_dict(), 'alexnet_pretrained.pth')
