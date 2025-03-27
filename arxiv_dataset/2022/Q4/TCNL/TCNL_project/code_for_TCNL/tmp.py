import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(torch.cuda.is_available())
model, preprocess = clip.load("RN50x64", device=device)

image = preprocess(Image.open("/home/user/wangzhihao/XCSGCNN/test.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["head", "torso", "leg", "shape"]).to(device)

with torch.no_grad():
    image_feature = model.encode_image(image)
    text_features = model.encode_text(text)
print(image_feature.shape)
print(text_features.shape)