import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from thop import profile
from skimage import img_as_ubyte


from Utils.data_loader import InferenceDataset
from Networks.unirNet import UNIRNet

from Utils.rce import RCE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


model = UNIRNet().to(device)
CCM = RCE(gamma_val=1.4).to(device)

model.load_state_dict(torch.load('./Checkpoints/UNIR-Net.pt'))
model.eval()  


input_dir = "./1_Input"
output_dir = "./2_Output"
os.makedirs(output_dir, exist_ok=True)  


dataset = InferenceDataset(input_dir)

inference_speed = 0
max_memory_used = 0
start_time = time.time()  

for i in tqdm(range(len(dataset)), desc="Enhancing images"):
    input_image = dataset[i]
    input_tensor = input_image.unsqueeze(0).to(device)
    
    with torch.no_grad():  
        output_tensor = model(input_tensor)
        output_tensor = CCM(output_tensor)
    
    output_image = (output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
    output_image = np.clip(output_image, 0, 255)
    output_filepath = os.path.join(output_dir, dataset.image_files[i])
    cv2.imwrite(output_filepath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

print("Enhancement complete!")
