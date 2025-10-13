import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io
from api_dataloader.dataloader_ns import NSDataset 
from network import SimVP_Model,ResNet,U_net
import numpy as np
from tqdm import tqdm  
import logging  
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

# Set up logging
LOG_FILE = 'log.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration parameters
DATA_PATH = '../NS_data/NavierStokes_V1e-5_N1200_T20.mat'
BATCH_SIZE = 10
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
SCALING_COEFF = 0.1
CHECKPOINT_DIR = 'outputs/Unet_T_simvp_van_T_simvp_uniformer'
LOG_DIR = 'logs/'
RESULTS_DIR = 'results/'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(LOG_DIR)

# Data loaders
def get_dataloaders(path_file, batch_size=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset = NSDataset(path_file=path_file)
    total_size = len(dataset)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Ensure consistent splits
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders(
    path_file=DATA_PATH,
    batch_size=BATCH_SIZE
)



class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    def __init__(self, feat_num, margin=1.0):
        super(ABLoss, self).__init__()
        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, g_s, g_t):
        bsz = g_s[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(g_s, g_t)]
        losses = [w * l for w, l in zip(self.w, losses)]
        # loss = sum(losses) / bsz
        # loss = loss / 1000 * 3
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        # return losses
        return torch.mean(torch.stack(losses))

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()





t_model_1 = SimVP_Model(in_shape=(10, 1, 64, 64),model_type='uniformer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_model_1.load_state_dict(torch.load('outputs/simvp_uniformer/best.pth'))
t_model_1.to(device)

t_model_2 = SimVP_Model(in_shape=(10, 1, 64, 64),model_type='van')
t_model_2.load_state_dict(torch.load('outputs/simvp_van/best.pth'))
t_model_2.to(device)


s_model =U_net(input_channels=10,output_channels=10, kernel_size=3, dropout_rate=0.2)
s_model.to(device)
logger.info(f'Model moved to device: {device}')

# Loss and optimizer
criterion = nn.MSELoss()
fea_criterion = ABLoss(1,margin=1.0).cuda()


optimizer = optim.Adam(s_model.parameters(), lr=LEARNING_RATE)
logger.info('Loss function and optimizer initialized.')

# Training function
def train_epoch(model_t1,model_t2,model_s, dataloader, criterion, optimizer, device):
    model_t1.eval()
    model_t2.eval()
    model_s.train()
    running_loss = 0.0
    # total_samples = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        with torch.no_grad():
            outputs_t1,middle_t1 = model_t1(inputs)
            outputs_t2,middle_t2 = model_t2(inputs)
        outputs,middle_s = model_s(inputs)
        # loss = criterion(outputs, targets)


        ##### 单教师Uniformer logit蒸馏 
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t1, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean')


        ##### 多教师 logit 蒸馏  mse Loss
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t1, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean')

        ##### 单教师 van logit 蒸馏  mse Loss
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean')

     

        ##### 双教师中间特征取平均权重
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t1, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean') +F.mse_loss(middle_s, middle_t1, reduction='mean')+F.mse_loss(middle_s, middle_t2, reduction='mean')


        ##### 单教师van 特征蒸馏  mse Loss
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean')+F.mse_loss(middle_s, middle_t2, reduction='mean')

        ##### 单教师Uniformer 特征蒸馏  mse Loss
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t1, reduction='mean')+F.mse_loss(middle_s, middle_t1, reduction='mean')


        ##### 双教师中间特征 ABloss
        # loss = F.mse_loss(outputs, targets, reduction='mean')+F.mse_loss(outputs, outputs_t1, reduction='mean')+F.mse_loss(outputs, outputs_t2, reduction='mean') +fea_criterion(middle_s, middle_t1)+fea_criterion(middle_s, middle_t2)


        #### AEKD
        # 计算学生模型与每个教师模型之间的损失
        loss_t1 = F.mse_loss(outputs, outputs_t1) + fea_criterion(middle_s, middle_t1)
        loss_t2 = F.mse_loss(outputs, outputs_t2) + fea_criterion(middle_s, middle_t2)
        # 初始化列表以存储梯度和损失
        grads = []
        losses = [loss_t1, loss_t2]
        # 注册钩子函数以捕获学生模型输出的梯度
        def hook_fn(grad):
            grads.append(grad.clone().detach())
        hook_handle = outputs.register_hook(hook_fn)
        # 对每个损失进行反向传播，计算梯度
        for loss in losses:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
        # 计算最优加权系数
        weights = find_optimal_svm(torch.stack(grads),
                                     nu=1/0.6/2,
                                     gpu_id=0,
                                     is_norm=False)
        total_loss = sum(w * l for w, l in zip(weights, losses))
        loss = F.mse_loss(outputs, targets, reduction='mean') + total_loss
        optimizer.zero_grad()

        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples =0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs,_ = model(inputs)
            # loss = criterion(outputs, targets)
            loss = nn.functional.mse_loss(outputs, targets, reduction='mean')
            running_loss += loss.item()* inputs.size(0)
            total_samples += inputs.size(0)
    
    # return running_loss / len(dataloader)
    return running_loss / total_samples

# Testing function
def test_model(model, dataloader, device, save_path):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Save results as NPZ
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    np.savez(save_path, predictions=all_predictions, targets=all_targets)
    logger.info(f"Test results saved to {save_path}")

# Training and validation loop
best_val_loss = float('inf')
for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
    logger.info(f'Epoch [{epoch}/{NUM_EPOCHS}]')
    
    train_loss = train_epoch(t_model_1,t_model_2,s_model,train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(s_model, val_loader, criterion, device)
    
    logger.info(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Best Loss:{best_val_loss:.4f}')
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_{epoch}.pth')
        torch.save(s_model.state_dict(), checkpoint_path)
        logger.info(f'Best model saved to {checkpoint_path}')
    


# Close TensorBoard writer
writer.close()
logger.info('TensorBoard writer closed.')

