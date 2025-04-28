import torch
from torch.utils.data import DataLoader,Dataset,Sampler
import numpy as np
import torch.optim as optim
from models.embedding import *
from models.backbone import *
from models.predictor import *
from loss.make_loss import *
from torch.nn import Module
import argparse; 
from torch.cuda.amp import autocast as autocast

device1 = torch.device("cuda:1")
device2 = torch.device("cuda:0")


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',default="./datas/merged_mu-_with_10m_water.npy")
parser.add_argument('--ckpt_name',default="mu-_10m_ckpt")
parser.add_argument('--mean', type=float,default= 3.1541)
parser.add_argument('--len_data',type=int,default=1845644)

args = parser.parse_args()

BATCH_SIZE=300
BATCH_SIZE_EVAL=300
BATCH_EVENT_NUM=1024
LEN_DATASET=1000
# LEN_DATA=18787640
# LEN_DATA=19311090
LEN_DATA=args.len_data
LATENT_DIM=100

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0,checkpoint_name='mu-_10m_ckpt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_name=checkpoint_name

    def __call__(self, val_loss, model,optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt={'net':model.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(ckpt,self.checkpoint_name)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

class TransformerGeneratorModel(Module):
    
    def __init__(self,
                 latent_dim: int,
                 quantities_num: int,
                 embedding_num: int,
                 nhead: int,
                 dff: int,
                 norm: str,
                 activation: 'str',
                 num_layers: int,
                 ):
        super().__init__()
        self.embedding = LinearEmbedding(latent_num=latent_dim,
                                         quantities_num=quantities_num,
                                         embedding_num=embedding_num,
                                         norm=norm,
                                         activation=activation)
        self.backbone = TransformerBlock(d_model=embedding_num,
                                         nhead=nhead,
                                         dff=dff,
                                         activation=activation,
                                         num_layers=num_layers)
        self.predictor = LinearPredictor(embedding_num=embedding_num)
        """
        self.model = nn.Sequential(
            LinearEmbedding(latent_num=latent_dim, quantities_num=quantities_num, embedding_num=embedding_num,
                            norm=norm,
                            activation=activation),
            TransformerBlock(d_model=embedding_num, nhead=nhead, dff=dff, activation=activation, num_layers=num_layers),
            LinearPredictor(embedding_num=embedding_num)
        )
        """

    def forward(self, x):
        x = self.embedding(
            x
        )  # (batch size, 100) -> (batch size, 3 * 1024) -> (batch size, 3, 1024) linear
        x = self.backbone(
            x)  # (batch size, 3, 1024) -> (batch size, 3, 1024) transformer
        x = self.predictor(x)
        return x


class IterableDataset(Dataset):
    
    def __init__(self,len_dataset,data_path,mean,latent_dim=LATENT_DIM):
        self.latent_dim=latent_dim
        self.data_path=data_path
        # self.data= np.memmap(self.data_path, dtype="float32", mode="r", shape=(LEN_DATA, 3))
        self.data=np.load(self.data_path)#.to(device1)
        
        #self.data=torch.from_numpy(self.data)
        self.len_dataset = len_dataset
        self.mean=mean

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        input = idx[:self.latent_dim]
        data=self.data[idx[self.latent_dim:].long()]
        target=torch.tensor(data,dtype=torch.float32)
        #对target的theta与phi做tanh放缩
        #target[:,2]=torch.tanh(target[:,2])
        # energy=target[:,0]
        # lmbda = 9  # 用于调整峰度的参数
        # energy = special.inv_boxcox(energy, lmbda)
        # target[:,0] = torch.tanh((energy-self.mean))
        # target=torch.randn(1024,1,dtype=torch.float32)
        return input,target
        


class IterRandomSampler(Sampler):

    def __init__(self, len_data, batch_event_num, len_dataset,latent_dim=LATENT_DIM):
        self.batch_event_num = batch_event_num
        self.len_data = len_data
        self.len_dataset = len_dataset
        self.latent_dim=latent_dim

    def __iter__(self):
        #make sure the input is equal distributed in one epoch
        latent_index=torch.rand(self.len_dataset,self.latent_dim,
                                  dtype=torch.float32)*2-1
        data_index = torch.randint(high=self.len_data,
                                   size=(self.len_dataset,
                                         self.batch_event_num),
                                   dtype=torch.long)
        return iter(torch.cat([latent_index, data_index], dim=-1))

    def __len__(self):
        return self.len_dataset


def train(model, train_loader, device,ckpt_name):
    torch.set_float32_matmul_precision('high')
    #从上一次开始训练
    # ckpt=torch.load(ckpt_name)
    
    #model = nn.DataParallel(model, device_ids=[device1, device2])
    model.to(device)
    #model = torch.compile(model)
    # model.load_state_dict(ckpt['net'])
    
    # 定义损失函数和优化器
    criterion = SWDLoss(gamma=0.1)
    early_stop=EarlyStopping(verbose=True,delta=0.00,patience=20,checkpoint_name=ckpt_name)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    # optimizer.load_state_dict(ckpt['optimizer'])
    
    # 训练模型
    for epoch in range(1000):
        model.train()
        train_loss = 0.0
        for i, (input,targets) in enumerate(train_loader):
            input,target=input.to(device),targets.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i>0 and i % 100 == 0:
                train_loss_print = train_loss / (100*i)
                print(f'train_loss: {train_loss_print:.6f}')
        train_loss /= len(train_loader)

        # 输出每个 epoch 的训练结果
        print(f'Epoch [{epoch + 1}/100], '
              f'train_loss: {train_loss:.6f}')
        early_stop(train_loss,model,optimizer)
        if early_stop.early_stop:
            print('Early stopping!')
            break

if __name__ == '__main__':
    LEN_DATA=args.len_data-1
    #torch.set_float32_matmul_precision('high')
    device=torch.device('cuda:1')
    train_loader = DataLoader(dataset=IterableDataset(LEN_DATASET*BATCH_SIZE,args.data_path,args.mean), 
                              sampler=IterRandomSampler(len_data=LEN_DATA,batch_event_num=BATCH_EVENT_NUM,len_dataset=LEN_DATASET*BATCH_SIZE),
                              batch_size=BATCH_SIZE, drop_last=True,num_workers=1)
    
    model=TransformerGeneratorModel(latent_dim=LATENT_DIM,quantities_num=3,embedding_num=1024,nhead=8,dff=2048,norm='ln',activation='gelu',num_layers=2)
    train(model=model, train_loader=train_loader,  device=device,ckpt_name=args.ckpt_name)