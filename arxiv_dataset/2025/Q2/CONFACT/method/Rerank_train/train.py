import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer, AutoModel
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaCredibilityDataset(Dataset):
    """自定义数据集，用于媒体可信度预测"""
    def __init__(self, texts: List[str], labels: List[float], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class MediaCredibilityPredictor(nn.Module):
    """基于 BigBird-RoBERTa 的模型用于预测媒体可信度评分"""
    def __init__(self, model_name: str = 'google/bigbird-roberta-large', dropout_rate: float = 0.1):
        super().__init__()
        # 加载 BigBird-RoBERTa 模型
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = 1024  # BigBird-RoBERTa 隐藏层大小
        
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 获取 BigBird-RoBERTa 输出
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用 last_hidden_state 作为句子表示
        last_hidden_state = outputs.last_hidden_state
        
        # 获取 [CLS] token 的表示（即第一列）
        cls_embedding = last_hidden_state[:, 0]
        
        # 经过 Dropout 和回归层得到预测结果
        pooled_output = self.dropout(cls_embedding)
        prediction = self.regressor(pooled_output)   
        return prediction


class CredibilityTrainer:
    """带有混合精度训练支持的训练类"""
    def __init__(
        self,
        model: MediaCredibilityPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        distributed: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(
            [
                {"params": model.roberta.parameters(), "lr": learning_rate},
                {"params": model.regressor.parameters(), "lr": learning_rate * 10}
            ],
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = GradScaler()
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.distributed = distributed
        
        if self.distributed:
            self.model = DDP(self.model, device_ids=[device])

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 混合精度训练
            with autocast():
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                if self.global_step < self.warmup_steps:
                    # 线性预热
                    lr_scale = min(1., float(self.global_step) / float(self.warmup_steps))
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = pg['lr'] * lr_scale

            total_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})

        return total_loss / len(self.train_loader)

    def evaluate(self) -> Tuple[float, float, float]:
        """计算验证集的损失、准确度（Acc）"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                
                # 计算预测值与标签的差值，并判断差值是否小于0.05
                predicted_labels = outputs.squeeze()  # 获取模型输出
                diff = torch.abs(predicted_labels - labels)  # 计算预测值与标签的绝对差值
                print(predicted_labels)
                print(labels)
                correct_predictions += (diff < 0.25).sum().item()  # 如果差值小于0.05，视为正确
                total_predictions += len(labels)

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy



def save_model(model, epoch: int, model_dir: str):
    """保存模型到指定路径，按epoch编号命名"""
    model_path = f"{model_dir}/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model for epoch {epoch} saved to {model_path}")


def read_and_prepare_data(csv_file: str) -> Tuple[List[str], List[float]]:
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 提取 'details' 列作为文本输入和 'credibility' 列作为标签
    texts = df['details'].tolist()
    labels = df['credibility'].values  # 取出原始标签
    
    # 获取标签的最小值和最大值
    min_value = labels.min()
    max_value = labels.max()
    
    # 映射标签到 [0, 1] 区间
    labels_mapped = (labels - min_value) / (max_value - min_value)
    print(labels_mapped)
    return texts, labels_mapped


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
def train_model_from_csv(csv_file: str, num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5, max_length: int = 1024, warmup_steps: int = 50, model_dir: str = "models", distributed: bool = False):
    # 读取数据并准备训练集和验证集
    texts, labels = read_and_prepare_data(csv_file)
    
    # 按 10:1 的比例切分训练和验证数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    # 初始化 Tokenizer 和数据集
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-large')
    train_dataset = MediaCredibilityDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MediaCredibilityDataset(val_texts, val_labels, tokenizer, max_length)
    
    # 初始化 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = MediaCredibilityPredictor()

    # 创建训练器
    trainer = CredibilityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        distributed=distributed
    )
    
    # 训练和评估模型
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch()
        val_loss, val_acc = trainer.evaluate()
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        save_model(model, epoch, model_dir)
    
    return model


if __name__ == "__main__":

    csv_file = "mfbc_media_desc.csv"
    
    model_save_dir = "/home/yuhao/UncertainQA/Rerank_train/Save_model"
    
    trained_model = train_model_from_csv(csv_file, num_epochs=5, model_dir=model_save_dir, distributed=False)
