import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 贝叶斯卷积层实现
class BayesianConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BayesianConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 确保kernel_size是元组
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # 权重参数 (均值和 log 方差)
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)

        # 初始化
        self.reset_parameters()
        
        # KL散度
        self.kl_divergence = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=0.01)
        nn.init.constant_(self.weight_log_sigma, -7)  # 初始化为较小的方差
        if self.bias:
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_log_sigma, -7)

    def forward(self, input):
        # 重参数化采样
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu).to(self.weight_mu.device)

        if self.bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu).to(self.bias_mu.device)
        else:
            bias = None

        # 计算KL散度
        self.kl_divergence = self._calculate_kl()
        
        return F.conv3d(input, weight, bias, self.stride, self.padding)
    
    def _calculate_kl(self):
        """计算权重和偏置的KL散度"""
        # 权重的KL散度
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_log_sigma * 2) + self.weight_mu**2 - 2 * self.weight_log_sigma - 1
        )
        
        # 偏置的KL散度（如果有）
        if self.bias:
            kl_bias = 0.5 * torch.sum(
                torch.exp(self.bias_log_sigma * 2) + self.bias_mu**2 - 2 * self.bias_log_sigma - 1
            )
        else:
            kl_bias = 0
            
        return kl_weight + kl_bias

# 贝叶斯转置卷积层实现
class BayesianConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BayesianConvTranspose3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 确保kernel_size是元组
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # 权重参数 (均值和 log 方差)
        self.weight_mu = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)

        # 初始化
        self.reset_parameters()
        
        # KL散度
        self.kl_divergence = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=0.01)
        nn.init.constant_(self.weight_log_sigma, -7)  # 初始化为较小的方差
        if self.bias:
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_log_sigma, -7)

    def forward(self, input):
        # 重参数化采样
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu).to(self.weight_mu.device)

        if self.bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu).to(self.bias_mu.device)
        else:
            bias = None
            
        # 计算KL散度
        self.kl_divergence = self._calculate_kl()
        
        return F.conv_transpose3d(input, weight, bias, self.stride, self.padding)
    
    def _calculate_kl(self):
        """计算权重和偏置的KL散度"""
        # 权重的KL散度
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_log_sigma * 2) + self.weight_mu**2 - 2 * self.weight_log_sigma - 1
        )
        
        # 偏置的KL散度（如果有）
        if self.bias:
            kl_bias = 0.5 * torch.sum(
                torch.exp(self.bias_log_sigma * 2) + self.bias_mu**2 - 2 * self.bias_log_sigma - 1
            )
        else:
            kl_bias = 0
            
        return kl_weight + kl_bias

# 贝叶斯线性层实现
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # 权重参数 (均值和 log 方差)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            
        # 初始化
        self.reset_parameters()
        
        # KL散度
        self.kl_divergence = 0
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=0.01)
        nn.init.constant_(self.weight_log_sigma, -7)  # 初始化为较小的方差
        if self.bias:
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_log_sigma, -7)
            
    def forward(self, input):
        # 重参数化采样
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu).to(self.weight_mu.device)
        
        if self.bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu).to(self.bias_mu.device)
        else:
            bias = None
            
        # 计算KL散度
        self.kl_divergence = self._calculate_kl()
        
        return F.linear(input, weight, bias)
    
    def _calculate_kl(self):
        """计算权重和偏置的KL散度"""
        # 权重的KL散度
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_log_sigma * 2) + self.weight_mu**2 - 2 * self.weight_log_sigma - 1
        )
        
        # 偏置的KL散度（如果有）
        if self.bias:
            kl_bias = 0.5 * torch.sum(
                torch.exp(self.bias_log_sigma * 2) + self.bias_mu**2 - 2 * self.bias_log_sigma - 1
            )
        else:
            kl_bias = 0
            
        return kl_weight + kl_bias

# 层工厂函数
def create_conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_bayesian=False):
    """创建普通或贝叶斯3D卷积层"""
    if is_bayesian:
        return BayesianConv3d(in_channels, out_channels, kernel_size, stride, padding, bias)
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

def create_conv_transpose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_bayesian=False):
    """创建普通或贝叶斯3D转置卷积层"""
    if is_bayesian:
        return BayesianConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias)
    else:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

def create_linear(in_features, out_features, bias=True, is_bayesian=False):
    """创建普通或贝叶斯线性层"""
    if is_bayesian:
        return BayesianLinear(in_features, out_features, bias)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

# 灵活的残差块模块
class FlexibleResidualBlock(nn.Module):
    def __init__(self, in_channels, is_bayesian=False):
        super().__init__()
        self.conv1 = create_conv3d(in_channels, in_channels, kernel_size=3, padding=1, is_bayesian=is_bayesian)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = create_conv3d(in_channels, in_channels, kernel_size=3, padding=1, is_bayesian=is_bayesian)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.is_bayesian = is_bayesian
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
    
    def get_kl_divergence(self):
        """获取KL散度"""
        kl = 0
        if self.is_bayesian:
            if hasattr(self.conv1, 'kl_divergence'):
                kl += self.conv1.kl_divergence
            if hasattr(self.conv2, 'kl_divergence'):
                kl += self.conv2.kl_divergence
        return kl

# 灵活的空间注意力模块
class FlexibleSpatialAttention(nn.Module):
    def __init__(self, is_bayesian=False):
        super().__init__()
        self.conv = create_conv3d(2, 1, kernel_size=7, padding=3, is_bayesian=is_bayesian)
        self.is_bayesian = is_bayesian
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = torch.sigmoid(self.conv(combined))
        return x * att
    
    def get_kl_divergence(self):
        """获取KL散度"""
        kl = 0
        if self.is_bayesian and hasattr(self.conv, 'kl_divergence'):
            kl += self.conv.kl_divergence
        return kl

# 灵活的通道注意力模块
class FlexibleChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, is_bayesian=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            create_linear(channel, channel // reduction, is_bayesian=is_bayesian),
            nn.ReLU(inplace=True),
            create_linear(channel // reduction, channel, is_bayesian=is_bayesian),
            nn.Sigmoid()
        )
        self.is_bayesian = is_bayesian
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1, 1)
    
    def get_kl_divergence(self):
        """获取KL散度"""
        kl = 0
        if self.is_bayesian:
            for module in self.fc:
                if hasattr(module, 'kl_divergence'):
                    kl += module.kl_divergence
        return kl

# 灵活的ImprovedPhysNet网络
class FlexibleBayesianPhysNet(nn.Module):
    def __init__(self, frames=128, all_bayesian=False, layer_config=None):
        super().__init__()
        
        # 保存frames参数
        self.frames = frames
        
        # 初始化配置
        self.all_bayesian = all_bayesian
        self.layer_config = layer_config or {}
        
        # 初始化各个模块
        self.than = nn.Tanh()
        self._init_original_branch()
        self._init_diff_branch()
        self._init_fusion_module()
        self._init_encoder()
        self._init_decoder()
        
        # KL散度
        self.total_kl = 0
        
    def is_layer_bayesian(self, layer_name):
        """判断某层是否为贝叶斯层"""
        if layer_name in self.layer_config:
            return self.layer_config[layer_name]
        return self.all_bayesian
        
    def _init_original_branch(self):
        """初始化原始输入处理分支"""
        is_bayesian = self.is_layer_bayesian('original_branch')
        self.ConvBlock1 = nn.Sequential(
            create_conv3d(3, 16, [1,5,5], stride=1, padding=[0,2,2], is_bayesian=is_bayesian),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        
    def _init_diff_branch(self):
        """初始化差分输入处理分支"""
        is_bayesian = self.is_layer_bayesian('diff_branch')
        self.diff_conv = nn.Sequential(
            create_conv3d(3, 16, [1,3,3], padding=(0,1,1), is_bayesian=is_bayesian),
            FlexibleResidualBlock(16, is_bayesian=is_bayesian),
            FlexibleSpatialAttention(is_bayesian=is_bayesian)
        )
        
    def _init_fusion_module(self):
        """初始化特征融合模块"""
        is_bayesian = self.is_layer_bayesian('fusion_module')
        self.fusion_block = nn.Sequential(
            create_conv3d(32, 64, 3, padding=1, is_bayesian=is_bayesian),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            FlexibleChannelAttention(64, is_bayesian=is_bayesian),
            nn.Dropout3d(0.2)
        )
        
    def _init_encoder(self):
        """初始化编码器模块"""
        # 第二个卷积块
        is_bayesian = self.is_layer_bayesian('ConvBlock2')
        self.ConvBlock2 = nn.Sequential(
            create_conv3d(64, 32, [3,3,3], padding=1, is_bayesian=is_bayesian),
            FlexibleResidualBlock(32, is_bayesian=is_bayesian),
            FlexibleSpatialAttention(is_bayesian=is_bayesian)
        )
        self.res_conv1 = create_conv3d(64, 32, kernel_size=1, is_bayesian=is_bayesian)
        
        # 时空下采样
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        # 创建后续卷积块
        self._create_conv_blocks()
        
    def _create_conv_blocks(self):
        """创建编码器的卷积块"""
        conv_blocks = [
            (3, 32, 64),  # ConvBlock3: 32->64
            (4, 64, 64),
            (5, 64, 64),
            (6, 64, 64),
            (7, 64, 64),
            (8, 64, 64),
            (9, 64, 64)
        ]
        
        for idx, in_ch, out_ch in conv_blocks:
            block_name = f'ConvBlock{idx}'
            is_bayesian = self.is_layer_bayesian(block_name)
            
            setattr(self, block_name, nn.Sequential(
                create_conv3d(in_ch, out_ch, [3,3,3], padding=1, is_bayesian=is_bayesian),
                FlexibleResidualBlock(out_ch, is_bayesian=is_bayesian),
                FlexibleSpatialAttention(is_bayesian=is_bayesian)
            ))
            
            if in_ch != out_ch:
                setattr(self, f'res_conv{idx}', create_conv3d(in_ch, out_ch, kernel_size=1, is_bayesian=is_bayesian))
    
    def _init_decoder(self):
        """初始化解码器模块"""
        # 上采样模块
        is_bayesian_upsample = self.is_layer_bayesian('upsample')
        self.upsample = nn.Sequential(
            create_conv_transpose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0], is_bayesian=is_bayesian_upsample),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        
        is_bayesian_upsample2 = self.is_layer_bayesian('upsample2')
        self.upsample2 = nn.Sequential(
            create_conv_transpose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0], is_bayesian=is_bayesian_upsample2),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        
        # 空间池化
        self.poolspa = nn.AdaptiveAvgPool3d((self.frames,1,1))
        
        # 最终输出层
        is_bayesian_output = self.is_layer_bayesian('output')
        self.ConvBlock10 = create_conv3d(64, 1, [1,1,1], stride=1, padding=0, is_bayesian=is_bayesian_output)
    
    def _process_original_input(self, x):
        """处理原始输入"""
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        return x
    
    def _process_diff_input(self, x):
        """处理差分输入"""
        x = self.diff_conv(x)
        x = self.MaxpoolSpa(x)
        return x
    
    def _fuse_features(self, x1, x2):
        """融合特征"""
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion_block(x)
        return x
    
    def _encode_block2(self, x):
        """编码器Block2处理"""
        residual = self.res_conv1(x)
        x = self.ConvBlock2(x)
        x += residual
        x = nn.ReLU()(x)
        return x
    
    def _encode_block3(self, x):
        """编码器Block3处理"""
        x = self.MaxpoolSpaTem(x)
        residual = x
        x = self.ConvBlock3(x)
        if hasattr(self, 'res_conv3'):
            residual = self.res_conv3(residual)
        x += residual
        x = nn.ReLU()(x)
        return x
    
    def _process_blocks(self, x, block_ids):
        """处理多个编码器块"""
        for idx in block_ids:
            block = getattr(self, f'ConvBlock{idx}')
            residual = x
            x = block(x)
            # 处理通道变化
            if x.shape[1] != residual.shape[1]:
                residual = getattr(self, f'res_conv{idx}')(residual)
            x += residual
            x = nn.ReLU()(x)
            # 根据需要进行下采样
            if idx in [3,5]:
                x = self.MaxpoolSpaTem(x)
        return x
    
    def _decode(self, x):
        """解码器处理"""
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ConvBlock10(x)
        return x
    
    def _calculate_kl_divergence(self):
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        num_bayesian_params = 0
        
        # 遍历所有模块
        for name, module in self.named_modules():
            if hasattr(module, 'get_kl_divergence'):
                kl = module.get_kl_divergence()
                if not isinstance(kl, torch.Tensor):
                    kl = torch.tensor(kl, device=total_kl.device)
                total_kl += kl
                
                # 统计参数数量
                if hasattr(module, 'weight_mu'):
                    num_bayesian_params += module.weight_mu.numel()
                if hasattr(module, 'bias_mu') and module.bias_mu is not None:
                    num_bayesian_params += module.bias_mu.numel()
            elif hasattr(module, 'kl_divergence'):
                kl = module.kl_divergence
                if not isinstance(kl, torch.Tensor):
                    kl = torch.tensor(kl, device=total_kl.device)
                total_kl += kl
                
                # 统计参数数量
                if hasattr(module, 'weight_mu'):
                    num_bayesian_params += module.weight_mu.numel()
                if hasattr(module, 'bias_mu') and module.bias_mu is not None:
                    num_bayesian_params += module.bias_mu.numel()
                    
        # 归一化 KL 散度
        if num_bayesian_params > 0:
            total_kl = total_kl / num_bayesian_params
            
        return total_kl
    
    def forward(self, input_original, input_diff, num_samples=1):
        """
        双输入前向传播
        参数:
            input_original: 原始输入 [batch, 3, 128, 64, 64]
            input_diff: 差分输入 [batch, 3, 128, 64, 64]
            num_samples: 采样次数（用于不确定性估计）
        返回:
            如果num_samples=1: rPPG信号 [batch, 128]
            如果num_samples>1: (均值 [batch, 128], 方差 [batch, 128])
        """
        if num_samples == 1:
            # 单次前向传播
            # 原始输入处理分支
            x1 = self._process_original_input(input_original)
            
            # 差分输入处理分支
            x2 = self._process_diff_input(input_diff)
            
            # 特征融合
            x = self._fuse_features(x1, x2)
            
            # 编码器处理
            x = self._encode_block2(x)
            x = self._encode_block3(x)
            x = self._process_blocks(x, [4,5,6,7,8,9])
            
            # 解码器处理
            x = self._decode(x)
            x = self.than(x)
            # 计算KL散度
            self.total_kl = self._calculate_kl_divergence()
            
            # 输出处理
            return x.view(-1, x.size(2))
        else:
            # 多次采样用于不确定性估计
            outputs = []
            for _ in range(num_samples):
                # 单次前向传播
                output = self.forward(input_original, input_diff, num_samples=1)
                outputs.append(output)
                
            # 计算均值和方差
            outputs = torch.stack(outputs, dim=0)
            mean = outputs.mean(dim=0)
            variance = outputs.var(dim=0)
            
            return mean, variance
    
    def set_layer_type(self, layer_name, is_bayesian):
        """
        设置某层的类型（普通或贝叶斯）
        注意：这只会影响下一次创建模型时的配置，不会修改当前模型
        """
        self.layer_config[layer_name] = is_bayesian
        
    def get_config(self):
        """获取当前配置"""
        config = {
            'all_bayesian': self.all_bayesian,
            'layer_config': self.layer_config.copy()
        }
        return config
    
    def save_config(self, path):
        """保存配置到文件"""
        import json
        config = self.get_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_config(cls, path):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config = json.load(f)
        return config

# 测试函数
def test_flexible_bayesian_physnet():
    """
    测试灵活贝叶斯PhysNet网络
    """
    # 创建一个全部使用普通层的模型
    model1 = FlexibleBayesianPhysNet()
    
    # 创建一个全部使用贝叶斯层的模型
    model2 = FlexibleBayesianPhysNet(all_bayesian=True)
    
    # 创建一个部分使用贝叶斯层的模型
    bayesian_config = {
        'ConvBlock1': True,  # 第一个卷积块使用贝叶斯层
        'ConvBlock2': False, # 第二个卷积块使用普通层
        'ConvBlock3': True,  # 第三个卷积块使用贝叶斯层
    }
    model3 = FlexibleBayesianPhysNet(layer_config=bayesian_config)
    
    # 设置批量大小
    batch_size = 2
    
    # 创建随机输入数据
    input_original = torch.randn(batch_size, 3, 128, 64, 64)
    input_diff = torch.randn(batch_size, 3, 128, 64, 64)
    
    # 测试普通前向传播
    output1 = model1(input_original, input_diff)
    print(f"普通模型输出形状: {output1.shape}")
    print(f"普通模型KL散度: {model1.total_kl}")
    
    # 测试贝叶斯前向传播
    output2 = model2(input_original, input_diff)
    print(f"贝叶斯模型输出形状: {output2.shape}")
    print(f"贝叶斯模型KL散度: {model2.total_kl}")
    
    # 测试混合模型前向传播
    output3 = model3(input_original, input_diff)
    print(f"混合模型输出形状: {output3.shape}")
    print(f"混合模型KL散度: {model3.total_kl}")
    
    # 测试不确定性估计
    mean, variance = model2(input_original, input_diff, num_samples=5)
    print(f"均值形状: {mean.shape}")
    print(f"方差形状: {variance.shape}")
    
    print("测试通过！")

if __name__ == "__main__":
    test_flexible_bayesian_physnet()
