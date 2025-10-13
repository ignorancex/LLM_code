import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes, Chans=4, Samples=2560, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Block 2
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.batchnorm3 = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Classification
        self.classify = nn.Linear(F2 * (Samples // 32), num_classes)

    def forward(self, x):
        # Block 1
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out

class ResNet_1D_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling, reduction_ratio=16):
        super(ResNet_1D_CBAM_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling
        self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.cbam = CBAM(out_channels, reduction_ratio)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.cbam(out)
        if self.downsampling:
            identity = self.downsample_layer(x)
        out += identity
        return out
    
class ResNet_1D_Depthwise_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling, reduction_ratio=16):
        super(ResNet_1D_Depthwise_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.depthwise_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.depthwise_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, groups=out_channels, bias=False)
        self.pointwise_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling
        self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.cbam = CBAM(out_channels, reduction_ratio)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.depthwise_conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        
        out = self.maxpool(out)
        out = self.cbam(out)
        if self.downsampling:
            identity = self.downsample_layer(x)
        out += identity
        return out
    
    
class UniPhyNetSingle(nn.Module):
    # def __init__(self, kernels=[3,5,7,9], num_feature_maps = 64,res_blocks = 8, in_channels=4, fixed_kernel_size=5, num_classes=2):
    def __init__(self, kernels=[3],samples = 2560, num_feature_maps = 64,res_blocks = 8, in_channels=4, fixed_kernel_size=5, num_classes=2):
        super(UniPhyNetSingle, self).__init__()
        self.kernels = kernels
        self.planes = num_feature_maps
        self.in_channels = in_channels
        self.parallel_conv = nn.ModuleList()
        for kernel_size in kernels:
            conv = nn.Conv1d(in_channels, self.planes, kernel_size, stride=1, padding=0, bias=False)
            self.parallel_conv.append(conv)
        self.bn1 = nn.BatchNorm1d(self.planes)
        self.relu = nn.SiLU(inplace=False)
        self.conv1 = nn.Conv1d(self.planes, self.planes, fixed_kernel_size, stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(fixed_kernel_size,blocks = res_blocks, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(self.planes)
        self.avgpool = nn.AvgPool1d(6, stride=6, padding=2)
        self.rnn = nn.GRU(self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)

        initial_len = len(self.kernels)*samples - sum(self.kernels) + len(self.kernels)
        conv1_out_len = initial_len // 2
        resnet_out_len = conv1_out_len // 2**res_blocks
        avgpool_out_len = (resnet_out_len - 6 + 2 * 2) // 6 + 1
        flattened_size = self.planes * avgpool_out_len
        self.fc = nn.Linear(flattened_size+2*128, num_classes)


    def _make_resnet_layer(self, kernel_size, stride, blocks, padding=0):
        layers = []
        for _ in range(blocks):
          layers.append(ResNet_1D_CBAM_Block(self.planes, self.planes, kernel_size, stride, padding, downsampling = True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = [conv(x) for conv in self.parallel_conv]
        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]
        new_out = torch.cat([out, new_rnn_h], dim=1)


        result = self.fc(new_out)
        return result
    
    
class FeatureExtractor(nn.Module):
    def __init__(self, kernels, samples,num_feature_maps, in_channels, fixed_kernel_size, res_blocks):
        super(FeatureExtractor, self).__init__()
        self.kernels = kernels
        self.planes = num_feature_maps
        self.in_channels = in_channels
        self.parallel_conv = nn.ModuleList()
        for kernel_size in kernels:
            conv = nn.Conv1d(in_channels, self.planes, kernel_size, stride=1, padding=0, bias=False)
            self.parallel_conv.append(conv)
        self.bn1 = nn.BatchNorm1d(self.planes)
        self.relu = nn.SiLU(inplace=False)
        self.conv1 = nn.Conv1d(self.planes, self.planes, fixed_kernel_size, stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(fixed_kernel_size, stride=1, blocks=res_blocks, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(self.planes)
        self.avgpool = nn.AvgPool1d(6, stride=6, padding=2)

        initial_len = len(self.kernels) * samples - sum(self.kernels) + len(self.kernels)
        conv1_out_len = initial_len // 2
        resnet_out_len = conv1_out_len // 2**res_blocks
        avgpool_out_len = (resnet_out_len - 6 + 2 * 2) // 6 + 1
        self.flattened_size = self.planes * avgpool_out_len

    def _make_resnet_layer(self, kernel_size, stride, blocks, padding=0):
        layers = []
        for _ in range(blocks):
            layers.append(ResNet_1D_CBAM_Block(self.planes, self.planes, kernel_size, stride, padding, downsampling=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = [conv(x) for conv in self.parallel_conv]
        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        return out

class UniPhyNetMulti(nn.Module):
    def __init__(self,kernels_list = [[3],[5],[13]],samples_list = [2560,5120,1280],num_feature_maps_list = [32,32,32],in_channels_list = [4,3,3],res_blocks_list = [7,8,7], num_classes=2):
        super(UniPhyNetMulti, self).__init__()
        self.type = type
        # self.eeg_net = FeatureExtractor(kernels = [3,5,7,9], samples = 2560,num_feature_maps = 32, in_channels = 4, fixed_kernel_size = 5, res_blocks = 8)
        # self.ecg_net = FeatureExtractor(kernels = [5,9,11,15], samples = 5120,num_feature_maps = 32, in_channels = 3, fixed_kernel_size = 5, res_blocks = 9)
        self.eeg_net = FeatureExtractor(kernels = kernels_list[0], samples = samples_list[0],num_feature_maps = num_feature_maps_list[0], in_channels = in_channels_list[0], fixed_kernel_size = 5, res_blocks = res_blocks_list[0])
        self.ecg_net = FeatureExtractor(kernels = kernels_list[1], samples = samples_list[1],num_feature_maps = num_feature_maps_list[1], in_channels = in_channels_list[1], fixed_kernel_size = 5, res_blocks = res_blocks_list[1])
        self.eda_net = FeatureExtractor(kernels = kernels_list[2], samples = samples_list[2],num_feature_maps = num_feature_maps_list[2], in_channels = in_channels_list[2], fixed_kernel_size = 5, res_blocks = res_blocks_list[2])

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.3)
        total_flattened_size = (self.eeg_net.flattened_size + self.ecg_net.flattened_size+self.eda_net.flattened_size)
        self.gru = nn.GRU(total_flattened_size, hidden_size=128, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(2 * 128+total_flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, eeg, ecg, eda):
        eeg_features = self.eeg_net(eeg)
        ecg_features = self.ecg_net(ecg)
        eda_features = self.eda_net(eda)

        combined_features = torch.cat((eeg_features, ecg_features,eda_features), dim=1)

        combined_features_unsqueezed = combined_features.unsqueeze(1).repeat(1, eeg.size(-1), 1)
        rnn_out, _ = self.gru(combined_features_unsqueezed)
        rnn_out = rnn_out[:, -1, :]

        new_out = torch.cat([combined_features, rnn_out], dim=1)
        new_out = self.fc1(new_out)
        new_out = self.silu(new_out)
        result = self.fc2(new_out)
        return result
    
    
    
class UniPhyNetDouble(nn.Module):
    def __init__(self,kernels_list = [[3],[5]],samples_list = [2560,5120],num_feature_maps_list = [32,32],in_channels_list = [4,3],res_blocks_list = [7,8], num_classes=2):
        super(UniPhyNetDouble, self).__init__()
        self.type = type
        # self.eeg_net = FeatureExtractor(kernels = [3,5,7,9], samples = 2560,num_feature_maps = 32, in_channels = 4, fixed_kernel_size = 5, res_blocks = 8)
        # self.ecg_net = FeatureExtractor(kernels = [5,9,11,15], samples = 5120,num_feature_maps = 32, in_channels = 3, fixed_kernel_size = 5, res_blocks = 9)
        self.eeg_net = FeatureExtractor(kernels = kernels_list[0], samples = samples_list[0],num_feature_maps = num_feature_maps_list[0], in_channels = in_channels_list[0], fixed_kernel_size = 5, res_blocks = res_blocks_list[0])
        self.ecg_net = FeatureExtractor(kernels = kernels_list[1], samples = samples_list[1],num_feature_maps = num_feature_maps_list[1], in_channels = in_channels_list[1], fixed_kernel_size = 5, res_blocks = res_blocks_list[1])
        # self.eda_net = FeatureExtractor(kernels = kernels_list[2], samples = samples_list[2],num_feature_maps = num_feature_maps_list[2], in_channels = in_channels_list[2], fixed_kernel_size = 5, res_blocks = res_blocks_list[2])

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(0.3)
        total_flattened_size = (self.eeg_net.flattened_size + self.ecg_net.flattened_size)
        self.gru = nn.GRU(total_flattened_size, hidden_size=128, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(2 * 128+total_flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, eeg, ecg):
        eeg_features = self.eeg_net(eeg)
        ecg_features = self.ecg_net(ecg)


        combined_features = torch.cat((eeg_features, ecg_features), dim=1)

        combined_features_unsqueezed = combined_features.unsqueeze(1).repeat(1, eeg.size(-1), 1)
        rnn_out, _ = self.gru(combined_features_unsqueezed)
        rnn_out = rnn_out[:, -1, :]

        new_out = torch.cat([combined_features, rnn_out], dim=1)
        new_out = self.fc1(new_out)
        new_out = self.silu(new_out)
        result = self.fc2(new_out)
        return result