import torch
import torch.nn as nn
import torch.nn.functional as F


# original method
'''
# from keras which is    (clip_frames, width, height, in_channels)
# to pytorch which is    (in_channels, clip_frames, width, height)
class FGN_RGB(nn.Module):
    def __init__(self, args):
        super(FGN_RGB, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(16)
        
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(32)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm7 = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm8 = nn.BatchNorm3d(32)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1)) 
        # ADDED    
        self.dropout3d = nn.Dropout3d(p=args.dropout3d)


    def forward (self, rgb):
        ################# RGB Block
        rgb = self.conv1(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm1(rgb)
        #rgb = self.dropout3d(rgb)
        rgb = self.conv2(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm2(rgb)
        rgb = self.maxpool1(rgb)
        #rgb = self.dropout3d(rgb)
        
        rgb = self.conv3(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm3(rgb)
        #rgb = self.dropout3d(rgb)
        rgb = self.conv4(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm4(rgb)
        rgb = self.maxpool1(rgb)
        #rgb = self.dropout3d(rgb)

        
        rgb = self.conv5(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm5(rgb)
        rgb = self.conv6(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm6(rgb)
        rgb = self.maxpool1(rgb)
        
        rgb = self.conv7(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm7(rgb)
        rgb = self.conv8(rgb)
        rgb = self.relu(rgb)
        #rgb = self.batchnorm8(rgb)
        rgb = self.maxpool1(rgb)
        return rgb

class FGN_FLOW(nn.Module):
    def __init__(self, args):
        super(FGN_FLOW, self).__init__()
        self.conv1f = nn.Conv3d(2, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm4 = nn.BatchNorm3d(16)
        
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm6 = nn.BatchNorm3d(32)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm7 = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        #self.batchnorm8 = nn.BatchNorm3d(32)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1))
        
        # ADDED    
        self.dropout3d = nn.Dropout3d(p=args.dropout3d)
        
    def forward (self, flow):
        ################## Optical flow block
        flow = self.conv1f(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm1(flow)
        #flow = self.dropout3d(flow)
        flow = self.conv2(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm2(flow)
        flow = self.maxpool1(flow)
        #flow = self.dropout3d(flow)

        flow = self.conv3(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm3(flow)
        #flow = self.dropout3d(flow)
        flow = self.conv4(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm4(flow)
        flow = self.maxpool1(flow)
        #flow = self.dropout3d(flow)

        
        flow = self.conv5(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm5(flow)
        flow = self.conv6(flow)
        flow = self.relu(flow)
        #flow = self.batchnorm6(flow)
        flow = self.maxpool1(flow)
        
        flow = self.conv7(flow)
        flow = self.sigmoid(flow)
        #flow = self.batchnorm7(flow)
        flow = self.conv8(flow)
        flow = self.sigmoid(flow)
        #flow = self.batchnorm8(flow)
        flow = self.maxpool1(flow)

        
        return flow


class FGN_MERGE_CLASSIFY(nn.Module):
    def __init__(self, args):
        super(FGN_MERGE_CLASSIFY, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1))
        
        self.conv1m = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.conv2m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.conv3m = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(64)
        self.conv4m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5m = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.conv6m = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(128)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 3, 3))
        
        # Invece di questo, altrimenti si riduce a zero la depth!
        #self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, 101) # 51 invece di 1 per HMDB51, 101 per UCF101
        #self.fc3 = nn.Linear(32,51)
        self.dropout = nn.Dropout(args.dropout)
        
    
    def forward (self, rgb, flow):
        ########### Fusion and pooling
        x = flow * rgb
        x = self.maxpool2(x)
    
        ########## Merging block
        x = self.conv1m(x)
        x = self.relu(x)
        #x = self.batchnorm1(x)
        x = self.conv2m(x)
        x = self.relu(x)
        #x = self.batchnorm2(x)
        x = self.maxpool3(x)
        x = self.conv3m(x)
        x = self.relu(x)
        #x = self.batchnorm3(x)
        x = self.conv4m(x)
        x = self.relu(x)
        #x = self.batchnorm4(x)
        x = self.maxpool3(x)

        x = self.conv5m(x)
        x = self.relu(x)
        #x = self.batchnorm5(x)
        x = self.conv6m(x)
        x = self.relu(x)
        #x = self.batchnorm6(x)
        x = self.maxpool4(x)

        ########## Classifier
        # No flatten altrimenti si perde informazione per singola batch
        #x = torch.flatten(x)
        #x = torch.squeeze(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc3(x)
        return x

class FGN(nn.Module):
    def __init__(self, model_rgb, model_flow, model_merge):
        super(FGN, self).__init__()
        self.model_rgb = model_rgb
        self.model_flow = model_flow
        self.model_merge = model_merge
        
    def forward(self, rgb, flow):
        rgb = self.model_rgb(rgb)
        flow = self.model_flow(flow)
        x = self.model_merge(rgb, flow)
        return x

'''



# from keras which is    (clip_frames, width, height, in_channels)
# to pytorch which is    (in_channels, clip_frames, width, height)
class FGN_RGB(nn.Module):
    def __init__(self, args):
        super(FGN_RGB, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(16)
        
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(32)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm7 = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm8 = nn.BatchNorm3d(32)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1)) 
        # ADDED    
        self.dropout3d = nn.Dropout3d(p=args.dropout3d)


    def forward (self, rgb):
        ################# RGB Block
        rgb = self.conv1(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm1(rgb)
        rgb = self.dropout3d(rgb)
        rgb = self.conv2(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm2(rgb)
        rgb = self.maxpool1(rgb)
        rgb = self.dropout3d(rgb)
        
        rgb = self.conv3(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm3(rgb)
        rgb = self.dropout3d(rgb)
        rgb = self.conv4(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm4(rgb)
        rgb = self.maxpool1(rgb)
        rgb = self.dropout3d(rgb)

        
        rgb = self.conv5(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm5(rgb)
        rgb = self.conv6(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm6(rgb)
        rgb = self.maxpool1(rgb)
        
        rgb = self.conv7(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm7(rgb)
        rgb = self.conv8(rgb)
        rgb = self.relu(rgb)
        rgb = self.batchnorm8(rgb)
        rgb = self.maxpool1(rgb)
        return rgb

class FGN_FLOW(nn.Module):
    def __init__(self, args):
        super(FGN_FLOW, self).__init__()
        self.conv1f = nn.Conv3d(2, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(16)
        
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(32)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm7 = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm8 = nn.BatchNorm3d(32)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1))
        
        # ADDED    
        self.dropout3d = nn.Dropout3d(p=args.dropout3d)
        
    def forward (self, flow):
        ################## Optical flow block
        flow = self.conv1f(flow)
        flow = self.relu(flow)
        flow = self.batchnorm1(flow)
        flow = self.dropout3d(flow)
        flow = self.conv2(flow)
        flow = self.relu(flow)
        flow = self.batchnorm2(flow)
        flow = self.maxpool1(flow)
        flow = self.dropout3d(flow)

        flow = self.conv3(flow)
        flow = self.relu(flow)
        flow = self.batchnorm3(flow)
        flow = self.dropout3d(flow)
        flow = self.conv4(flow)
        flow = self.relu(flow)
        flow = self.batchnorm4(flow)
        flow = self.maxpool1(flow)
        flow = self.dropout3d(flow)

        
        flow = self.conv5(flow)
        flow = self.relu(flow)
        flow = self.batchnorm5(flow)
        flow = self.conv6(flow)
        flow = self.relu(flow)
        flow = self.batchnorm6(flow)
        flow = self.maxpool1(flow)
        
        flow = self.conv7(flow)
        flow = self.sigmoid(flow)
        flow = self.batchnorm7(flow)
        flow = self.conv8(flow)
        flow = self.sigmoid(flow)
        flow = self.batchnorm8(flow)
        flow = self.maxpool1(flow)

        
        return flow


class FGN_MERGE_CLASSIFY(nn.Module):
    def __init__(self, args):
        super(FGN_MERGE_CLASSIFY, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1))
        
        self.conv1m = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.conv2m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.conv3m = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(64)
        self.conv4m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5m = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.conv6m = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(128)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 3, 3))
        
        # Instead of this, otherwise the depth go to zero!
        #self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32,1)
        self.dropout = nn.Dropout(args.dropout)
        
    
    def forward (self, rgb, flow):
        ########### Fusion and pooling
        x = flow * rgb
        x = self.maxpool2(x)
    
        ########## Merging block
        x = self.conv1m(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2m(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool3(x)
        x = self.conv3m(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.conv4m(x)
        x = self.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool3(x)

        x = self.conv5m(x)
        x = self.relu(x)
        x = self.batchnorm5(x)
        x = self.conv6m(x)
        x = self.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool4(x)

        ########## Classifier
        # No flatten altrimenti si perde informazione per singola batch
        #x = torch.flatten(x)
        #x = torch.squeeze(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FGN(nn.Module):
    def __init__(self, model_rgb, model_flow, model_merge):
        super(FGN, self).__init__()
        self.model_rgb = model_rgb
        self.model_flow = model_flow
        self.model_merge = model_merge
        
    def forward(self, rgb, flow):
        rgb = self.model_rgb(rgb)
        flow = self.model_flow(flow)
        x = self.model_merge(rgb, flow)
        return x

    
########################################### SELF-SUPERVISED ARCHITECTURES ###################################
class Odd_One_Out(nn.Module):
    def __init__(self, args, model):
        super(Odd_One_Out, self).__init__()
        self.model = model
        self.avgpool = nn.AvgPool3d(kernel_size=(9, 7, 7))
        
        # Classifier
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, questions):
        rgb0 = self.model(questions[0])
        rgb1 = self.model(questions[1])
        rgb2 = self.model(questions[2])
        rgb3 = self.model(questions[3])
        rgb4 = self.model(questions[4])
        rgb5 = self.model(questions[5])
        
        x0 = self.avgpool(rgb0)
        x1 = self.avgpool(rgb1)
        x2 = self.avgpool(rgb2)
        x3 = self.avgpool(rgb3)
        x4 = self.avgpool(rgb4)
        x5 = self.avgpool(rgb5)
        
        x0 = torch.flatten(x0, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x3 = torch.flatten(x3, start_dim=1)
        x4 = torch.flatten(x4, start_dim=1)
        x5 = torch.flatten(x5, start_dim=1)
              
        x = (x5 - x4) + (x5 - x3) + (x5 - x2) + (x5 - x1) + (x5 - x0) 
        x +=(x4 - x3) + (x4 - x2) + (x4 - x1) + (x4 - x0)
        x +=(x3 - x2) + (x3 - x1) + (x3 - x0)
        x +=(x2 - x1) + (x2 - x0)
        x +=(x1 - x0)
        
        # Classifier 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Arrow_Of_Time(nn.Module):
    def __init__(self, model):
        super(Arrow_Of_Time, self).__init__()
        self.model = model
        self.conv1 = nn.Conv3d(64, 64, kernel_size=(8, 7, 7), stride = (1, 1, 1), padding=(1, 1, 1))
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(8, 7, 7), stride = (1, 1, 1), padding=(1, 1, 1))
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 2, kernel_size=(8, 7, 7), stride = (1, 1, 1), padding=(1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(2)
        
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        
    def forward(self, questions):
        flow0 = self.model(questions[0])
        flow1 = self.model(questions[1])
        
        x = torch.cat((flow0, flow1), dim=1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        
        return x


    
class Cubic_Puzzle(nn.Module):
    def __init__(self, model):
        super(Cubic_Puzzle, self).__init__()
        self.model = model
        self.avgpool = nn.AvgPool3d(kernel_size=(4, 2, 2))
        self.fc1 = nn.Linear(2048, 24)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, questions):
        rgb0 = self.model(questions[0])
        rgb1 = self.model(questions[1])
        rgb2 = self.model(questions[2])
        rgb3 = self.model(questions[3])
                
        x0 = self.avgpool(self.relu(rgb0))
        x1 = self.avgpool(self.relu(rgb1))
        x2 = self.avgpool(self.relu(rgb2))
        x3 = self.avgpool(self.relu(rgb3))

        x0 = torch.flatten(x0, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x3 = torch.flatten(x3, start_dim=1)
        
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.fc1(x)
        return x


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    

class VIC_Reg_Siamese(nn.Module):
    def __init__(self, args, model):
        super(VIC_Reg_Siamese, self).__init__()
        self.model = model
        self.num_features = args.expander_dimensionality
        self.batch_size = args.batch_size
        self.sim_coeff = args.lambd # lambda
        self.std_coeff = args.mu # mu
        self.cov_coeff = args.nu  # nu
        
        
        #self.avgpool = nn.AvgPool3d(kernel_size=(5, 4, 2))
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(3, 3, 7))
        self.adaptmaxpool = nn.AdaptiveMaxPool3d(output_size=(4, 4, 4))
        
        self.fc1 = nn.Linear(args.expander_input_dim, self.num_features)
        
        self.fc2 = nn.Linear(self.num_features, self.num_features)
        self.fc3 = nn.Linear(self.num_features, self.num_features, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm1d(args.expander_input_dim)
        self.batchnorm2 = nn.BatchNorm1d(self.num_features)
        self.batchnorm3 = nn.BatchNorm1d(self.num_features)
    

    def forward(self, questions):
        # BACKBONE: compute representations
        y_a = self.model(questions[0])
        y_b = self.model(questions[1])
        # y_a.shape =  [8, 32, 16, 14, 14]
        
        
        y_a = self.adaptmaxpool(self.relu(y_a))
        y_b = self.adaptmaxpool(self.relu(y_b))
        # y_a.shape = [16, 32, 4, 4, 4]
        y_a = torch.flatten(y_a, start_dim=1)
        y_b = torch.flatten(y_b, start_dim=1)
        # y_a.shape = [8, 2048]
              
        # PROJECTOR
        z_a = self.fc1(self.relu(self.batchnorm1(y_a)))
        z_a = self.fc2(self.relu(self.batchnorm2(z_a)))
        z_a = self.fc3(z_a)
        
        z_b = self.fc1(self.relu(self.batchnorm1(y_b)))
        z_b = self.fc2(self.relu(self.batchnorm2(z_b)))
        z_b = self.fc3(z_b)
        
        # LOSS COMPUTATION
        x = z_a
        y = z_b
        
        # Invariance loss
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        # Variance loss
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
        
        # Covariance loss
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # Total loss
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss 
    
    
class MergingBlock(nn.Module):
    def __init__(self):
        super(MergingBlock, self).__init__()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(8, 1, 1))
        
        self.conv1m = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.conv2m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.conv3m = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm3 = nn.BatchNorm3d(64)
        self.conv4m = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5m = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.conv6m = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(1,1,1), padding='same', bias=True)
        self.batchnorm6 = nn.BatchNorm3d(128)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 3, 3))
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        #x = self.maxpool2(x) # ADDED wrt first experiments (0-1)
        x = self.conv1m(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2m(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool3(x)
        x = self.conv3m(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.conv4m(x)
        x = self.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool3(x)

        x = self.conv5m(x)
        x = self.relu(x)
        x = self.batchnorm5(x)
        x = self.conv6m(x)
        x = self.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool4(x)
        
        return x    
    
    
class VIC_Reg(nn.Module):
    def __init__(self, args, model_rgb, model_flow, merging_block):
        super(VIC_Reg, self).__init__()
        self.model_rgb = model_rgb
        self.model_flow = model_flow
        self.merging_block = merging_block
        self.num_features = args.expander_dimensionality
        self.batch_size = args.batch_size
        self.sim_coeff = args.lambd # lambda
        self.std_coeff = args.mu # mu
        self.cov_coeff = args.nu  #nu
        
        
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(3, 3, 7))
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(3, 3, 7))
        self.adaptmaxpool = nn.AdaptiveMaxPool3d(output_size=(3, 3, 3))
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(2, 2, 1))
        
        self.fc1 = nn.Linear(args.expander_input_dim, self.num_features)

        self.fc2 = nn.Linear(self.num_features, self.num_features)
        self.fc3 = nn.Linear(self.num_features, self.num_features, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm1d(args.expander_input_dim)
        self.batchnorm2 = nn.BatchNorm1d(self.num_features)
        self.batchnorm3 = nn.BatchNorm1d(self.num_features)
        

    
    def forward(self, questions):
        # BACKBONE: compute representations
        y_a = self.model_rgb(questions[0])
        y_b = self.model_flow(questions[1])
        # y_a.shape =  [8, 32, 16, 14, 14]
        
        ########## Merging block
        '''
        y_a = self.merging_block(y_a)
        y_b = self.merging_block(y_b)
        
        y_a = self.avgpool(self.relu(y_a))
        y_b = self.avgpool(self.relu(y_b))
        '''
        y_a = self.adaptmaxpool(self.relu(y_a))
        y_b = self.adaptmaxpool(self.relu(y_b))
        
        y_a = torch.flatten(y_a, start_dim=1)
        y_b = torch.flatten(y_b, start_dim=1)
        # y_a.shape = [8, 1024]
        
        # PROJECTOR
        z_a = self.fc1(self.relu(self.batchnorm1(y_a)))
        z_a = self.fc2(self.relu(self.batchnorm2(z_a)))
        z_a = self.fc3(z_a)
        
        z_b = self.fc1(self.relu(self.batchnorm1(y_b)))
        z_b = self.fc2(self.relu(self.batchnorm2(z_b)))
        z_b = self.fc3(z_b)
        
        # LOSS COMPUTATION
        x = z_a
        y = z_b
        
        # Invariance loss
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        # Variance loss
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
        
        # Covariance loss
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # Total loss
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss, repr_loss, std_loss, cov_loss    
     

