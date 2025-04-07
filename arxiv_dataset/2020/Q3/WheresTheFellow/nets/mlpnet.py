import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from math import atan2, acos, sqrt, sin, cos
from os.path import exists
import pickle
import json

class MLPNet(nn.Module):
    def __init__(self, train_2D):
        super(MLPNet, self).__init__()
        
        if train_2D:
            input_size, Hconv, Hfc, output_size = 306, [4*128]*6, [4*128]*6, 5
        else:
            input_size, Hconv, Hfc, output_size = 612, [4*128]*6, [4*128]*6, 5
        
        self.input_size = input_size
        self.conv = nn.ModuleList()
        self.fc = nn.ModuleList()
        if len(Hconv)>1:
          self.conv.append(nn.Conv1d(1,Hconv[0],int(input_size/3),stride=int(input_size/3)))
          h0 = Hconv[0]
          for h in Hconv[1:]:
            self.conv.append(nn.Conv1d(h0,h,1,stride=1))
            h0=h
          self.fc.append(nn.Linear(3*h0, Hfc[0]))          
        h0 = Hfc[0]
        for h in Hfc[1:]:
          self.fc.append(nn.Linear(h0, h))
          h0 = h
        self.fc.append(nn.Linear(h0, output_size))
        
        self.train_2D = train_2D
        
        self.load_or_train()

    def forward(self, x):
        x = x[:,None,:]
        for conv in self.conv:
          x = F.relu(conv(x))
        x = x.view(x.shape[0],-1)
        #x = F.dropout(x,p=0.5)
        for fc in self.fc[:-1]:
          x = F.relu(fc(x))
        x = self.fc[-1](x)
        return x

    @staticmethod
    def load_data(filename):
        json_file = open(filename)
        data = json.load(json_file)['data_set']

        joint_labels = ['left_ankle','left_ear', 'left_elbow', 'left_eye', 'left_hip', 'left_knee', 'left_shoulder', 'left_wrist', 'nose', 'right_ankle', 'right_ear', 'right_elbow', 'right_eye', 'right_hip', 'right_knee', 'right_shoulder', 'right_wrist']
        cameras=[1,2,3]

        data_rows=[]
        ground_truth_rows=[]
        for x in data:
            cam_joints = dict()
            cam_joints_exists = dict()
            cam_timestamp=dict()
            cam_ground_truth = dict()

            for y in x['superbody']:
                cameraId = y['cameraId']
                cam_ground_truth[cameraId] = y['ground_truth']
                cam_joints[cameraId] = []
                cam_joints_exists[cameraId] = []
                for jl in joint_labels:
                    if jl in y['joints']:
                        z = y['joints'][jl]
                        cam_joints[cameraId] += [z[0]/4000, z[1]/4000, z[2]/4000, (z[3]-320.)/320, (240.-z[4])/240, z[5]]
                        cam_joints_exists[cameraId] += [1] * 6
                    else:
                        cam_joints[cameraId] += [0] * 6
                        cam_joints_exists[cameraId] += [0] * 6
                cam_timestamp[cameraId] = y['timestamp']
                
            last_camera = sorted(cam_timestamp, key=cam_timestamp.get,reverse=True)[0]
            ground_truth = cam_ground_truth[last_camera]
            z = ground_truth
            ground_truth=[z[0]/4000,z[1]/4000,z[2]/4000,.7*sin(z[3]),.7*cos(z[3])]
            row=[]
            for c in cameras:
                if c in cam_joints:
                    row += cam_joints[c] + cam_joints_exists[c]
                else:
                    row += [0]*len(joint_labels)*6*2
            data_rows.append(row)
            ground_truth_rows.append(ground_truth)  
            
        datastruct = {'data':data_rows, 'groundtruth':ground_truth_rows}
        return datastruct

    @staticmethod
    def load_data_2D(filename):
        json_file = open(filename)
        data = json.load(json_file)['data_set']

        joint_labels = ['left_ankle','left_ear', 'left_elbow', 'left_eye', 'left_hip', 'left_knee', 'left_shoulder', 'left_wrist', 'nose', 'right_ankle', 'right_ear', 'right_elbow', 'right_eye', 'right_hip', 'right_knee', 'right_shoulder', 'right_wrist']
        cameras=[1,2,3]

        data_rows=[]
        ground_truth_rows=[]
        for x in data:
            cam_joints = dict()
            cam_joints_exists = dict()
            cam_timestamp=dict()
            cam_ground_truth = dict()

            for y in x['superbody']:
                cameraId = y['cameraId']
                cam_ground_truth[cameraId] = y['ground_truth']
                cam_joints[cameraId] = []
                cam_joints_exists[cameraId] = []
                for jl in joint_labels:
                    if jl in y['joints']:
                        z = y['joints'][jl]
                        cam_joints[cameraId] += [(z[3]-320.)/320, (240.-z[4])/240, z[5]]
                        cam_joints_exists[cameraId] += [1] * 3
                    else:
                        cam_joints[cameraId] += [0] * 3
                        cam_joints_exists[cameraId] += [0] * 3
                cam_timestamp[cameraId] = y['timestamp']
                
            last_camera = sorted(cam_timestamp, key=cam_timestamp.get,reverse=True)[0]
            ground_truth = cam_ground_truth[last_camera]
            z = ground_truth
            ground_truth=[z[0]/4000,z[1]/4000,z[2]/4000,.7*sin(z[3]),.7*cos(z[3])]
            row=[]
            for c in cameras:
                if c in cam_joints:
                    row += cam_joints[c] + cam_joints_exists[c]
                else:
                    row += [0]*len(joint_labels)*3*2
            data_rows.append(row)
            ground_truth_rows.append(ground_truth)  
            
        datastruct = {'data':data_rows, 'groundtruth':ground_truth_rows}
        return datastruct

    def train_f(self, dataloader, optimizer, loss_fun):
        self.train()
        for x, t in dataloader:
            x = x.cuda()
            t = t.cuda()
            optimizer.zero_grad()
            L = loss_fun(self(x), t)
            L.backward()
            optimizer.step()

    def test(self, dataloader, loss_fun):
        self.eval()
        total_L = 0
        with torch.no_grad():
            for x, t in dataloader:
                x = x.cuda()
                t = t.cuda()
                out = self(x)
                total_L += loss_fun(out, t)
            total_L /= len(dataloader.dataset)
            print('Test set: Avg. loss: {:.4f}'.format(total_L))
            return total_L

    def test_detailed(self, dataloader):
        self.eval()
        errors={'X':0,'Y':0,'Z':0,'angle':0}
        with torch.no_grad():
            for x, t in dataloader:
                x = x.cuda()
                t = t.cuda()
                out = self(x)
                dx = torch.sum(torch.abs(4000*(out[:,0:3]-t[:,0:3])),axis=0)
                a1 = torch.atan2(out[:,3],out[:,4])
                a2 = torch.atan2(t[:,3],t[:,4])
                da = a1-a2
                da = 180*torch.atan2(torch.sin(da), torch.cos(da))/3.141592653589793
                da = torch.sum(torch.abs(da),axis=0)            
                
                for i,s in enumerate(['X','Y','Z']):
                    errors[s] += dx[i]
                errors['angle'] += da
            
        for s, v in errors.items():
            errors[s] = v/len(dataloader.dataset)
            print(f'Error in {s}: {errors[s]}')
        return errors
        # print('\nTest set: Avg. loss: {:.4f}\n'.format(total_L))

    def model_file_name(self):
        if self.train_2D:
            return 'net_2D.dat'
        else:
            return 'net_3D.dat'

    def load_or_train(self):
        self.cuda()
        if exists(self.model_file_name()):
            print('loading existing model...')
            self.load_state_dict(torch.load(self.model_file_name()))
            return
        training_data = MLPNet.load_data('./datasets/training_DS1.json')
        test_data = MLPNet.load_data('./datasets/test.json')
        if self.train_2D:
            pickle.dump( training_data, open( 'training_data_2d.p', 'wb' ) )
            pickle.dump( test_data, open('test_data_2d.p', 'wb' ) )
        else:
            pickle.dump( training_data, open( 'training_data.p', 'wb' ) )
            pickle.dump( test_data, open('test_data.p', 'wb' ) )
        
        if self.train_2D:
            training_data = pickle.load( open('training_data_2d.p', 'rb' ) )
            test_data = pickle.load( open('test_data_2d.p', 'rb' ) )
        else:
            training_data = pickle.load( open('training_data.p', 'rb' ) )
            test_data = pickle.load( open('test_data.p', 'rb' ) )

        X = torch.tensor(training_data['data'])
        Y = torch.tensor(training_data['groundtruth'])
        train_dataset= torch.utils.data.TensorDataset(X,Y)

        X = torch.tensor(test_data['data'])
        Y = torch.tensor(test_data['groundtruth'])
        test_dataset= torch.utils.data.TensorDataset(X,Y)

        train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True)
        test_loader    = DataLoader(test_dataset,batch_size=1000, shuffle=False)


        # optim_SGD = optim.RMSprop(mlpnet.parameters(), lr=0.0001)
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        error_plots={'X':[],'Y':[],'Z':[],'angle':[],'mse':[]}

        lr_schedule = [0.000001, 0.00001]
        NUM_EPOCHS=500
        for e in range(NUM_EPOCHS):
            print(f'Epoch: {e+1}/{NUM_EPOCHS}. Training ...')
            self.train_f(train_loader, optimizer, nn.MSELoss())
            print('Testing ...')
            L = self.test(test_loader, nn.MSELoss(reduction='sum'))
            error_plots['mse'].append(L)
            error = self.test_detailed(test_loader)
            for s,v in error.items():
                error_plots[s].append(v)
            if e in [50,300,499]:
                torch.save(self.state_dict(), f'net{e:03d}.dat')
                if len(lr_schedule)>0:
                    optimizer = optim.Adam(self.parameters(), lr=lr_schedule.pop())

        torch.save(self.state_dict(), self.model_file_name())

        plt.plot(range(len(error_plots['mse'])),error_plots['mse'],'-',label='angle')
        plt.legend()
        plt.show()



    def run_tests(self):
        test_datasets = [ './datasets/test.json']
        for test_path in test_datasets:
            self.run_test(test_path)

    def run_test(self, test_path):
        if self.train_2D:
            test_data = MLPNet.load_data_2D(test_path)
        else:
            test_data = MLPNet.load_data(test_path)
        X = torch.tensor(test_data['data'])
        Y = torch.tensor(test_data['groundtruth'])
        test_dataset = torch.utils.data.TensorDataset(X,Y)
        test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        print(test_path)
        L=self.test(test_loader, nn.MSELoss(reduction='sum'))
        print(f'mse={L}')
        error = self.test_detailed(test_loader)
        print()

    def predict(self, dataloader):
        self.eval()
        with torch.no_grad():
            for x, t in dataloader:
                x = x.cuda()
                t = t.cuda()
                out = self(x)
                try:
                    result = torch.cat((result, out), 0)
                except UnboundLocalError:
                    result = out
        return result
            
if __name__ == '__main__':
    train_2D = False
    mlpnet = MLPNet(train_2D)
    mlpnet.run_tests()
