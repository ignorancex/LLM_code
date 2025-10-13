import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time,os
import mmnn

## 2D function Example


# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################
def func(x):
    # x has a size batch_size * dim
    a=[  [0.3,0.2], 
            [0.2,0.3]
    ]
    
    b=[12*np.pi,8*np.pi]
    
    c=[ [4*np.pi,18*np.pi],
        [16*np.pi,10*np.pi]
    ]
    
    d=[ [14*np.pi,12*np.pi],
        [18*np.pi,10*np.pi]
    ]

    r=1
    a=np.array(a)
    b=np.array(b)*r
    c=np.array(c)*r
    d=np.array(d)*r
    
    y=np.zeros_like(x[:,0])
    for i in range(2):
        for j in range(2):
            y=y+a[i,j]*np.sin(b[i]*x[:,i]+c[i,j]*x[:,i]*x[:,j])*np.abs( 
                                        np.cos(b[j]*x[:,j]+d[i,j]*x[:,i]**2) )
    return y


num_epochs = 1000
batch_size = 1200
training_samples_gridsize = [600, 600] # uniform grid samples
num_test_samples = 66666 # random samples
  
# learning rate in epoch k is 
# lr_init*lr_gamma**floor(k/lr_step_size)
lr_init=0.001
lr_gamma=0.9
lr_step_size= 20

# Set this to False if running the code on a remote server.
# Set this to True if running the code on a local PC 
# to monitor the training process.
show_plot = True 

interval=[-1,1]
ranks = [2] + [36]*7 + [1]
widths = [1024]*8
model = mmnn.MMNN(ranks = ranks, 
                 widths = widths,
                 device = device,
                 ResNet = False,
                 init_scaling = True)

   
x1 = np.linspace(*interval, training_samples_gridsize[0])
x2 = np.linspace(*interval, training_samples_gridsize[1])
X1, X2 = np.meshgrid(x1, x2)
X = np.concatenate([np.reshape(X1,[-1,1]),
                          np.reshape(X2,[-1,1])],axis=1)
Y = func(X).reshape([-1,1])
x_train = torch.tensor(X, device=device, dtype=mydtype)
y_train = torch.tensor(Y, device=device, dtype=mydtype)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, shuffle=True)


time1=time.time()
errors_train=[]
errors_test=[]
errors_test_max=[]

optimizer = optim.Adam(model.parameters(), lr=lr_init)
scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
criterion = nn.MSELoss()
 
for epoch in range(1,1+num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
              
    if epoch % 1 == 0:
        training_error = loss.item()
        print(f"\nEpoch {epoch} / {num_epochs}" + 
              f"  ( {epoch/num_epochs*100:.2f}% )" +
              f"\nTraining error (MSE): { training_error :.2e}" + 
              f"\nTime used: { time.time() - time1 :.2f}s")
        errors_train.append(training_error)
    
        def learned_nn(x): # input and output are numpy.ndarray
            x=x.reshape([-1, 2])            
            input_data = torch.tensor(x, dtype=mydtype).to(device)
            y = model(input_data)
            y = y.cpu().detach().numpy().reshape([-1])
            return y     
        
        x = np.random.rand(num_test_samples, 2) * 2 - 1
        y_nn = learned_nn(x)
        y_true = func(x)
        
        # Calculate errors
        e = y_nn - y_true
        e_max = np.max(np.abs(e))
        e_mse = np.mean(e**2)
        errors_test.append(e_mse)
        errors_test_max.append(e_max)
        
        print("Test errors (MAX and MSE): " + 
              f"{e_max:.2e} and {e_mse:.2e}")
        
        if epoch % 1 == 0:
            # Plot the results
            gridsize=[150, 150]
            x1 = np.linspace(*interval, gridsize[0])
            x2 = np.linspace(*interval, gridsize[1])
            X1, X2 = np.meshgrid(x1, x2)
            X = np.concatenate([np.reshape(X1,[-1,1]),
                                      np.reshape(X2,[-1,1])],axis=1)
            Y_true = func(X).reshape(gridsize[::-1])
            Y_nn = learned_nn(X).reshape(gridsize[::-1])
            fig=plt.figure(figsize=(12, 4.8))
            plt.subplot(1, 2, 1)
            ax=plt.gca()
            ctf = ax.contourf(X1, X2, Y_true, 100,
                    alpha=0.8, cmap="coolwarm")
            cbar = fig.colorbar(ctf, shrink=0.99, aspect=6)
            plt.title(f'true function', fontsize=19)
            plt.subplot(1, 2, 2)
            ax=plt.gca()
            ctf = ax.contourf(X1, X2, Y_nn, 100,
                    alpha=0.8, cmap="coolwarm")
            cbar = fig.colorbar(ctf, shrink=0.99, aspect=6)
            plt.title(f'learned network (Epoch {epoch})', fontsize=19)
            plt.tight_layout()
    
            FPN="./figures/"
            if not os.path.exists(FPN):
                os.makedirs(FPN)
            plt.savefig(f"{FPN}mmnn2D_epoch{epoch}.png", dpi=50)
            if show_plot:
                plt.show()

torch.save(model.state_dict(), 'model_parameters2D.pth')

fig=plt.figure(figsize=(6,4))
n=len(errors_test) 
m=len(errors_train)
k=round(m/n)
np.savez("errors2D", 
         test=np.array(errors_test), 
         testmax=np.array(errors_test_max), 
         train = np.array(errors_train), 
         time=time.time()-time1
         )
t=np.linspace(1,n,n)   
plt.plot(t, np.log10(errors_train[::k]), label="log of training error")
plt.plot(t, np.log10(errors_test), label="test error")
plt.legend()   

