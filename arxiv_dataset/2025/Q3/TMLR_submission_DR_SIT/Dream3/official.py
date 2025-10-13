import pandas as pd
import torch
import scipy.stats as stats
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score
import argparse


def load_data(file_path):
    """Load DREAM3 trajectory TSV and return tensor of shape (Nseries, Nsteps, Nvariables).

    Expects files like InSilicoSize100-<Task>-trajectories.tsv with blank-line-separated series.
    """
    # Load the TSV file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip the header (first line)
    lines = lines[1:]

    # Combine the lines and split the content into groups by empty lines
    content = ''.join(lines)
    groups = content.split('\n\n')

    # Initialize an empty list to store data from each group
    data = []

    # Iterate through the groups
    for group in groups:
        # Remove trailing newlines and split by lines
        lines = group.strip().split('\n')
        # Parse each line and convert to a list of lists
        parsed_lines = [list(map(float, line.split('\t')[1:])) for line in lines]
        data.append(parsed_lines)

    # Convert the list of lists to a torch tensor
    tensor = torch.tensor(data)

    # Check if the dimensions are as expected
    if tensor.shape == (46, 21, 100):
        print("Parsed tensor with shape:", tensor.shape)
    else:
        print("Error: unexpected tensor shape:", tensor.shape)
    return tensor


def create_dataset(tensor,lag):
    """Create lagged supervised dataset X,Y from multivariate time-series tensor.

    X contains flattened lag windows; Y contains current step values.
    """
    # tensor tensor with dimensions (Nseries, Nsteps, Nvariables)
    # Initialize empty lists for X and Y
    X = []
    Y = []

    # Iterate through each time series and each time step (starting from the lag value)
    for n in range(tensor.shape[0]):
        for t in range(lag, tensor.shape[1]):
            # Extract the variables from the current and previous lag time steps
            prev_lag_steps_vars = tensor[n, t - lag:t].reshape(-1)
            curr_step_vars = tensor[n, t]

            # Append the previous lag time steps variables to X and the current time step variables to Y
            X.append(prev_lag_steps_vars)
            Y.append(curr_step_vars)

    # Convert the lists X and Y to torch tensors
    X = torch.stack(X)
    Y = torch.stack(Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    return X,Y


def load_groundtruth(path):
    """Load DREAM3 gold-standard edges into a 100x100 adjacency matrix (int32)."""
    # Load the TSV file
    with open(path, 'r') as f:
        lines = f.readlines()

    # Initialize a 100x100 torch tensor with zeros
    adjacency_matrix = torch.zeros((100, 100),dtype=torch.int32)

    # Iterate through the rows of the file
    for line in lines:
        # Split the line into columns
        columns = line.strip().split('\t')

        # Extract the vertex indices and edge weight
        src_vertex = int(columns[0][1:]) - 1  # Remove the 'G' prefix and subtract 1 for zero-based indexing
        dst_vertex = int(columns[1][1:]) - 1  # Remove the 'G' prefix and subtract 1 for zero-based indexing
        weight = int(columns[2])

        # Update the adjacency matrix
        adjacency_matrix[src_vertex, dst_vertex] = weight

    return adjacency_matrix


def cross_fit(X, y, k,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form):
    """K-fold cross-fitting helper that returns per-fold predictions aligned to original order."""
    # Shuffle the data using a random permutation
    perm = common_permutation
    X = X[perm]
    y = y[perm]

    # Split the shuffled data into k folds
    folds_X = torch.split(X, len(X)//k)
    folds_y = torch.split(y, len(y)//k)

    y_preds = []
    indices = []
    for i in range(k):
        # Use the ith fold for validation, and concatenate the other k-1 folds for training
        test_X, test_y = folds_X[i], folds_y[i]
        train_X = torch.cat([f for j, f in enumerate(folds_X) if j != i])
        train_y = torch.cat([f for j, f in enumerate(folds_y) if j != i])
        
        gpr=KernelRidge(alpha=1,kernel='poly',degree=3,coef0=1).fit(train_X.numpy(), train_y.numpy().reshape(-1))
        
        y_pred=torch.from_numpy(gpr.predict(test_X.numpy())).unsqueeze(1).unsqueeze(1)
        if single_regression==False:
            y_preds.append(y_pred)
        else:
            res=[y_pred]
            mea=test_X[:,1:].mean().item()
            means=test_X.mean(dim=0)
            for i in range(1,100):
                temp_X=test_X.clone()
                for j in range(i,test_X.shape[-1],99):
                    temp_X[:,j]=0
                y_pred=torch.from_numpy(gpr.predict(temp_X.numpy())).unsqueeze(1).unsqueeze(1)
                res.append(y_pred)
            y_preds.append(torch.cat(res,dim=1))
    # Concatenate the predicted y values from each fold and return them
    return torch.cat(y_preds)[torch.argsort(perm)]


def infer_causal_parents(X,y,groundtruth,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,train_double,closed_form,indices_to_check):
    """Infer causal parents via DR and direct tests using cross-fitted residuals."""
    X=torch.cat([torch.ones((len(X),1)),X],dim=1)
    node_names=groundtruth.index
    
    def mean_and_std(z):
        print(f'inference mean: {z.mean().item():.3f} inference std: {z.std().item():.3f}')    
    def calculate_accuracy(a,b):
        a,b=a.to_numpy()[0],b.to_numpy()
        z=a-b
        res=np.abs(z).sum()/len(z)
        return 1-res
    
    def ttest(y,a0,r0,ai,ri,method):
        if method=='direct':
            z0,zi=y*r0,y*ri
        else:
            z0,zi=y*r0-a0*(r0-y),y*ri-ai*(ri-y)
            z=zi-z0
            if z.abs().sum().item()==0:
                return (torch.tensor(0),torch.tensor(0))
            return (torch.tensor(0),z.mean().abs())
        return stats.ttest_rel(z0,zi)
    
    methods=['direct','dr']
    causal_graph = {method: pd.DataFrame([{node_name: 0 for node_name in node_names}]) for method in methods}
    pValues = {method: pd.DataFrame([{node_name: 0 for node_name in node_names}]) for method in methods}    
    confidence_level=0.95
    Nfolds=5
    
    a0=cross_fit(X,y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form)
    r0=cross_fit(X,y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form) if train_double else a0
    z1,z2=a0[:,0]-y,r0[:,0]-y
    print('y std:',y.std().item())
    mean_and_std(z1)
    for i in range(1,100):
        fltr=[False if (j-i)%99 == 0 else True for j in range(X.shape[1]) ]
        if single_regression==False:
            ai=cross_fit(X[:,fltr],y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form)
            ri=cross_fit(X[:,fltr],y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form) if train_double else ai
            z1,z2=ai[:,0]-y,ri[:,0]-y
        else:
            z1,z2=a0[:,i]-y,r0[:,i]-y
        for method in methods:
            if not single_regression:
                test_res=ttest(y,a0[:,0],r0[:,0],ai[:,0],ri[:,0],method)
            else:
                test_res=ttest(y,a0[:,0],r0[:,0],a0[:,i],r0[:,i],method)
            pValues[method][node_names[i-1]]=(test_res[1]).item()
    return pValues['dr'].iloc[0, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DREAM3 InSilicoSize100 benchmark runner for DR-SIT. Run from the Dream3 directory.")
    parser.add_argument("N", type=int, choices=list(range(5, 41, 5))+[43,46], help="Number of trajectories to use for training")
    parser.add_argument("task", choices=['Ecoli1', 'Ecoli2', 'Yeast1', 'Yeast2', 'Yeast3'], help="Target dataset")
    args = vars(parser.parse_args())
    args.update({'algorithm':'DR-SIT'})

    single_regression=True
    num_epochs, learning_rate, l1_reg,l2_reg, batch_size=500,0.01,0.0005,0,16
    file_path = "InSilicoSize100/InSilicoSize100-"+args['task']+"-trajectories.tsv"
    groundtruth_path='InSilicoSize100/DREAM3GoldStandard_InSilicoSize100_'+args['task']+'.txt'
    lag=2
    data_tensor=load_data(file_path).double()
    data_tensor=data_tensor[torch.randperm(data_tensor.shape[0])][:args['N']]
    X,y=create_dataset(data_tensor,lag)
    N_data=X.shape[0]-X.shape[0]%10
    X,y=X[:N_data],y[:N_data]
    common_permutation=torch.randperm(X.shape[0])
    all_pvalues=[]
    for feat in range(100):
        print('feat is', feat)
        groundtruth=load_groundtruth(groundtruth_path)
        cols_to_keep = [col for col in range(X.shape[1]) if (col - feat) % 100 != 0]
        node_names=['G'+ str(i+1) for i in cols_to_keep[:99]]
        groundtruth=pd.Series(groundtruth[cols_to_keep[:99],feat],index=node_names)
        indices_to_check=np.nonzero(groundtruth.values == 1)[0]
        pvalues=infer_causal_parents(X[:,cols_to_keep],y[:,feat].reshape(-1,1),groundtruth,num_epochs ,learning_rate, l1_reg,l2_reg, batch_size,train_double=False,closed_form=False,indices_to_check=[i+1 for i in indices_to_check])
        all_pvalues.append(pvalues)

    groundtruth=load_groundtruth(groundtruth_path)
    predicted_pvalue,all_labels=[],[]
    for feat in range(100):
        for j in range(99):
            predicted_pvalue.append(all_pvalues[feat].iloc[j])
    for feat in range(100):
        for j in range(100):
            if feat==j:
                continue
            all_labels.append(groundtruth[j][feat].item())

    auroc =roc_auc_score(all_labels,predicted_pvalue)
    print("AUROC:", auroc)
