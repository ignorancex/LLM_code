"""
- Module implements simple neural network with  7 hidden layer. 
-Module make predictions for model training and test. Module  draws training and validation loss for better understanding of model behavior
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
seed_plt=42
random.seed(seed_plt)
torch.manual_seed(seed_plt)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_plt)
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else
import numpy as np
#from HoloProtRepAFPML import BinaryTrainModelsWithHyperParameterOptimization
from Function_Prediction import F_max_scoring

class Net(nn.Module):
    def __init__(self, input_size, class_number):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 8)
        self.fc8 = nn.Linear(8, 8)
        self.fc9 = nn.Linear(8, class_number)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x


def model_call(input_size, class_number):
    model = Net(input_size, class_number)
    model = model.double()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    model.to(device)
    return model


def NN(
    kf,
    protein_representation,
    model_label,
    input_size,
    representation_name,
    protein_and_representation_dictionary,
):
    
    running_loss_lst_s = []
    f_max_cv_train = []
    f_max_cv_test = []
    val_loss_lst_s = []
    protein_name = []
    f_max_cv = []
    loss_train = []
    loss = []
    loss_tr = []
    loss_test = []
    protein_name_tr = []
    model_label_pred_test_lst = []
    label_lst_test = []
    model_label_pred_lst = []
    classifier_name = "Fully_Connected_Neural_Network"
    label_lst = []
    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )

    for fold_train_index, fold_test_index in kf.split(
        protein_representation, model_label
    ):

        running_loss_lst = []
        class_number = 1
        x_df = pd.DataFrame(
            protein_representation["Vector"], index=list(fold_train_index)
        )
        x = x_df["Vector"]
        x_test_df = pd.DataFrame(
            protein_representation["Vector"], index=list(fold_test_index)
        )
        x_test = x_test_df["Vector"]
        
        model = Net(input_size, class_number)
        model = model.double()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
        model.to(device)
        x = torch.tensor(list(x)).to(device)
        x = x.double()
        y = torch.tensor(model_label[fold_train_index]).to(device)
        y = y.double()
        x_test = torch.tensor(list(x_test)).to(device)
        x_test = x_test.double()
        y_test = torch.tensor(model_label[fold_test_index]).to(device)
        y_test = y_test.double()
        val_loss_lst = []

        for epoch in range(2):#25000

            # training
            running_loss = 0.0
            
            output = model(x)
            batch_loss = criterion(output, y.unsqueeze(1))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()  # epoch loss
            epoch_loss = running_loss / len(x)
            if epoch % 20 == 0:
                print("Loss: {:.3f}".format(batch_loss.item()))
            running_loss_lst.append(epoch_loss)
            # test
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                out_probs = model(x_test)
                loss_val = criterion(out_probs, y_test.unsqueeze(1))
                val_loss += loss_val.item()
                epoch_loss = running_loss / len(x_test)
                val_loss_lst.append(epoch_loss)

        running_loss_lst_s.append(running_loss_lst)
        output[output >= 0.0] = 1  # training
        output[output < 0.0] = 0

        fmax_train = 0.0
        tmax_train = 0.0
        for k in range(1, 101):
            threshold = k / 100.0
            fscore = F_max_scoring.evaluate_annotation_f_max(
                y, output
            )
            if fmax_train < fscore:
                fmax_train = fscore
                tmax_train = threshold
        f_max_cv_train.append(fmax_train)
        val_loss_lst_s.append(val_loss_lst)
        model_label_pred_lst.append(output.detach().numpy())
        label_lst.append(model_label[fold_train_index])

        for vec in protein_representation_array[fold_train_index]:
            for protein, vector in protein_and_representation_dictionary.items():
                if str(vector) == str(list(vec)):
                    protein_name_tr.append(protein)
                    continue

        out_probs[out_probs >= 0.0] = 1
        out_probs[out_probs < 0.0] = 0

        model_label_pred_test_lst.append(out_probs.detach().numpy())
        label_lst_test.append(model_label[fold_test_index])

        for vec in protein_representation_array[fold_test_index]:
            for protein, vector in protein_and_representation_dictionary.items():
                if str(vector) == str(list(vec)):
                    protein_name.append(protein)
                    continue
        fmax = 0.0
        tmax = 0.0

        for t in range(1, 101):
            threshold = t / 100.0
            fscore = F_max_scoring.evaluate_annotation_f_max(
                y_test, out_probs
            )
            if fmax < fscore:
                fmax = fscore
                tmax = threshold
        f_max_cv_test.append(fmax)

    test_loss = [sum(x) for x in zip(*val_loss_lst_s)]
    training_loss = [sum(x) for x in zip(*running_loss_lst_s)]
    """plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(test_loss, label="val")
    plt.plot(training_loss, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()"""

    parameter = {
        "classifier_name": classifier_name,
        "criterion": [criterion],
        "optimizer": str(optimizer),
    }
    #breakpoint()
    return (
        f_max_cv_train,
        f_max_cv_test,
        model,
        model_label_pred_lst,
        label_lst,
        protein_name_tr,
        parameter,
        protein_name,
        parameter,
        model_label_pred_test_lst,
        label_lst_test,
    )
