import torch 
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.hidden = torch.nn.Linear(input_length, 128)   # hidden layer
        self.hidden1 = torch.nn.Linear(1024, 512)   # hidden layer
        self.hidden2 = torch.nn.Linear(512, 256)   # hidden layer
        self.hidden3 = torch.nn.Linear(256, 128)   # hidden layer
        self.hidden4 = torch.nn.Linear(128, 64)   # hidden layer
        self.predict = torch.nn.Linear(64, 128)   # output layer        
        self.dropout = nn.Dropout(0.75)
        self.activation = nn.Sigmoid()
        

    def forward(self, x):
        x = self.hidden(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        x = self.hidden4(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        x = self.predict(x)             
        x = self.dropout(x)

        return x
    

class Discriminator_Agnostic(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Agnostic, self).__init__()
        self.problem = problem
        self.hidden = torch.nn.Linear(input_length, 128)   # hidden layer
        self.hidden1 = torch.nn.Linear(1024, 512)   # hidden layer
        self.hidden2 = torch.nn.Linear(512, 256)   # hidden layer
        self.hidden3 = torch.nn.Linear(256, 128)   # hidden layer
        self.hidden4 = torch.nn.Linear(128, 64)   # hidden layer
        self.predict = torch.nn.Linear(64, 1)   # output layer        
        self.dropout = nn.Dropout(0.75)
        self.activation = nn.Sigmoid()
        

    def forward(self, x):
        x = self.hidden(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        # x = self.hidden1(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      

        # x = self.hidden2(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      

        # x = self.hidden3(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      


        x = self.hidden4(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        x = self.predict(x)        
        x = self.dropout(x)

        return x
    

class Discriminator_Awareness(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Awareness, self).__init__()
        self.problem = problem
        self.hidden = torch.nn.Linear(input_length, 128)   # hidden layer
        self.hidden1 = torch.nn.Linear(1024, 512)   # hidden layer
        self.hidden2 = torch.nn.Linear(512, 256)   # hidden layer
        self.hidden3 = torch.nn.Linear(256, 128)   # hidden layer
        self.hidden4 = torch.nn.Linear(128, 64)   # hidden layer
        self.predict = torch.nn.Linear(64, 1)   # output layer        
        self.dropout = nn.Dropout(0.75)
        self.activation = nn.Sigmoid()
        

    def forward(self, Z, S):
        x = torch.cat((Z,S),1)    
        
        x = self.hidden(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        # x = self.hidden1(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      

        # x = self.hidden2(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      

        # x = self.hidden3(x)  
        # x = self.dropout(x)
        # x = F.relu(x)      


        x = self.hidden4(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        x = self.predict(x)             
        return x
    