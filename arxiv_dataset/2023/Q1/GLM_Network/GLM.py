import torch
import numpy as np

class GL_from_NN():
    
    def __init__(self,num_lay,num_ner,act):
        self.num_lay = num_lay
        self.num_ner = num_ner
        self.activation = act
        assert num_lay==len(num_ner),"Number of neurons in the list must match with number of layers"

        #create model
        self.model = self.create_NN_model()

    def create_NN_model(self):
        model = torch.nn.Sequential()

        #The first layer and its activation
        model.add_module("lay_1",torch.nn.Linear(1,self.num_ner[0]))
        model.add_module("act_1",self.activation)

        #Add middle layers to the model
        for i in range (len(self.num_ner)-1):
            model.add_module(f"lay_{i+2}",torch.nn.Linear(self.num_ner[i],self.num_ner[i+1]))
            model.add_module(f"act_{i+2}",self.activation)
        
        #last layer without bias term
        model.add_module("last_lay",torch.nn.Linear(self.num_ner[-1],1,bias=False))
        model.add_module("Flat",torch.nn.Flatten(0, 1))

        return model

    def xi2_loss(self,y_pre,y_obs,dy_obs):
        return ((y_pre - y_obs)/dy_obs).pow(2).sum()    


    def fit(self,data,num_epocks,learning_rate):
        #preparing the data
        self.x_obs = data[:,0]
        self.y_obs = data[:,1]
        self.err = data[:,2]

        #convert to tensor and reshape
        x_obs = torch.from_numpy(data[:,0]).float()
        x_obs = torch.reshape(x_obs,(len(data[:,0]),1))
        y_obs = torch.from_numpy(data[:,1]).float()
        sig = torch.from_numpy(data[:,2]).float()

        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

        for t in range(num_epocks +1):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(x_obs)

            # Compute and print loss.
            loss = self.xi2_loss(y_pred,y_obs,sig)

            if t % 1000==0 and t!=0:
                print(f"Epoch = {t}",f"$\chi^2$ = {round(loss.item(),2)}")

            optimizer.zero_grad()

            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        print("End")    


    def sep_fun(self,x):
        #convert to tensor and reshape
        x = torch.from_numpy(x).float()
        x = torch.reshape(x,(len(x),1))

        #Out put of the first layer
        w = self.model.state_dict()[f"lay_{1}.weight"]
        b = self.model.state_dict()[f"lay_{1}.bias"]
        z = self.activation(x@w.T + b)

        #output of the layer before the last one
        for i in range(self.num_lay-1):
            w = self.model.state_dict()[f"lay_{i+2}.weight"]
            b = self.model.state_dict()[f"lay_{i+2}.bias"]
            z = self.activation(z@w.T + b)

        return z.detach().numpy()

    def predict(self,x):
        x = torch.from_numpy(x).float()
        x = torch.reshape(x,(len(x),1))
        y = self.model(x) 
        return y.detach().numpy()
        

    def last_layer_weight(self):
        w = torch.flatten(self.model.state_dict()["last_lay.weight"],0)
        return w.detach().numpy()


    def ini_Bayesian(self,mean_prior,cov_prior):
        self.mea_pri = mean_prior
        self.cov_pri = cov_prior
        self.n_obs = len(self.x_obs)
        self.n_par = self.num_ner[-1]

        self.b = self.y_obs/self.err   

        F = np.zeros(shape=(self.n_obs,self.n_par))
        A = np.zeros(shape=(self.n_obs,self.n_par))

        for j in range(self.n_par):
            F[:,j] = self.sep_fun(self.x_obs)[:,j]
            A[:,j] = F[:,j]/self.err

        self.A = A
        self.L = np.dot(A.T,A)
        L_inv = np.linalg.inv(self.L)
        self.theta_0 = L_inv@self.A.T@self.b     


    def max_likelihood_estimator(self):

        return self.theta_0


    def post_dist(self):

        cov_pos = self.L + self.cov_pri
        cov_pos_inv = np.linalg.inv(cov_pos)
        t = self.L@self.theta_0 + self.cov_pri@self.mea_pri
        mea_pos = cov_pos_inv@t
    
        return mea_pos,cov_pos_inv 

    def theta_err(self):
        cov_pos = self.L + self.cov_pri
        cov_pos_inv = np.linalg.inv(cov_pos)
        theta_err = [np.sqrt(cov_pos_inv[i,i]) for i in range(self.num_ner[-1])]
        return theta_err
    
    def best_fun_err(self,x):
        funcs = self.sep_fun(x)
        t_err = self.theta_err()
        t2_err = [(t_err[i]*funcs[:,i])**2 for i in range(self.num_ner[-1])]
        t2_err_sum = np.sum(t2_err,axis=0)
        return np.sqrt(t2_err_sum)


    def evidenc(self):

        pow_l0 = -0.5*np.dot((self.b-np.dot(self.A,self.theta_0)).T,self.b-np.dot(self.A,self.theta_0))
        t1 = pow(2*np.pi,self.n_obs/2)*np.prod(self.err)

        cov_pos = self.L + self.cov_pri
        cov_pos_inv = np.linalg.inv(cov_pos)
        t2 = pow(np.linalg.det(cov_pos)/np.linalg.det(self.cov_pri),-0.5)
        D1 = 0.5*(  self.theta_0.T@self.L@self.theta_0   +  self.mea_pri.T@self.cov_pri@self.mea_pri)
        D2 = 0.5*( (self.theta_0.T@self.L + self.mea_pri.T@self.cov_pri)@cov_pos_inv @ 
        (self.L@self.theta_0 + self.cov_pri@self.mea_pri)     ) 

        return np.log(t2) - np.log(t1) + pow_l0 + D2 - D1         
           
