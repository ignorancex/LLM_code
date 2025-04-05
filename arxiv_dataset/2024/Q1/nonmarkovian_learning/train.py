import numpy as np
from data_gen import generate_forward,gen_rand_lambda
import argparse 
import tensorflow as tf
from models import gen_model_AR,gen_model_CNN,gen_model_RNN

parser = argparse.ArgumentParser(description="Problem specs")
parser.add_argument('--k_value',default=1.,type=float,help='k parameter of the gamma distribution')
parser.add_argument('--w_value',default=15,type=int,help='window size')
parser.add_argument('--f_value',default=10,type=int,help='CNN filter size')
parser.add_argument('--model',default='AR',type=str,help='name of the model (AR, CNN, RNN) to train.')
parser.add_argument('--num_epochs',default=40,type=int,help='number of epochs')
parser.add_argument('--num_samps',default=50000,type=int,help='number of training samples')
parser.add_argument('--sub_rate',default=30,type=int,help='the subsampling rate')
parser.add_argument('--run_id',default=0,type=int,help='stamp id')
specs = parser.parse_args()


if __name__=='__main__':
    
    k_value = specs.k_value
    sub_rate = specs.sub_rate
    f_value = specs.f_value # only relevant for the CNN model
    num_epochs = specs.num_epochs
    id_var = specs.run_id
    num_samps = specs.num_samps
    w_value = specs.w_value
    model_name = specs.model
    
    print(f'Training the {model_name} model with k={k_value}, w={w_value}, and {num_epochs} epochs')
    
    t_list = np.linspace(0,30,3000) 
    lambdas_train = gen_rand_lambda(t_list,num_samps,k_gamma=k_value)    
    trajs_train = generate_forward(lambdas_train,t_list)
    
    
    x_train = trajs_train[:,::sub_rate,None] 
    y_train= (lambdas_train[:,::sub_rate]+1)/2

    n_steps = x_train.shape[1]
    
    
    tf.keras.backend.clear_session()
    
    if model_name=='AR':
        student_model = gen_model_AR(w_value,n_steps)
    elif model_name=='CNN':
        student_model = gen_model_CNN(w_value,n_steps,f_value)
    elif model_name=='RNN':
        student_model = gen_model_RNN(w_value,n_steps)

    student_model.compile(optimizer='adam', loss='mse')

    if model_name=='RNN':
        student_model.fit(x_train, y_train, epochs=num_epochs, verbose=1)
    else:
        student_model.fit(x_train, y_train[:,(w_value-1):], epochs=num_epochs, verbose=1) #fixing the issue with different lengths

    student_model.save(f'models/{model_name}_w_{w_value}_k_{k_value}_id_{id_var}')