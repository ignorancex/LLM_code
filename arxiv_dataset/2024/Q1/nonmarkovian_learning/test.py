import numpy as np
from data_gen import generate_forward,gen_rand_lambda
import argparse 
import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description="Problem specs")
parser.add_argument('--k_value',default=1.,type=float,help='k parameter of the gamma distribution')
parser.add_argument('--w_value',default=15,type=int,help='window size')
parser.add_argument('--model',default='AR',type=str,help='name of the model (AR, CNN, GRU) to test.')
parser.add_argument('--buffer',default=20,type=int,help='number of buffer steps for testing')
parser.add_argument('--num_samps',default=10000,type=int,help='number of testing samples')
parser.add_argument('--sub_rate',default=30,type=int,help='the subsampling rate')
parser.add_argument('--run_id',default=0,type=int,help='stamp id')
specs = parser.parse_args()

if __name__=='__main__':
    k_value = specs.k_value
    sub_rate = specs.sub_rate
    buffer = specs.buffer
    id_var = specs.run_id
    num_samps = specs.num_samps
    w_value = specs.w_value
    model_name = specs.model
    
    print(f'Testing the {model_name} model with k={k_value}, w={w_value}')
    
    t_list = np.linspace(0,30,3000) 
    lambdas_test = gen_rand_lambda(t_list,num_samps,k_gamma=k_value)    
    trajs_test = generate_forward(lambdas_test,t_list)
    
    
    x_test = trajs_test[:,::sub_rate,None] 
    y_test= (lambdas_test[:,::sub_rate]+1)/2

    n_steps = x_test.shape[1]
    
    
    tf.keras.backend.clear_session()
    
    student_model = load_model(f'models/{model_name}_w_{w_value}_k_{k_value}_id_{id_var}')
    y_pred = student_model.predict(x_test)

    if model_name=='RNN':
        mean_errors = np.mean(np.power(y_test[:,buffer:]-(y_pred[:,buffer:]>0.5),2)) 
    else:
        mean_errors = np.mean(np.power(y_test[:,buffer:]-(y_pred[:,(buffer-w_value+1):]>0.5),2)) #fixing the issue with different lengths


    np.save(f'results/{model_name}_w_{w_value}_k_{k_value}_id_{id_var}.npy',mean_errors)
