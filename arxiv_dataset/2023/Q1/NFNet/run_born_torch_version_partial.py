# Author: Bill Pengyuan Zhai. Harvard University. Yelin Group. Oct 2022

from Utils_torch_version import Network, get_nn_pairs, binary_basis, unpacknbits,\
    mix_rbf_kernel, kernel_expect, make_bas, barstripe_pdf, thin_barstripe_pdf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

geometry = (2,2) # number of measured qubits, the total amount of fermions is equal to this amount
n_fermions = np.prod(geometry)
N = 2 * np.prod(geometry) # Number of fermion modes in the total system, which is twice the amount of shown qubits
basis_m_n = binary_basis(geometry=(n_fermions,)) # this is the data basis, half of the chain

# Conn_list is the connection scheme on the whole system
conn_list = [p for p in get_nn_pairs(geometry=(N,)) if p != (N-1, 0) and p != (0, N-1)]*10 # zero indexed and should not be periodic (not a closed circle)
conn_list = [[i, N-1] for i in range(N-1)]*10
print('conn_list', conn_list)
L = len(conn_list) # Number of layers

# Generate the real image patterns which are the dataset samples essentially:
# bar_samples, stripe_samples = make_bas_thin(geometry)
# Flatten the image from 2d to 1 d
bas_samples = torch.tensor(make_bas(geometry)) # shape has (batch, 2, 2)
bas_samples = bas_samples.reshape(-1, n_fermions) # shape is (batch, 4)

#bas_samples = torch.tensor(np.concatenate((bar_samples, stripe_samples), axis=0)).reshape(-1, N)
print('bas_samples', bas_samples)
p_data = torch.tensor(1/(bas_samples.shape[0])) # a scalar value
pdf_data = torch.tensor(thin_barstripe_pdf(geometry, n_fermions)) # a torch tensor  
pdf_data[0] = .5
pdf_data[-1] = .5


# Default starting state
x_input = torch.tensor([[1,0]*n_fermions])
# x_input = torch.tensor([])
#x_input = torch.tensor([[1,0,1] * n_fermions])
# x_input = torch.tensor([[0, 0, 1, 1]])

#x_input = torch.zeros((1, N))

#indices = torch.tensor(np.random.choice(np.arange(N), size=n_fermions, replace=False))
#x_input[0, indices] = 1
print('x_input', x_input)
# Generate the BAS dataset, where we keep track of the probability distribution (spiky distribution)
# baspdf = thin_barstrips_pdf(geometry=geometry, n_fermions=n_fermions)
# print('baspdf', baspdf)
# for i in range(len(baspdf)):
#     print('basis, prob', (basis_m_n[i], baspdf[i]))


# Initialize the circuit
circuit = Network(conn_list, N)
print('circuit.parameters()', circuit.parameters())


beta1 = 0.9
lr_G = 1e-3
optimizerG = torch.optim.Adam(circuit.parameters(), lr=lr_G, betas=(beta1, 0.999)) #The parameters are the th
def kl(p_data, probs_batch):
    return torch.sum(p_data * torch.log(p_data/(probs_batch+1e-12)))
def tv_sqr(p_data, probs_batch):
    return torch.sum(torch.square(probs_batch-p_data))


# MMD loss by tracking the full probability space [0.5, 0.1, 0.2, 0.25,4,10]
K = torch.tensor(mix_rbf_kernel(basis_m_n, basis_m_n, sigma_list=[0.5]))
print('debug K', K)
def exact_mmd(pdf_data, pdf_model): #input are tensors
    # print('debug pdf_data', pdf_data)
    # print('debug pdf_model', pdf_model)
    # parmas_L_by_4 = parmas_L_times_4.reshape((L, 4))
    # bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
    # params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)
    p_diff = pdf_data-pdf_model # Although this puts a constant term to the loss value, it is easier to code this way
    return kernel_expect(K, p_diff, p_diff)

# For KL, only keep track of the p_theta scores only on the valid samples patterns
def run_KL():
    batchsize = 10
    num_batches = len(bas_samples)//batchsize if len(bas_samples)%batchsize==0 else len(bas_samples)//batchsize+1

    for itr in range(10000): # At each iteration, measure the kl divergence and update
        probs_theta_valid_samples = torch.zeros(bas_samples.shape[0]) # to collect the model probabilities at valid patterns
        # pdf_model = torch.zeros(basis_m_n.shape[0]) # To keep full pdf for exact calculation of the MMD loss
        

        circuit.zero_grad() # clear the parameter gradients
        obj = torch.tensor(0.0)
        # Loop through all of the valid image patterns and collect the kl sum
        for i in range(num_batches):
            y_batch = bas_samples[i*10:(i+1)*10]
            x_batch = x_input.repeat_interleave(y_batch.shape[0], axis=0)
            sub_mask_batch = (torch.tensor([ [1]*(N//2)+[0]*(N//2) ])).repeat(y_batch.shape[0], 1) # Measure the first half of the qubits

            probs_batch = circuit.forward_partial_observation(y_batch, x_batch, sub_mask_batch)
            # probs_batch = torch.conj(amps_batch)*amps_batch # this is still torch.cfloat
            # Only keep the real part, as all information goes to the real part
            probs_batch = probs_batch.real
            print('probs_batch', probs_batch)
            probs_theta_valid_samples[i*10:(i+1)*10] = probs_batch.detach()
            # make the loss function switching case more elegant later
            obj = obj + kl(p_data, probs_batch)
        # in the case of calculating the exact MMD loss, which cannot be written as a sum
        # obj = exact_mmd(pdf_data, )

        obj.backward()
        optimizerG.step()
        print('iter, obj', (itr, obj))

        if itr % 100 == 0:
            #print('predicted', probs_theta_valid_samples)
            plt.plot(p_data*np.ones(bas_samples.shape[0]), '^-')
            plt.plot(probs_theta_valid_samples, 'x-')
            plt.savefig('kl_training_figs/iter_'+str(itr)+'.png')
            plt.close()
            
            # plot the gradients
            grads_l_list = []
            for l in range(L):
                # each layer has four parameters
                grads_l = np.array([[circuit.V_l_module_list[l].bii.grad.detach().numpy()[0], circuit.V_l_module_list[l].bjj.grad.detach().numpy()[0],\
                 circuit.V_l_module_list[l].bij_real.grad.detach().numpy()[0], \
                circuit.V_l_module_list[l].bij_img.grad.detach().numpy()[0]]])

                grads_l_list.append(grads_l)

            grads_4_by_l = np.concatenate(grads_l_list, axis=0).T

            ax = sns.heatmap( grads_4_by_l , linewidth = 0.5 , cmap = 'coolwarm' )
            plt.xlabel('layer No.')
            plt.ylabel('thetas 0, 1, 2, 3')
            plt.savefig('kl_training_gradients/iter_'+str(itr)+'.png')
            plt.close()
        
def run_MMD():
    n_space = basis_m_n.shape[0] # The number of total probability patterns
    batchsize = 10
    num_batches = n_space//batchsize if n_space%batchsize==0 else n_space//batchsize+1

    for itr in range(10000): # At each iteration, measure the kl divergence and update
        #probs_theta_valid_samples = torch.zeros(bas_samples.shape[0]) # to collect the model probabilities at valid patterns
        pdf_model = torch.zeros(basis_m_n.shape[0]) # To keep full pdf for exact calculation of the MMD loss
        

        circuit.zero_grad() # clear the parameter gradients
        obj = torch.tensor(0.0)
        # Loop through all possible patterns in the total probability space
        for i in range(num_batches):
            y_batch = basis_m_n[i*10:(i+1)*10]
            x_batch = x_input.repeat_interleave(y_batch.shape[0], axis=0)
            sub_mask_batch = (torch.tensor([ [1]*(N//2)+[0]*(N//2) ])).repeat(y_batch.shape[0], 1) # Measure the first half of the qubits

            probs_batch = circuit.forward_partial_observation(y_batch, x_batch, sub_mask_batch)
            # probs_batch = torch.conj(amps_batch)*amps_batch # this is still torch.cfloat
            # Only keep the real part, as all information goes to the real part
            probs_batch = probs_batch.real

            print('debug probs_batch', probs_batch)
            #print('probs_batch', probs_batch)
            pdf_model[i*10:(i+1)*10] = probs_batch # Keep track of the gradient, as this directly goes into the loss calculation
            
        # in the case of calculating the exact MMD loss, which cannot be written as a sum
        obj = exact_mmd(pdf_data, pdf_model)

        obj.backward()
        optimizerG.step()
        print('iter, obj', (itr, obj))

        if itr % 100 == 0:
            #print('predicted', probs_theta_valid_samples)
            plt.plot(pdf_data.detach().numpy(), '^-')
            plt.plot(pdf_model.detach().numpy(), 'x-')
            plt.savefig('kl_training_figs/iter_'+str(itr)+'.png')
            plt.close()
            
            # plot the gradients
            grads_l_list = []
            for l in range(L):
                # each layer has four parameters
                grads_l = np.array([[circuit.V_l_module_list[l].bii.grad.detach().numpy()[0], circuit.V_l_module_list[l].bjj.grad.detach().numpy()[0],\
                 circuit.V_l_module_list[l].bij_real.grad.detach().numpy()[0], \
                circuit.V_l_module_list[l].bij_img.grad.detach().numpy()[0]]])

                grads_l_list.append(grads_l)

            grads_4_by_l = np.concatenate(grads_l_list, axis=0).T

            ax = sns.heatmap( grads_4_by_l , linewidth = 0.5 , cmap = 'coolwarm' )
            plt.xlabel('layer No.')
            plt.ylabel('thetas 0, 1, 2, 3')
            plt.savefig('kl_training_gradients/iter_'+str(itr)+'.png')
            plt.close()
  
def run_TV2():
    n_space = basis_m_n.shape[0] # The number of total probability patterns
    batchsize = 10
    num_batches = n_space//batchsize if n_space%batchsize==0 else n_space//batchsize+1

    for itr in range(10000): # At each iteration, measure the kl divergence and update
        #probs_theta_valid_samples = torch.zeros(bas_samples.shape[0]) # to collect the model probabilities at valid patterns
        pdf_model = torch.zeros(basis_m_n.shape[0]) # To keep full pdf for exact calculation of the MMD loss
        

        circuit.zero_grad() # clear the parameter gradients
        obj = torch.tensor(0.0)
        # Loop through all possible patterns in the total probability space
        for i in range(num_batches):
            y_batch = basis_m_n[i*10:(i+1)*10]
            x_batch = x_input.repeat_interleave(y_batch.shape[0], axis=0)
            sub_mask_batch = (torch.tensor([ [1]*(N//3)+[0]*(N-N//3) ])).repeat(y_batch.shape[0], 1) # Measure the first half of the qubits

            probs_batch = circuit.forward_partial_observation(y_batch, x_batch, sub_mask_batch)
            # probs_batch = torch.conj(amps_batch)*amps_batch # this is still torch.cfloat
            # Only keep the real part, as all information goes to the real part
            probs_batch = probs_batch.real
            #print('probs_batch', probs_batch)
            pdf_model[i*10:(i+1)*10] = probs_batch # Keep track of the gradient, as this directly goes into the loss calculation
            
        # in the case of calculating the exact MMD loss, which cannot be written as a sum
        obj = torch.sum((pdf_data-pdf_model)**2)

        obj.backward()
        optimizerG.step()
        print('iter, obj', (itr, obj))

        if itr % 100 == 0:
            #print('predicted', probs_theta_valid_samples)
            plt.plot(pdf_data.detach().numpy(), '^-')
            plt.plot(pdf_model.detach().numpy(), 'x-')
            plt.savefig('kl_training_figs/iter_'+str(itr)+'.png')
            plt.close()
            
            # plot the gradients
            grads_l_list = []
            for l in range(L):
                # each layer has four parameters
                grads_l = np.array([[circuit.V_l_module_list[l].bii.grad.detach().numpy()[0], circuit.V_l_module_list[l].bjj.grad.detach().numpy()[0],\
                 circuit.V_l_module_list[l].bij_real.grad.detach().numpy()[0], \
                circuit.V_l_module_list[l].bij_img.grad.detach().numpy()[0]]])

                grads_l_list.append(grads_l)

            grads_4_by_l = np.concatenate(grads_l_list, axis=0).T

            ax = sns.heatmap( grads_4_by_l , linewidth = 0.5 , cmap = 'coolwarm' )
            plt.xlabel('layer No.')
            plt.ylabel('thetas 0, 1, 2, 3')
            plt.savefig('kl_training_gradients/iter_'+str(itr)+'.png')
            plt.close()
  

run_MMD()


# the p_theta scores only on the valid samples patterns
# probs_theta_valid_samples = torch.zeros(bas_samples.shape[0])
# for itr in range(5): # At each iteration, measure the kl divergence and update
#     circuit.zero_grad() # clear the parameter gradients
#     print('circuit grad', circuit.V_l_module_list[0].bii.grad)
#     print('circuit grad', circuit.V_l_module_list[0].bjj.grad)
#     print('circuit grad', circuit.V_l_module_list[0].bij_real.grad)
#     print('circuit grad', circuit.V_l_module_list[0].bij_img.grad)
#     tv = torch.tensor(0.0)
#     for i in range(num_batches):
#         y_batch = bas_samples[i*10:(i+1)*10]
#         x_batch = x_input.repeat_interleave(y_batch.shape[0], axis=0)
#         amps_batch = circuit(y_batch, x_batch)
#         print('amps_batch is leaf', amps_batch.is_leaf)
#         probs_batch = torch.conj(amps_batch)*amps_batch # this is still torch.cfloat
#         probs_batch = probs_batch.real
#         print('probs_batch', probs_batch)
#         print('probs_batch is leaf', probs_batch.is_leaf)
#         probs_theta_valid_samples[i*10:(i+1)*10] = probs_batch.detach()

#         tv = tv + torch.sum(probs_batch)
    

#     print('predicted', probs_theta_valid_samples)
#     plt.plot(p_data*np.ones(bas_samples.shape[0]), '^-')
#     plt.plot(probs_theta_valid_samples, 'x-')
#     plt.savefig('kl_training_figs/iter_'+str(itr)+'.png')
#     plt.close()
#     print('iter, tv', (itr, tv))

    
#     tv.backward()
#     print('circuit grad', circuit.V_l_module_list[-1].bii.grad)
#     print('circuit grad', circuit.V_l_module_list[-1].bjj.grad)
#     print('circuit grad', circuit.V_l_module_list[-1].bij_real.grad)
#     print('circuit grad', circuit.V_l_module_list[-1].bij_img.grad)
#     # Update D
#     optimizerG.step()

# # Start the training based on the KL divergence

# # Given the parameters (bii, bjj, bij_real, b_ij_imag)
# # Params is a L by 4 numpy nd array

# # a batch of real samples
# num_real_data = len(baspdf.nonzero())
# y_batch = basis_m_n[baspdf.nonzero()]
# print('y_batch', y_batch)
# def plot_compare(feval_params, itr):
#     parmas_L_by_4 = feval_params.reshape((L, 4))
#     bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
#     params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)
#     qbm.set_params(params_L_3)
#     p_model = qbm.get_full_pdf(x_input)
#     #print('full pdf', p_model)

#     plt.figure()
#     plt.plot(baspdf)
#     plt.plot(p_model)
#     plt.legend(['P_data','P_model'])
#     plt.savefig('training_figs/iter_'+str(itr)+'.png')
#     plt.close()

# def feval_kl(parmas_L_times_4): # Calculate the kl divergence
#     kl = 0
#     parmas_L_by_4 = parmas_L_times_4.reshape((L, 4))
#     for y in y_batch:
#         # update the parameters
#         bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
#         params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)
#         qbm.set_params(params_L_3)
#         # first get the probability of the model's proobability of measuring this x string
#         px = qbm.get_density(y, x_input)
#         kl = kl + (1/num_real_data) * np.log((1/num_real_data)/(px+1e-12))

#     # Calculate the pseudo total variation, only for small systems!
#     return kl

# # Try it with TV loss! Tracking the total probability space!
# def feval_tv(parmas_L_times_4): # Calculate the kl divergence
#     parmas_L_by_4 = parmas_L_times_4.reshape((L, 4))
#     bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
#     params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)

#     pdf_data = baspdf
#     qbm.set_params(params_L_3)
#     pdf_model = qbm.get_full_pdf(x_input)

#     tv = np.sum((pdf_model-pdf_data)**2)

#     # Calculate the pseudo total variation, only for small systems!
#     return tv

# # A semi TV loss, only compare non-zero p_data(x), needs keeping track of whole pdf
# def feval_tv2(parmas_L_times_4): # Calculate the kl divergence
#     parmas_L_by_4 = parmas_L_times_4.reshape((L, 4))
#     bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
#     params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)
#     qbm.set_params(params_L_3)

#     tv = 0
#     pdf_data = baspdf
#     for i in range(len(pdf_data)):
#         if pdf_data[i] != 0:
#             px_model = qbm.get_density(basis_m_n[i], x_input)
#             tv = tv + (px_model-pdf_data[i])**2
#     # Calculate the pseudo total variation, only for small systems!
#     return tv

# # MMD loss by tracking the full probability space [0.5, 0.1, 0.2, 0.25,4,10]
# K = mix_rbf_kernel(basis_m_n, basis_m_n, sigma_list=[0.0001])
# def feval_mmd(parmas_L_times_4):
#     parmas_L_by_4 = parmas_L_times_4.reshape((L, 4))
#     bij_vec = parmas_L_by_4[:,2]+parmas_L_by_4[:,3]*1j # the bij has a real part and a complex part
#     params_L_3 = np.concatenate((parmas_L_by_4[:,0:2], bij_vec[:,None]), axis=1)

#     pdf_data = baspdf
#     qbm.set_params(params_L_3)
#     pdf_model = qbm.get_full_pdf(x_input)
#     p_diff = pdf_data-pdf_model
#     return kernel_expect(K, p_diff, p_diff)


# # Initialize the parameters
# params_L_times_4 = np.random.rand(L*4)
# print('full pdf', qbm.get_full_pdf(x_input))
# print('feval', feval(params_L_times_4))

# eval_counts = []
# f_evals = []
# stds = []
# def store_intermediate_result(eval_count, parameters, mean, std):
#     eval_counts.append(eval_count)
#     f_evals.append(mean)
#     stds.append(std)
#     print('curr number of step, feval', (eval_count, mean))
#     plot_compare(parameters, eval_count)

# def store_intermediate_result_SLSQP(mean):
#     print('feval', (mean))

# from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, NELDER_MEAD, ADAM, GradientDescent
# optimizer = ADAM(tol=1e-4, maxiter=500) # Note! In this version, the gradient is approximated via numerical perturbation.
# #optimizer = GradientDescent(callback=store_intermediate_result, learning_rate=0.01, perturbation=1e-12, maxiter=100)
# #optimizer = SLSQP(callback=store_intermediate_result)



# result = optimizer.minimize(feval, params_L_times_4)
# print('after training', feval(result.x))
# plot_compare(result.x, -1)


