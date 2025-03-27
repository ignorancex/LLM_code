import numpy as np
import tensorflow as tf
import keras
import scipy

##########################################
##########################################

def Version_check():
    if np.__version__ != "1.26.0":
        print("The notebook was built using numpy version 1.26.0, while you are currently using the " + str(np.__version__) + " version. This may cause differences in the obtained results.")

    if tf.__version__ != "2.12.0":
        print("The notebook was built using tensorflow version 2.12.0, while you are currently using the " + str(tf.__version__) + " version. This may cause differences in the obtained results.")

    if keras.__version__ != "2.12.0":
        print("The notebook was built using keras version 2.12.0, while you are currently using the " + str(keras.__version__) + " version. This may cause differences in the obtained results.")

    if scipy.__version__ != "1.10.1":
        print("The notebook was built using numpy version 1.10.1, while you are currently using the " + str(scipy.__version__) + " version. This may cause differences in the obtained results.")

    if np.__version__ == "1.26.0" and tf.__version__ == "2.12.0" and keras.__version__ == "2.12.0" and scipy.__version__ == "1.10.1":
        print("All packages are using the targeted versions.")

##########################################
##########################################

def Load_data_file(path_):
    npz_file = np.load(path_, allow_pickle=True)
    part_1 = npz_file[npz_file.files[0]]
    part_2 = npz_file[npz_file.files[1]]
    return part_1, part_2

##########################################
##########################################

def Purity(rhos_array):
	return np.real(np.trace(np.linalg.matrix_power(rhos_array,2), axis1=-2, axis2=-1))

def Radius_dist(number_of_samples_):
    y = np.random.uniform(0, 1, number_of_samples_)
    x = y ** (1/3)
    return x

def Generate_mixed_rhos(data_rhos_, radial_positions_):
    mixing_weights = radial_positions_ / np.tile(Purity(data_rhos_)[None,:], (radial_positions_.shape[0],1))   #To allow generating higher-than-measured purities
    
    all_mixed_rhos = data_rhos_.mean(axis=0)
    
    stacked_data_rhos = np.tile(data_rhos_[None,:], (mixing_weights.shape[0],1,1,1))
    stacked_all_mixed_rhos = np.tile(all_mixed_rhos[None,None,:], (mixing_weights.shape[0],mixing_weights.shape[1],1,1))
    stacked_mixing_weights = np.tile(mixing_weights[:,:,None,None], (1,1,2,2))
    
    mixed_rhos = stacked_mixing_weights * stacked_data_rhos + (1-stacked_mixing_weights) * stacked_all_mixed_rhos

    nonph_positions = np.argwhere(np.linalg.eigvals(mixed_rhos).min(axis=-1) < 0)
    while nonph_positions.shape[0] > 0:
        mixed_rhos_flatten = np.reshape(mixed_rhos, (mixed_rhos.shape[0]*mixed_rhos.shape[1],2,2))
        mixing_weights_flatten = np.reshape(mixing_weights, (mixing_weights.shape[0]*mixing_weights.shape[1]))
        nonph_flatten_positions = np.squeeze(np.argwhere(np.linalg.eigvals(mixed_rhos_flatten).min(axis=-1) < 0))
        
        new_radial_positions = Radius_dist(nonph_positions.shape[0])
        current_rhos = data_rhos_[nonph_positions[:,1]]
        
        new_mixing_weights = new_radial_positions / Purity(current_rhos)
        mixing_weights_flatten[nonph_flatten_positions] = new_mixing_weights
        
        new_rhos = np.tile(new_mixing_weights[:,None,None], (1,2,2)) * current_rhos + (1-np.tile(new_mixing_weights[:,None,None], (1,2,2))) * np.tile(all_mixed_rhos[None,:], (new_mixing_weights.shape[0],1,1))
        mixed_rhos_flatten[nonph_flatten_positions] = new_rhos
        
        mixed_rhos = np.reshape(mixed_rhos_flatten, (mixed_rhos.shape[0],mixed_rhos.shape[1],2,2))
        mixing_weights = np.reshape(mixing_weights_flatten, (mixing_weights.shape[0],mixing_weights.shape[1]))
        nonph_positions = np.argwhere(np.linalg.eigvals(mixed_rhos).min(axis=-1) < 0)
    
    return np.reshape(mixed_rhos, (mixed_rhos.shape[0]*mixed_rhos.shape[1],2,2)), mixing_weights 

def Generate_mixed_counts(data_counts_, mixing_weights_):
    all_mixed_counts = data_counts_.mean(axis=0)
    
    stacked_data_counts = np.tile(data_counts_[None,:], (mixing_weights_.shape[0],1,1))
    stacked_all_mixed_counts = np.tile(all_mixed_counts[None,None,:], (mixing_weights_.shape[0],mixing_weights_.shape[1],1))
    stacked_mixing_weights = np.tile(mixing_weights_[:,:,None], (1,1,7))
    
    mixed_counts = stacked_mixing_weights * stacked_data_counts + (1-stacked_mixing_weights) * stacked_all_mixed_counts
    
    return np.reshape(mixed_counts, (mixed_counts.shape[0]*mixed_counts.shape[1], mixed_counts.shape[2]))

def Generate_mixed_dataset(input_rhos_, input_counts_, radial_positions_):
    mixed_rhos, mixing_weights = Generate_mixed_rhos(input_rhos_, radial_positions_)
    
    mixed_counts = Generate_mixed_counts(input_counts_, mixing_weights)
    
    return mixed_rhos, mixed_counts
	
##########################################
##########################################

def Flat_to_Density_tf(flat_):
    tf_flat_ = tf.cast(flat_, dtype=tf.float64)
    tau_zero = tf_flat_[:,0] * 0   #Weird way of getting zero-tensor with correct shape
    
    tau_real = [[tf_flat_[:,0], tau_zero], 
                [tf_flat_[:,1], tf_flat_[:,3]]]
    tau_imag = [[tau_zero, tau_zero], 
                [tf_flat_[:,2], tau_zero]]
    tau = tf.transpose(tf.complex(tau_real, tau_imag), perm=[2,0,1])
    
    rho_unnormed = tf.linalg.matmul(tau, tf.math.conj(tf.transpose(tau, perm=[0,2,1])))
    norms = tf.reshape(tf.linalg.trace(rho_unnormed), (-1,1,1))
    rho = rho_unnormed / norms
    return rho

def fidelity_metric(y_true, y_pred):
    rho_true = Flat_to_Density_tf(y_true)
    rho_pred = Flat_to_Density_tf(y_pred)
    
    sqrt_true = tf.linalg.sqrtm(rho_true)
    in_sqrt = tf.linalg.matmul(sqrt_true, tf.linalg.matmul(rho_pred, sqrt_true))
    trace = tf.linalg.trace(tf.linalg.sqrtm(in_sqrt))
    fidelity = tf.math.real(trace**2)
    
    return fidelity

##########################################
##########################################

def Probability_norm(data_counts_):
    sum_value = data_counts_.sum(axis=-1)
    return data_counts_ / np.expand_dims(sum_value, -1)
    
##########################################
##########################################

#tau[0] is matrix [[tau_1, 0], 
#                  [tau_2 + i*tau_3, tau_4]]
#flat[0] is vector [tau_1, tau_2, tau_3, tau_4]
def Tau_to_Flat(tau_):
	return np.stack((
			np.real(tau_[:, 0, 0]), 
			np.real(tau_[:, 1, 0]), 
			np.imag(tau_[:, 1, 0]), 
			np.real(tau_[:, 1, 1])), 
			axis=-1)

def Flat_to_Density(flat_):
	tau = np.moveaxis(np.array([
				[flat_[:, 0], np.zeros(flat_.shape[0])], 
				[flat_[:, 1] + 1j*flat_[:, 2], flat_[:, 3]]
				], dtype=np.complex_), -1, 0)
	
	rho = np.zeros([tau.shape[0], 2, 2], dtype=np.complex_)
	for i in range(rho.shape[0]):
		rho[i] = tau[i] @ np.conjugate(np.transpose(tau[i]))
		rho[i] /= np.trace(rho[i])
	return rho
    
##########################################
##########################################

def Sqrt_matrix(matrix_array_):                                                                                             #(..., M, M)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_array_)                                                               #(..., M2), (..., M1, M2), note: M1=M2=M
                                                                                                                            #-> [..., i] are eigenvalues, [..., :, i] are eigenvectors
    eigenvalues = np.round(eigenvalues, 15)                                                                                 #Eliminating negative eigenvalues from numerical precision, i.e., 10**(-16)
    
    if (eigenvalues == np.abs(eigenvalues)).all():                                                                          #If: Positive eigenvalues
        sqrt_eigenvalues = np.expand_dims(np.expand_dims(np.sqrt(eigenvalues), -1), -1)
        
        eigenvectors_intermediate = np.expand_dims(np.moveaxis(eigenvectors, -1, -2), axis=-1)                              #(..., M2, M1, 1), i.e., (..., i, :, 1) are eigenvectors
        eigenmatrices = (eigenvectors_intermediate @ np.moveaxis(eigenvectors_intermediate.conjugate(), -1, -2))            #(..., M2, M1, 1) @ (..., M2, 1, M1) = (..., M2, M1, M1), i.e., (..., i, :, :) are eigenmatrices |eigenvec_i><eigenvec_i|
        
        sqrt_matrices = (sqrt_eigenvalues * eigenmatrices).sum(axis=-3)                                                     #Spectral re-composition
        
        return sqrt_matrices
        
    else:
        print("Negative eigenvalues detected. Matrix is non-Hermitian.")

def Fidelity(rhos_array_1_, rhos_array_2_):
    fidelities = np.trace(Sqrt_matrix(Sqrt_matrix(rhos_array_1_) @ rhos_array_2_ @ Sqrt_matrix(rhos_array_1_)), axis1=-2, axis2=-1) **2
    return np.real(fidelities)

##########################################
##########################################

def Stokes_to_Rho(cart_):
	return 1/2 * np.moveaxis(np.array([
					[1 + cart_[:, 0], cart_[:, 1] - 1j*cart_[:, 2]], 
					[cart_[:, 1] + 1j*cart_[:, 2], 1 - cart_[:, 0]]
					]), -1, 0)

def Rho_to_Stokes(rho_):	
	sigma1 = np.array([[1., 0.], [0., -1.]])
	sigma2 = np.array([[0., 1.], [1., 0.]])
	sigma3 = np.array([[0., -1j], [1j, 0.]])
	
	stokes = np.real(np.array([
				np.trace(np.dot(rho_,sigma1), axis1=-2, axis2=-1), 
				np.trace(np.dot(rho_,sigma2), axis1=-2, axis2=-1),
				np.trace(np.dot(rho_,sigma3), axis1=-2, axis2=-1)
				]))
	return np.moveaxis(stokes, 0, -1)

def Flatten_Rho(rho_):
	return np.stack((
			np.real(rho_[:, 0, 0]), 
			np.real(rho_[:, 1, 0]), 
			np.imag(rho_[:, 1, 0]), 
			np.real(rho_[:, 1, 1])), 
			axis=-1)

##########################################
##########################################

def Error_function(x_data_, width_, x0_, left_value_, right_value_):
    err_func = scipy.special.erf(np.sqrt(2) * (x0_ - x_data_) / width_ )
    err_func_rescaled = (left_value_ - right_value_) * (1 + err_func) / 2 + right_value_
    return err_func_rescaled

def Fit_Error_function(x_data_, y_data_, width_estimate_ = 0.008):
    left_average = np.average(y_data_[:5])
    right_average = np.average(y_data_[-5:])
    center = np.average(x_data_)
    #Estimates for the initial guess

    parameters, _ = scipy.optimize.curve_fit(Error_function, x_data_, y_data_, 
                                             p0=(width_estimate_ , center, left_average, right_average), 
                                             method='dogbox')
    return parameters

def Calculate_resolution(y_data_, pixel_size_):
    x_data = np.arange(0, y_data_.shape[0]) * pixel_size_
    x_data_finer = np.arange(0, np.max(x_data), pixel_size_/1000)
    
    parameters = Fit_Error_function(x_data, y_data_)
    err_func_fit = Error_function(x_data_finer, *parameters)
    #Fitting the data with the Error_function
    
    err_func_fit_normed = err_func_fit - err_func_fit.min()
    err_func_fit_normed = err_func_fit_normed / err_func_fit_normed.max()
    #Rescaling the fitted Error_function to a [0,1] range
    
    pos_20 = np.argmin(np.abs(err_func_fit_normed - 0.2))
    pos_80 = np.argmin(np.abs(err_func_fit_normed - 0.8))
    #Position on the x-axis whose value lies the closest to the 0.2 (or 0.8)
    
    resolution = np.abs(x_data_finer[pos_20] - x_data_finer[pos_80])
    return resolution, parameters






