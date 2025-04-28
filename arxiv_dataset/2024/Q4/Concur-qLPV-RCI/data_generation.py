from imports import *
from utils.generate_file_path import generate_file_path
from utils.multisine_inputs import generate_multisine

current_directory = Path(__file__).parent

def generate_dataset(system, N_train, N_test, scale_data, folder_name, file_name, overwrite_data):
    #Data generation
    file_path = generate_file_path(folder_name, file_name, current_directory)
    if not file_path.exists() or overwrite_data:
        print('Generating dataset...')
        nu = system.nu_plant

        #Training data
        #U_tot = generate_multisine(nu, N_train, 0.1, 100, -system.u_lim, system.u_lim, 10)
        U_tot = -system.u_lim + 2.*np.random.rand(N_train, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_train = U_tot[:N_train,:]
        Y_train = Y_tot[:N_train,:]

        #Testing data
        system.state = np.zeros(system.nx_plant)
        U_tot = -system.u_lim + 2.*np.random.rand(N_test, system.nu_plant) * np.array(system.u_lim)
        Y_tot = system.simulate(U_tot)
        Y_tot = Y_tot[:-1,:]
        U_test = U_tot[:N_test,:]
        Y_test = Y_tot[:N_test,:]
        ny = Y_train.shape[1]

        #Scale data
        if scale_data == True:
            Ys_train, ymean, ygain = standard_scale(Y_train)
            Us_train, umean, ugain = standard_scale(U_train)
        else:
            Ys_train = Y_train
            ymean = np.zeros((ny))
            ygain = np.ones((ny))
            Us_train = U_train
            umean = np.zeros((nu))
            ugain = np.ones((nu))
        Us_test = (U_test-umean)*ugain
        Ys_test = (Y_test-ymean)*ygain

        plt.plot(Ys_train[:,0])
        plt.show()

        #Build constraints as box of inputs and convex hull of outputs
        u_lim_model = (system.u_lim-umean)*ugain
        HU = np.vstack((np.eye(nu),-np.eye(nu)))
        hU = np.hstack((u_lim_model,u_lim_model))
        Y_hull = Polytope(V = Ys_train)
        HY = Y_hull.A
        hY = Y_hull.b
        constraints = {'HU': HU, 'hU': hU, 'HY': HY, 'hY': hY}

        #Build dataset
        dataset = {}
        dataset['system'] = system
        dataset['U_train'] = U_train
        dataset['Y_train'] = Y_train
        dataset['U_test'] = U_test
        dataset['Y_test'] = Y_test
        dataset['u_scaler'] = np.array([umean, ugain])
        dataset['y_scaler'] = np.array([ymean, ygain])
        dataset['Us_train'] = Us_train
        dataset['Ys_train'] = Ys_train
        dataset['Us_test'] = Us_test
        dataset['Ys_test'] = Ys_test
        dataset['constraints'] = constraints

        #Save dataset
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('Dataset saved to ', file_path)
    else:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

    return dataset
    
