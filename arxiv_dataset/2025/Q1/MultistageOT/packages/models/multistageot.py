from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import time
import copy
import os
import sys
import scipy
from datetime import timedelta

#CHANGELOG: Updated 20250124 with updates that updated doc-comments and removed unused 
#           methods, attributes and variables.


class CellGraph:
    """ Cell graph class 
    
    Parameters:
    -----------

    None


    Attributes:
    -----------

    transition_matrix      : numpy array (2D matrix)
                        Transition probability matrix corresponding to the transition probabilities between states in
                        a Markov chain.

    log_transition_matrix  : numpy array (2D matrix)
                        Log-transformed transition probability matrix (used for maximum likelihood estimates).

    """
    def __init__(self, transition_matrix):

        self.adjacency_matrix  = np.round(transition_matrix, 1)
        self.transition_matrix = transition_matrix
        self.log_transition_matrix =  np.log(transition_matrix + 1e-32)





class OTModel(ABC):
    """ Base optimal transport model class (inherits from abstract base class ABC)
         

    """
    def __init__(self):

        # Public attributes:
        self.couplings            = None                # Strength of couplings between data points based on optimal transport plans (type: np.array).

        self.transport_plans      = []                  # Optimal transport plans (type: list).

        self.transport_costs      = []                  # Costs of optimal transport plans (type: list)
 
        self.data                 = None                # Data to which the model is fitted.

    #Methods:

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        raise NotImplementedError



class MultistageOT(OTModel):
    """ Multistage Optimal mass Transport (MultistageOT) model.
         
        Finds an optimal temporal ordering and association between cells in a snapshot of single cell data:

                             o  (daughter cell)
                            /
            (parent cell) (*) -- o  (daughter cell)
                             
        
        #################################################################################################################

        Parameters
        ----------

        initial_cells        : list
                        List containing the indices corresponding to the initial marginal (i.e., the initial cells)
                        
        terminal_cells       : list
                        List containing the indices corresponding to the terminal marginal (i.e., the terminal cells)

        n_groups             : int
                        Integer specifying the number of intermediate marginal groups       

        fate_groups          : list
                        List of labels for each terminal fate group (default = None)

        epsilon              : float
                        Float specifying the value of the entropy regularization parameter
        
        ##################################################################################################################
                        
        # Example use:

        # Class is initialized i.e., via
 
        initial_cells    = [1,2,3]  # List of indices of root cells
        terminal_cells   = [7,8,9]  # List of indices of terminal cells
        T                = 21       # Final "time point"
        epsilon          = 0.01     # Regularization parameter

        msot = MultistageOT(initial_cells   = initial_cells,
                            terminal_cells  = terminal_cells,
                            n_groups        = T-1,
                            epsilon         = 0.015
                            )
    
        # Optimal couplings are found by running the public .fit() method:

        msot.fit(data)

        ##################################################################################################################
    """
    

    def __init__(self, 
                        initial_cells        : list = None,  
                        terminal_cells       : list  = None,  
                        n_groups             : int   = None,  
                        fate_groups          : list  = None, 
                        auxiliary_cell_cost  : float = None,  
                        epsilon              : float = None, 
                ):
        
        # Private attributes:  
        self._NUM_GROUPS        = n_groups
        self._EPSILON           = epsilon
       

        ## Proximal Sinkhorn scheme private attributes:
        self._PROXIMAL_EPSILON  = None

        self._PROXIMAL_EPSILON_HISTORY = []

        self._TOTAL_EPSILON_HISTORY = []

        self._INNER_ITERATIONS = []

        self._OUTER_PROXIMAL_ITERATIONS = 0

        self.__PRIOR            = None

        self.__USE_PRIOR        = None
        ##

        self._AUXILIARY_CELL_COST = auxiliary_cell_cost

        self._TRANSITION_MATRIX = None
        
    
        # Public attributes:
        self.initial_cells         = initial_cells 
        self.intermediate_cells = None
        self.terminal_cells     = terminal_cells
        self.fate_groups          = fate_groups
        self.median_cost          = None


        self.dual_variables         = [] # Dual variables (Lagrange multipliers) (type: list)
        self.utility_variables      = [[],[],0]
        self.history                = None




    def fit(self, data, verbose             : bool = True, 
                        log                 : bool = False, 
                        patience            : int   = 1, 
                        tolerance           : float = 1e-8,
                        prior               : list = None, 
                        sparse              : bool = False,
                        checkpoints         : int = None, 
                        path_to_checkpoints : str = None):

        """ Run MMOT algorithm to find optimal couplings in the data. Updates the following public class attributes:

                transport_plans       - Optimal transport plans (type: list).

                transport_costs       - Cost of optimal transport plans (type: list)

                dual_variables        - Dual variables (Lagrange multipliers)

                history               - Convergence history of variables 'max steps' and 'infeasibility'.

                checkpoints           - Number of checkpoints to save throughout Sinkhorn scheme (type: int)


            Input parameters: 
                    
                data                 : Pandas DataFrame object

                verbose              : bool
                                    Boolean: set to True for printouts, False otherwise (default : False).

                patience             : int
                                    Integer specifying the number of steps between each update and feasibility check (and any printouts) (defaults to 1)
            
                tolerance            : float
                                    Float specifying the tolerance in terms of maximum update steps and infeasibility (max_step + infeasbility) before terminating the Sinkhorn iterates          

            Returns:

                None


        """
        # Set verbose attribute:
        self._VERBOSE    = verbose

        # Set additional private attributes:
        self._PATIENCE   = patience
        self._TOLERANCE  = tolerance
        self._SPARSE     = sparse

        # Store the data:
        self.data = data
        self._set_intermediate_cells()

        self._path_to_checkpoints = path_to_checkpoints
        self._checkpoints = checkpoints

        self._n0 = len(self.initial_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.initial_cells)
        self._n = len(self.intermediate_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.intermediate_cells)
        self._nF = len(self.terminal_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.terminal_cells)

        # Define some constant utility vectors:
        self._1n0        = np.ones((self._n0,1))
        self._1n0_trans  = self._1n0.T
        self._1n        = np.ones((self._n,1))
        self._1n_trans  = self._1n.T
        self._1nT       = np.ones((self._nF,1))
        self._1nT_trans = self._1nT.T
        self._0n0       = np.zeros((self._n0,1))
        self._0n        = np.zeros((self._n,1))
        self._0nT       = np.zeros((self._nF,1))

        # Define the final "time":
        self._T = self._NUM_GROUPS + 1


        # Run Sinkorn iterations (block-coordinate ascent in the dual)
        if prior is not None: 
            self.__PRIOR = prior
            self.__USE_PRIOR = True
            self._sinkhorn_iterations_with_prior() 
        elif prior is None:
            self._sinkhorn_iterations()

        return



    def _converged(self,infeasibility, max_step):
        """ Utility funciton for determining whether the Sinkhorn algorithm has converged."""
        if (max_step + infeasibility) <= self._TOLERANCE:
            return True
        else: 
            return False

    def _set_intermediate_cells(self):
        """ Utility method for identifying intermediate indices """
        self.intermediate_cells = self.data.loc[~self.data.index.isin(self.initial_cells + self.terminal_cells)].index.tolist()
        return


    def _get_C(self):
        """ Utility method for computing the squared Euclidean distance between
        marginals (i.e. the pairwise cost between 'cells')"""


        start_data = self.data.loc[self.initial_cells].values
        mid_data   = self.data.loc[self.intermediate_cells].values
        end_data   = self.data.loc[self.terminal_cells].values

        mid_end_data = np.concatenate((mid_data, end_data), axis = 0)
        C_0 = scipy.spatial.distance.cdist(start_data, mid_data, 'sqeuclidean')   # Costs between initial and intermediate
        C  = scipy.spatial.distance.cdist(mid_data, mid_end_data, 'sqeuclidean') # Costs between intermediate and intermediate
        C_T_1  = scipy.spatial.distance.cdist(mid_data, end_data, 'sqeuclidean') # Costs between intermediate and terminal
        

    
        median = np.median(C_0.flatten().tolist()+C.flatten().tolist()+C_T_1.flatten().tolist())
        self.median_cost = median
        
        C_0 = C_0/median  
        C   = C/median 
        C_T_1 = C_T_1/median 

        #Modify C if mass pool is used:
        if (self._AUXILIARY_CELL_COST is not None):
            C_0     = np.block([[C_0, np.inf*np.ones((self._n0-1,1)), self._AUXILIARY_CELL_COST*np.ones((self._n0-1,1))], 
                                [self._AUXILIARY_CELL_COST*np.ones((1,self._n+1))]])
            
            C_tilde = C[:,:self._n-1]
            C_hat   = C_T_1
            
            C_tilde = np.block([[C_tilde, self._AUXILIARY_CELL_COST*np.ones((self._n-1,1))],
                            [self._AUXILIARY_CELL_COST*np.ones((1,self._n-1)), np.zeros((1,1))]])

            C_hat   = np.block([[C_hat, self._AUXILIARY_CELL_COST*np.ones((self._n-1,1))],[np.inf*np.ones((1,self._nF-1)), self._AUXILIARY_CELL_COST*np.ones((1,1))]])

            C = np.block([C_tilde, C_hat])
            
            C_T_1 = C_hat
            

        #Prohibit self-couplings by giving infinite cost:
        C[:,:self._n] += np.diag(np.ones(self._n)*np.inf)


        return [C_0, C, C_T_1]

    def _retrieve_plans(self, K_0, K, K_hat, u, v, s):
        """ Compute transport plans on the form M = K * (u v') """

        T = self._T

        M_0 = K_0 * (np.outer(u[0],v[1]))
        M_list = [M_0]

        for t in range(1, T-1):
            if (t == 1) & (self._AUXILIARY_CELL_COST is not None):
                M_list.append( K * ( np.outer(u[t][:-1]*s, np.vstack((v[t+1],v[T])) ) ) )
            else:
                M_list.append( K * ( np.outer(u[t]*s, np.vstack((v[t+1],v[T])) ) ) )

        M_list.append( K_hat * np.outer(u[T-1]*s, v[T]) )

        return M_list
    
    def _retrieve_plans_proximal(self, K_0, K, u, v, s):
        """ Compute transport plans on the form M = K * (u v') (for the proximal-point scheme) """

        T = self._T

        M_0 = K_0 * (np.outer(u[0],v[1]))
        M_list = [M_0]

        for t in range(1, T-1):
            if (t == 1) & (self._AUXILIARY_CELL_COST is not None):
                M_list.append( K[t] * ( np.outer(u[t][:-1]*s, np.vstack((v[t+1],v[T])) ) ) )
            else:
                M_list.append( K[t] * ( np.outer(u[t]*s, np.vstack((v[t+1],v[T])) ) ) )

        M_list.append( K[T-1] * np.outer(u[T-1]*s, v[T]) )

        return M_list


    def _compute_feasibility(self):
        """ Utility function for computing deviations from constraints"""

        T = self._T

        M = 0
        for i in range(1,len(self.transport_plans)-1):
            M += self.transport_plans[i]
        
        if (self._AUXILIARY_CELL_COST is not None):
            M = M[:-1,:]

            delta_0 = 1 - np.minimum(np.min(np.sum(self.transport_plans[0][:-1], axis=1)), 1)
            delta_n = 1 - np.minimum(np.min(np.sum(M,axis=1) + np.sum(self.transport_plans[-1][:-1,:],axis=1)), 1 ) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[1],axis=1)), 1)
            delta_T = 1 - np.minimum(np.min(np.sum([np.sum(self.transport_plans[k][:,-self._nF:-1],axis=0) for k in range(1,len(self.transport_plans))],axis=0)), 1) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[-1],axis=0)), 1)
        
        else:
            delta_0 = 1 - np.minimum(np.min(np.sum(self.transport_plans[0], axis=1)), 1)
            delta_n = 1 - np.minimum(np.min(np.sum(M,axis=1) + np.sum(self.transport_plans[-1],axis=1)), 1 ) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[1],axis=1)), 1)
            delta_T = 1 - np.minimum(np.min(np.sum([np.sum(self.transport_plans[k][:,-self._nF:],axis=0) for k in range(1,len(self.transport_plans))],axis=0)), 1) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[-1],axis=0)), 1)

        delta_t = []
        for t in range(1,T):
            delta_t.append( np.max(np.abs(np.sum(self.transport_plans[t-1][:,:self._n], axis=0) - np.sum(self.transport_plans[t], axis=1))) )  
        
        delta  = np.array([delta_0] + [delta_n] + delta_t + [delta_T])
        
        self.delta_dict = {'mu_0' : delta_0, 'mu_sum' : delta_n, 'max_mu_t' : np.max(delta_t), 'mu_T' : delta_T}

        self.delta_vec = delta
        
        return




    def _printouts(self, iter, max_step, infeasibility, curr_time ):
        """ Utility function for printing progress in the Sinkhorn scheme """
        print("\r", "Iteration: {k} [========]".format(k=iter) + " Max dual step: {0:.3e}".format(max_step) + " | Infeasibility: {0:.3e}".format(infeasibility) + " | Elapsed time: {time}".format(time=timedelta(seconds=curr_time)), end = "", flush=True)

        return


    def _sinkhorn_iterations(self):
        """ Perform Sinkhorn iterations on the OT problem """

        T = self._T
        self.__C = self._get_C() 
        
        #Compute K-matrices:
        if self._SPARSE: #Sparse implementation
            K             = scipy.sparse.csr_array(np.exp(-self.__C[1] / self._EPSILON))
            print("Sparsity of K: ", scipy.sparse.csr_matrix.count_nonzero(K) / (K.shape[0]*K.shape[1]))
        else:
            K         = np.exp(-self.__C[1] / self._EPSILON)

        K_0           = np.exp(-self.__C[0] / self._EPSILON)
        K_0_tilde     = K_0[:,:self._n]
        K_hat         = np.exp(-self.__C[2] / self._EPSILON)
        K_tilde       = K[:,:self._n]
 
        #Transposes:
        K_0_trans       = K_0.T
        K_0_tilde_trans = K_0_tilde.T
        K_hat_trans     = K_hat.T
        K_tilde_trans   = K_tilde.T

        #Initialize variables:
        if len(self.dual_variables) > 0: #If the model is already trained, start from current estimates:
            u = [np.exp(self.dual_variables[t]/self._EPSILON) for t in range(T + 1)]
            s = np.exp(self.dual_variables[-1]/self._EPSILON)
            v = [1 / u[t] for t in range(T + 1)]
        else:    #Otherwise, start from an initial guess:
            u     = [np.ones((self._n,1)) for t in range(T + 1)]
            if self._AUXILIARY_CELL_COST is not None:
                u[1] = np.ones((self._n+1,1))

            u[0]  = np.ones((self._n0,1))
            u[T]  = np.ones((self._nF,1))
            
            v = [1 / u[t] for t in range(T + 1)]
            
            s = np.ones((self._n,1))

        if (self._AUXILIARY_CELL_COST is not None):
                u_sum    = np.sum(u[2:-1],axis=0) + u[1][:-1]
        else:
            u_sum    = np.sum(u[1:-1],axis=0)


        if self.history is None:
            max_steps       = []
            infeasibilities = []
            ts              = []
        else:
            max_steps       = self.history['max_steps']
            infeasibilities = self.history['infeasibility']
            ts              = self.history['wall_clock_time']

        iteration = 0


        # Compute initial progression variables:
        self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        self._compute_feasibility()
        infeasibility = np.max(self.delta_vec)
        max_step   = 1

        # Sinkhorn iterations:
        start_time = time.time()
        while not self._converged(infeasibility, max_step):

            old_u = copy.deepcopy(u)
            old_s = copy.deepcopy(s)
            
            # Update v:
            v = [1 / u[t] for t in range(T + 1)]
            R =  K_hat @ v[T]
            
            # Update s:
            if (self._AUXILIARY_CELL_COST is not None):
                s_sum = u[1][:-1] * (K_tilde @ v[2]) + np.sum([u[t] * (K_tilde @ v[t + 1]) for t in range(2, T-1)], axis=0) +  u_sum*R
            else:
                s_sum = np.sum([u[t] * (K_tilde @ v[t + 1]) for t in range(1, T-1)], axis=0) +  u_sum*R
            s = np.maximum(self._1n, 1 / s_sum)

            if (self._AUXILIARY_CELL_COST is not None):
                s[-1,0] = 1

            # Update u:
            u[0] = np.maximum(self._1n0, 1 / (K_0 @ v[1]) )
            if (self._AUXILIARY_CELL_COST is not None):
                u[0][-1,0] = 1

            if (self._AUXILIARY_CELL_COST is not None):
                u[1][:-1] = np.sqrt((K_0_tilde_trans @ u[0]) / (s * (K_tilde @ v[2] + R)))
                u[1][-1,0] = 1
            else:
                u[1] = np.sqrt((K_0_trans @ u[0]) / (s * (K_tilde @ v[2] + R)))

            for t in range(2, T-1):
                if (t == 2) & (self._AUXILIARY_CELL_COST is not None):
                    u[t] = np.sqrt((K_tilde_trans @ (s * u[t-1][:-1])) / (s * (K_tilde @ v[t+1] + R)))
                else:
                    u[t] = np.sqrt((K_tilde_trans @ (s * u[t-1])) / (s * (K_tilde @ v[t+1] + R)))

            u[T-1] = np.sqrt((K_tilde_trans @ (s * u[T-2])) / (s * (K_hat @ v[T])))

            if (self._AUXILIARY_CELL_COST is not None):
                u_sum    = np.sum(u[2:-1],axis=0) + u[1][:-1]
            else:
                u_sum    = np.sum(u[1:-1],axis=0)

            u[T]     = np.minimum(self._1nT, K_hat_trans @ (s * u_sum))
            
            if (self._AUXILIARY_CELL_COST is not None):
                u[T][-1,0] = 1

            if (iteration%self._PATIENCE == 0):

                self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)
                self._compute_feasibility()

                # Compute change in the dual variables:
                max_step = np.maximum(np.max([np.max(np.abs(self._EPSILON*np.log(old_u[t]) - self._EPSILON*np.log(u[t])))  for t in range(len(old_u))]), np.max(np.abs(self._EPSILON*np.log(old_s) - self._EPSILON*np.log(s))))
                
                max_steps.append(max_step)
                

                infeasibility = np.max(self.delta_vec)

                infeasibilities.append(infeasibility)
                
                curr_time = time.time() - start_time
                if self._VERBOSE:
                    
                    self._printouts(iteration, max_step, infeasibility, curr_time)
                ts.append(curr_time)
                
            # Save checkpoint:
            if (self._path_to_checkpoints is not None) and (iteration%self._checkpoints== 0):

                self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

                self.dual_variables = [] #Reset dual variables
                # Store dual variables:
                for t in range(len(u)):
                    self.dual_variables.append(self._EPSILON*np.log(u[t]))
                self.dual_variables.append(self._EPSILON*np.log(s))

                self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}


                #Create directory:
                directory = self._path_to_checkpoints + "/checkpoint_{it}".format(it=iteration)
                os.mkdir(directory)

                self.save_model(directory)


            del old_s
            del old_u

            iteration += 1


        if self._VERBOSE:
            print("\n")
            print("Sinkhorn algorithm converged to a solution within the given tolerance ({0:.4e}) of both feasibility and max dual-variable update step.".format(self._TOLERANCE))
            # Retrieve transport plans:
            print("\n")
            print("Retrieving transport plans...")

        self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        if self._VERBOSE:
            print("Done.")

        # Store dual variables:
        self.dual_variables = [] #Reset dual variables
        if self._VERBOSE:
            print("Storing dual variables...")
        for t in range(len(u)):
            self.dual_variables.append(self._EPSILON*np.log(u[t]))
        self.dual_variables.append(self._EPSILON*np.log(s))
        if self._VERBOSE:
            print("Done.")

        # Store utility variables:
        for t in range(len(u)):
            self.utility_variables[0].append(u[t])
        for t in range(len(v)):
            self.utility_variables[1].append(v[t])
        self.utility_variables[2] = s
        
        if self._VERBOSE:
            print("Max absolute feasibility errors in the marginal constraints: \n ", self.delta_dict)
        self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}
        
        return


    def _sinkhorn_iterations_with_prior(self):
        """ Perform Sinkhorn iterations on the OT problem with a proximal entropy regularization
        (where the "prior" is previously obtained transport plans corresponding to some choice of the regularization parameter (epsilon)).

        """
        T = self._T
        self.__C = self._get_C()
    
        #Compute G-matrices (G = K*P, where P is a prior transport plan, and K is the "standard" OT exponential matrix K = exp(-C/eps)):
        G_0           = np.exp(-self.__C[0] / self._PROXIMAL_EPSILON) * self.__PRIOR[0]

        G_0_tilde     = G_0[:,:self._n]
        G             = [G_0] + [np.exp(-self.__C[1] / self._PROXIMAL_EPSILON) * self.__PRIOR[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / self._PROXIMAL_EPSILON) * self.__PRIOR[T-1]]
        G_hat         = [None] + [G[t][:,self._n:] for t in range(1,T-1)] + [G[T-1]]
        G_tilde       = [None] + [G[t][:,:self._n] for t in range(1,T-1)]

        #Transposes:
        G_0_trans     = G_0.T 
        G_0_tilde_trans = G_0_tilde.T
        G_hat_trans   = [None] + [G_hat[t].T for t in range(1,T)]
        G_tilde_trans = [None] + [G_tilde[t].T for t in range(1,T-1)]


        
        #Initialize variables:
        if len(self.utility_variables[0]) > 0: #If the model is already trained, start from current estimates:
            u = [self.utility_variables[0][t] for t in range(T + 1)]
            v = [self.utility_variables[1][t] for t in range(T + 1)]
            s = self.utility_variables[2]
            

        else:    #Otherwise, start from an initial guess (may cause instability):
            u     = [np.ones((self._n,1)) for t in range(T + 1)]
            if self._AUXILIARY_CELL_COST is not None:
                u[1] = np.ones((self._n+1,1))
            u[0]  = np.ones((self._n0,1))
            u[T]  = np.ones((self._nF,1))
            v = [1 / u[t] for t in range(T + 1)] 
            s = np.ones((self._n,1))

            
        if self.history is None:
            max_steps       = []
            infeasibilities = []
            ts              = []
        else:
            max_steps       = self.history['max_steps']
            infeasibilities = self.history['infeasibility']
            ts              = self.history['wall_clock_time']

        iteration = 0

        # Compute initial progression variables:
        self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
        self._compute_feasibility()
        infeasibility = np.max(self.delta_vec)
        max_step   = 1

        
        # Sinkhorn iterations:
        start_time = time.time()


            
        while not self._converged(infeasibility, max_step):

            # Check for numerical instability:
            u_list = []
            for elem in u:
                u_list += elem.flatten().tolist()
            nan_detected_in_u = np.isnan(u_list).any()
            zeros_detected_in_u = (np.array(u_list).size - np.count_nonzero(u_list)) > 0


            #############################################################################################
            # Check for instability:
            if nan_detected_in_u or zeros_detected_in_u:# np.isnan(max_step):

                
                #Update epsilon if it is unstable:
                self._PROXIMAL_EPSILON = 1.1*self._PROXIMAL_EPSILON
                
                if self._VERBOSE:
                    print("NaNs encountered, increasing proximal epsilon to ->", self._PROXIMAL_EPSILON)

                #Re-compute K-matrices:
                G_0           = np.exp(-self.__C[0] / self._PROXIMAL_EPSILON) * self.__PRIOR[0]
                
                G_0_tilde     = G_0[:,:self._n]    


                G             = [G_0] + [np.exp(-self.__C[1] / self._PROXIMAL_EPSILON) * self.__PRIOR[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / self._PROXIMAL_EPSILON) * self.__PRIOR[T-1]]
                G_hat         = [None] + [G[t][:,self._n:] for t in range(1,T-1)] + [G[T-1]]
                G_tilde       = [None] + [G[t][:,:self._n] for t in range(1,T-1)]

                #Transposes:
                G_0_trans     = G_0.T
                G_0_tilde_trans = G_0_tilde.T
                G_hat_trans   = [None] + [G_hat[t].T for t in range(1,T)]
                G_tilde_trans = [None] + [G_tilde[t].T for t in range(1,T-1)]


                #Initialize variables:
                if (len(self.utility_variables[0]) > 0) and (iteration > 0): #If the model is already trained, and it is not the first iteration, then start from current estimates:
                    u = [self.utility_variables[0][t] for t in range(T + 1)]
                    v = [self.utility_variables[1][t] for t in range(T + 1)]
                    s = self.utility_variables[2]

                else:   #Otherwise, start from a different initial guess:
                    u     = [np.ones((self._n,1)) for t in range(T + 1)]
                    if self._AUXILIARY_CELL_COST is not None:
                        u[1] = np.ones((self._n+1,1))
                    u[0]  = np.ones((self._n0,1))
                    u[T]  = np.ones((self._nF,1))
                    v = [1 / u[t] for t in range(T + 1)] 
                    s = np.ones((self._n,1))
                    
                if self.history is None:
                    max_steps       = []
                    infeasibilities = []
                    ts              = []
                else:
                    max_steps       = self.history['max_steps']
                    infeasibilities = self.history['infeasibility']
                    ts              = self.history['wall_clock_time']

                iteration = 0
            #############################################################################################


            old_u = copy.deepcopy(u)
            old_s = copy.deepcopy(s)
            
                        
            # Update v: 
            v = [1 / u[t] for t in range(T + 1)]

            
            # Update s:
            if (self._AUXILIARY_CELL_COST is not None):
                s = np.maximum(self._1n, 1 / ( u[T-1] * (G_hat[T-1] @ v[T]) + u[1][:-1]*(G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) + np.sum([u[t] * (G_tilde[t] @ v[t + 1] + G_hat[t] @ v[T] ) for t in range(2, T-1)], axis=0) ) )
            else:
                s = np.maximum(self._1n, 1 / ( u[T-1] * (G_hat[T-1] @ v[T]) + np.sum([u[t] * (G_tilde[t] @ v[t + 1] + G_hat[t] @ v[T] ) for t in range(1, T-1)], axis=0) ) )

            if (self._AUXILIARY_CELL_COST is not None):
                s[-1,0] = 1

            # Update u:
            u[0] = np.maximum(self._1n0, 1 / (G_0 @ v[1]) ) 

            if (self._AUXILIARY_CELL_COST is not None):
                u[0][-1,0] = 1

            if (self._AUXILIARY_CELL_COST is not None):
                u[1][:-1] = np.sqrt((G_0_tilde_trans @ u[0]) / (s * ( G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) ))
                u[1][-1,0] = 1
            else:
                u[1] = np.sqrt((G_0_trans @ u[0]) / (s * ( G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) ))
                
            for t in range(2, T-1):
                if (t == 2) & (self._AUXILIARY_CELL_COST is not None):
                    u[t] = np.sqrt((G_tilde_trans[t-1] @ (s * u[t-1][:-1])) / (s * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T]) ))
                else:
                    u[t] = np.sqrt((G_tilde_trans[t-1] @ (s * u[t-1])) / (s * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T]) ))
            
            u[T-1] = np.sqrt((G_tilde_trans[T-2] @ (s * u[T-2])) / (s * (G_hat[T-1] @ v[T]) ))
            
            if (self._AUXILIARY_CELL_COST is not None):
                u[T]   = np.minimum(self._1nT, G_hat_trans[T-1]@(s*u[T-1]) + G_hat_trans[1] @ (s*u[1][:-1])  + np.sum([G_hat_trans[t] @ (s*u[t]) for t in range(2, T-1)], axis=0) )
            else:
                u[T]   = np.minimum(self._1nT, G_hat_trans[T-1]@(s*u[T-1]) + np.sum([G_hat_trans[t] @ (s*u[t]) for t in range(1, T-1)], axis=0) )

            if (self._AUXILIARY_CELL_COST is not None):
                u[T][-1,0] = 1

            
            # Compute change in the dual variables:
            max_step = np.maximum(np.max([np.max(np.abs(self._PROXIMAL_EPSILON*np.log(old_u[t]) - self._PROXIMAL_EPSILON*np.log(u[t])))  for t in range(len(old_u))]), np.max(np.abs(self._PROXIMAL_EPSILON*np.log(old_s) - self._PROXIMAL_EPSILON*np.log(s))))
            


            if (iteration%self._PATIENCE == 0):
                self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
                self._compute_feasibility()

                

                curr_time = time.time() - start_time
                max_steps.append(max_step)
                ts.append(curr_time)

                infeasibility = np.max(self.delta_vec)

                infeasibilities.append(infeasibility)

                if self._VERBOSE:
                    self._printouts(iteration, max_step, infeasibility, curr_time)

            # Save checkpoint:
            if (self._path_to_checkpoints is not None) and (iteration%self._checkpoints== 0):

                self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)

      
                # Store dual variables:
                for t in range(len(u)):
                    self.utility_variables[0].append(u[t])
                for t in range(len(v)):
                    self.utility_variables[1].append(v[t])
                self.utility_variables[2] = s

                self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}


                #Create directory:
                directory = self._path_to_checkpoints + "/checkpoint_{it}".format(it=iteration)
                os.mkdir(directory)

                self.save_model(directory)


            del old_s
            del old_u

            iteration += 1

        
        #Free up some memory:
        del self.__PRIOR

        if self._VERBOSE:
            print("\n")
            print("Sinkhorn algorithm converged to a solution within the given tolerance ({0:.4e}) of both feasibility and max dual-variable update step.".format(self._TOLERANCE))
            # Retrieve transport plans:
            print("\n")
            print("Retrieving transport plans...")

        self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
        if self._VERBOSE:
            print("Done.")

        if self._VERBOSE:
            print("Storing utility variables...")


        # Store dual variables:
        self.dual_variables = [] #Reset dual variables
        if self._VERBOSE:
            print("Storing dual variables...")
        for t in range(len(u)):
            self.dual_variables.append(self._PROXIMAL_EPSILON*np.log(u[t]))
        self.dual_variables.append(self._PROXIMAL_EPSILON*np.log(s))
        if self._VERBOSE:
            print("Done.")


        # Store utility variables:
        self.utility_variables      = [[],[],0] #Reset utility_variables

        for t in range(len(u)):
            self.utility_variables[0].append(u[t])
        for t in range(len(v)):
            self.utility_variables[1].append(v[t])
        
        self.utility_variables[2] = s

        if self._VERBOSE:
            print("Done.")
        
        if self._VERBOSE:
            print("Max absolute feasibility errors in the marginal constraints: \n ", self.delta_dict)
        self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}
        
        self._EPSILON = 1/(1/self._EPSILON + 1/self._PROXIMAL_EPSILON)
        return        


    #Public methods:
    def save_model(self, path):
        """ Save the model to path """
        if (len(os.listdir(path)) != 0):
            sys.exit('WARNING: The given path directory is NOT empty. Please specify an empty directory to avoid overwriting an existing model. ')

        print(">>> Saving MMOT model...")
        print("To: ", path) 
        for t in range(len(self.transport_plans)):
            np.save(path + "lambda_{t}".format(t=t), self.dual_variables[t])      
        np.save(path + "lambda_{t}".format(t=t+1), self.dual_variables[t+1])   
        np.save(path + "rho", self.dual_variables[-1])

        if self.__USE_PRIOR:
            for i in range(len(self.__prior_utility_variables)):
                for t in range(self._T+1):
                    np.save(path + "{i}_prior_u_{t}".format(t=t,i=i), self.__prior_utility_variables[i][0][t])
                    np.save(path + "{i}_prior_v_{t}".format(t=t,i=i), self.__prior_utility_variables[i][1][t])
                np.save(path + "{i}_prior_s".format(i=i), self.__prior_utility_variables[i][2])
            
        np.save(path + "initial_cells", self.initial_cells)
        np.save(path + "intermediate_cells", self.intermediate_cells)
        np.save(path + "terminal_cells", self.terminal_cells)
        
        #Data:
        self.data.to_csv(path + "data.csv")

        #Parameters:
        np.save(path + "NUM_GROUPS", self._NUM_GROUPS)
        if self.__USE_PRIOR:  
            np.save(path +"PROXIMAL_EPSILON_HISTORY", self._PROXIMAL_EPSILON_HISTORY)
            np.save(path +"TOTAL_EPSILON_HISTORY", self._TOTAL_EPSILON_HISTORY)
            np.save(path +"INNER_ITERATIONS", self._INNER_ITERATIONS) 
            np.save(path + "OUTER_PROXIMAL_ITERATIONS", self._OUTER_PROXIMAL_ITERATIONS)

        np.save(path + "AUXILIARY_CELL_COST", self._AUXILIARY_CELL_COST)
        np.save(path + "EPSILON", self._EPSILON)
        np.save(path + "PATIENCE", self._PATIENCE)
        np.save(path + "TOLERANCE", self._TOLERANCE)
        np.save(path + "WALL_CLOCK_TIME", self.history['wall_clock_time'])
        np.save(path + "MAX_STEPS", self.history['max_steps'])
        np.save(path + "INFEASIBILITY", self.history['infeasibility'])

        print("Done.")
        return
    

    def load_model(self, path):
        """ Load the model from path. """
        
        def get_file_names_containing_string(string):
            full_list = os.listdir(path)
            final_list = [filename for filename in full_list if string in filename]
            return final_list

        print("<<< Loading MMOT model...")
        print("From: ", path)

        # Reset public attributes:
        self.couplings = None   # Strength of couplings between data points based on optimal transport plans (type: np.array).

        # Set information about artificial cell cost:
        try:
            self._AUXILIARY_CELL_COST = np.load(path + "AUXILIARY_CELL_COST.npy")
        except:
            self._AUXILIARY_CELL_COST = None

        lambda_files = get_file_names_containing_string("lambda_")
            
        self.dual_variables = [np.load(path + "lambda_{t}.npy".format(t=t)) for t in range(len(lambda_files))]
        self.dual_variables.append(np.load(path + "rho.npy"))

        prior_u_files = get_file_names_containing_string("prior_u_") 
        prior_v_files = get_file_names_containing_string("prior_v_")


        if len(prior_u_files) > 0 or len(prior_v_files) > 0:
            self.__USE_PRIOR = True
        

        if self.__USE_PRIOR:   
            self._OUTER_PROXIMAL_ITERATIONS = int(np.load(path + "OUTER_PROXIMAL_ITERATIONS.npy")) 
            self.__prior_utility_variables = []
            
            self._PROXIMAL_EPSILON_HISTORY = np.load(path + "PROXIMAL_EPSILON_HISTORY.npy").tolist()
            self._TOTAL_EPSILON_HISTORY = np.load(path + "TOTAL_EPSILON_HISTORY.npy").tolist()
            self._INNER_ITERATIONS = np.load(path + "INNER_ITERATIONS.npy").tolist()

            for i in range(self._OUTER_PROXIMAL_ITERATIONS+1):

                temp = [0,0,0]

                temp[0] = [np.load(path + "{i}_prior_u_{t}.npy".format(t=t,i=i)) for t in range(len(lambda_files))]
                temp[1] = [np.load(path + "{i}_prior_v_{t}.npy".format(t=t,i=i)) for t in range(len(lambda_files))]
                temp[2] = np.load(path + "{i}_prior_s.npy".format(i=i))

                self.__prior_utility_variables.append(temp)
        

        # Define the final "time":
        self._NUM_GROUPS = int(np.load(path + "NUM_GROUPS.npy"))
        self._T = self._NUM_GROUPS + 1

        T = self._T

        #Initialize variables:
        self._EPSILON    = float(np.load(path + "EPSILON.npy"))

        #Retrieve indices:
        self.initial_cells         = np.load(path + "initial_cells.npy").tolist()
        self.intermediate_cells = np.load(path + "intermediate_cells.npy").tolist()
        self.terminal_cells     = np.load(path + "terminal_cells.npy").tolist() 
        index_type = type(self.initial_cells[0])

        self._n0 = len(self.initial_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.initial_cells)
        self._n = len(self.intermediate_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.intermediate_cells)
        self._nF = len(self.terminal_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.terminal_cells)

        #Data:
        self.data = pd.read_csv(path + "data.csv", index_col='Unnamed: 0')
        self.data.index = self.data.index.astype(index_type)

        #Retrieve transport plans:
        self.__C = self._get_C()

        
        if self.__USE_PRIOR: # Retrieve transport plans by computing a sequence of prior transport plans (for Proximal Sinkhorn):
            proximal_epsilons = self._PROXIMAL_EPSILON_HISTORY[1:]

            self._PROXIMAL_EPSILON = self._PROXIMAL_EPSILON_HISTORY[-1]

            K         = np.exp(-self.__C[1] / proximal_epsilons[0])
            K_0       = np.exp(-self.__C[0] / proximal_epsilons[0])
            K_hat     = np.exp(-self.__C[2] / proximal_epsilons[0])
                
            u = self.__prior_utility_variables[0][0]
            v = self.__prior_utility_variables[0][1]
            s = self.__prior_utility_variables[0][2]
            prior = self._retrieve_plans(K_0, K, K_hat, u, v, s)
            
            for i in range(1,self._OUTER_PROXIMAL_ITERATIONS+1):
                
                G_0  = np.exp(-self.__C[0] / proximal_epsilons[i]) * prior[0]
                G    = [G_0] + [np.exp(-self.__C[1] / proximal_epsilons[i]) * prior[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / proximal_epsilons[i]) * prior[T-1]]

                u = self.__prior_utility_variables[i][0]
                v = self.__prior_utility_variables[i][1]
                s = self.__prior_utility_variables[i][2]

                prior = self._retrieve_plans_proximal(G_0, G, u, v, s) 
                
            self.transport_plans = prior
            del prior


        else: # Retrieve transport plans:
            K         = np.exp(-self.__C[1] / self._EPSILON)
            K_0       = np.exp(-self.__C[0] / self._EPSILON)
            K_hat     = np.exp(-self.__C[2] / self._EPSILON)

            u = [np.exp(self.dual_variables[t]/self._EPSILON) for t in range(T + 1)]
            s = np.exp(self.dual_variables[-1]/self._EPSILON)
            v = [1 / u[t] for t in range(T + 1)]

            self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        self.utility_variables = [u, v, s] #Set utility variables

        self.history = {'max_steps' : np.load(path + "MAX_STEPS.npy").tolist(), 'wall_clock_time' : np.load(path + "WALL_CLOCK_TIME.npy").tolist(), 'infeasibility' : np.load(path + "INFEASIBILITY.npy").tolist()}

        # Load additional constant parameters:
        self._PATIENCE   = int(np.load(path + "PATIENCE.npy"))
        self._TOLERANCE  = float(np.load(path + "TOLERANCE.npy"))

        if self.__USE_PRIOR:
            # Load proximal related history:
            self.history['proximal_sinkhorn'] = {'iterations' : self._INNER_ITERATIONS, 'proximal_epsilon' : self._PROXIMAL_EPSILON_HISTORY, 'total_epsilon' : self._TOTAL_EPSILON_HISTORY}   

        print("Done.")



    def marginals(self):
        """ Compute marginals based on the fitted model's transport plans

            Returns:
            --------
            
            marginals - pandas DataFrame where the rows correspond to the intermediate cells, and the columns correspond 
                        to the amount of mass sent from a given intermediate cell in the different marginal groups

        
        """
        if self._AUXILIARY_CELL_COST is not None:
            mu_t = np.array([np.sum(self.transport_plans[k][:-1,:],axis=1) for k in range(1,len(self.transport_plans))]).T
        else:
            mu_t = np.array([np.sum(self.transport_plans[k],axis=1) for k in range(1,len(self.transport_plans))]).T

        marginal_labels = ["mu_{t}".format(t=t) for t in range(1,self._T) ]
  
        
        marginals_df = pd.DataFrame(mu_t, columns=marginal_labels, index=self.intermediate_cells)

        if self._AUXILIARY_CELL_COST is not None:
            root_df = pd.DataFrame(np.nan*np.zeros((self._n0-1, len(marginal_labels))), columns=marginal_labels, index=self.initial_cells)
        else:
            root_df = pd.DataFrame(np.nan*np.zeros((self._n0, len(marginal_labels))), columns=marginal_labels, index=self.initial_cells)

        if self._AUXILIARY_CELL_COST is not None:
            terminal_df = pd.DataFrame(np.nan*np.zeros((self._nF-1, len(marginal_labels))), columns=marginal_labels, index=self.terminal_cells)
        else:
            terminal_df = pd.DataFrame(np.nan*np.zeros((self._nF, len(marginal_labels))), columns=marginal_labels, index=self.terminal_cells)

        marginals_df = pd.concat((root_df, marginals_df, terminal_df)).sort_index()

        return marginals_df


    def mass_to_terminals(self):
        """ Compute mass sent to each final cell in each time step. 

            Returns:
            --------
            mass_to_terminals - pandas DataFrame object where the rows correspond to the terminal cells, and the columns
                                correspond to the amount of mass sent to a given terminal cell in the different marginal groups.

            """  
        if self._AUXILIARY_CELL_COST is not None:
            nu_t = np.array([np.sum(self.transport_plans[k][:,-self._nF:-1],axis=0) for k in range(1,len(self.transport_plans))]).T
        else:
            nu_t = np.array([np.sum(self.transport_plans[k][:,-self._nF:],axis=0) for k in range(1,len(self.transport_plans))]).T

        nu_labels = ["nu_{t}".format(t=t) for t in range(1,self._T)]
        
        mass_to_terminals = pd.DataFrame(nu_t, columns=nu_labels, index=self.terminal_cells)
        return mass_to_terminals


    def cost_of_transport(self):
        """ Compute the cost of transport based on the fitted model's transport plans

            Returns:
            --------
            cost_of_transport - float corresponding to the transport cost of the unregularized problem based on the fitted model's transport plans

        """

        C = self._get_C()

        C[0][C[0] > 1e32] = 0
        C[1][C[1] > 1e32] = 0
        C[-1][C[-1] > 1e32] = 0

        cost = np.sum(self.transport_plans[0]*C[0])
        for t in range(1,self._T-1):
            cost += np.sum(self.transport_plans[t]*C[1])
        cost += np.sum(self.transport_plans[self._T-1]*C[-1])

        return cost


    def max_marginal_groups(self):
        """ Find the max marginal group for all intermediate cells in the given data set based on the 
            fitted model's transport plans

            Returns:
            --------

            cell_annotation - dict of the form { cell_index : max_group } """

        mu = self.marginals()
        cell_annotation = {}
        #Get max groups:
        for i in self.intermediate_cells:
            if np.max(mu.loc[i]) > 1e-8:
                cell_annotation[i] = np.argmax(mu.loc[i]) + 1
            else:
                cell_annotation[i] = np.nan
        return cell_annotation



    def transition_matrix(self):
        """ Construct a transition matrix based on the transport plans 
                    
            Returns: 

            transition_matrix :  N x N numpy array, where N is the number of states (cells).

        """

        M = np.sum(self.transport_plans[1:-1],axis=0)
        M[:,-self._nF:] += self.transport_plans[-1]
        if self._AUXILIARY_CELL_COST is not None:
            TM = np.block([[np.zeros((self._n0,self._n0)),  self.transport_plans[0][:,:-1], np.zeros((self._n0,self._nF-1)), np.array([self.transport_plans[0][:,-1]]).T],
                            [np.zeros((self._n, self._n0)), M],
                        [np.zeros((self._nF, self._n0+self._n)), np.eye(self._nF)]] )
        else:
            TM = np.block([[np.zeros((self._n0,self._n0)),  self.transport_plans[0], np.zeros((self._n0,self._nF))],
                        [np.zeros((self._n, self._n0)), M],
                      [np.zeros((self._nF, self._n0+self._n)), np.eye(self._nF)]] )

        TM = np.diag(1/np.sum(TM,axis=1))@TM

        return TM
    
    def mean_absorption_time(self,transition_matrix=None):
        """ Compute mean absorption time for each cell state
           in a Markov chain model based on the optimal transport plans 
           
            Returns:
            --------
           
            mat -  mean absorption time for each cell in the form of a Pandas Series """
        
        if transition_matrix is None:
            transition_matrix = self.transition_matrix()

        I  = np.eye(self._n0+self._n)
        P  = transition_matrix[:self._n0+self._n,:self._n0+self._n]
        _1 = np.ones(P.shape[0])

        
        u  = np.linalg.solve(I-P,_1)
        u  = np.concatenate((u,np.zeros(self._nF)),axis=0) 
        
        u_df = pd.Series(u)
        if self._AUXILIARY_CELL_COST is not None:
            u_df.index = self.initial_cells+['auxiliary_initial']+self.intermediate_cells+['auxiliary_intermediate']+self.terminal_cells + ['auxiliary_terminal'] 
        else: 
            u_df.index = self.initial_cells+self.intermediate_cells+self.terminal_cells



        return u_df


    def mean_marginal_group(self):
        """ Compute a mean marginal group for each intermediate cell
        
            Returns:
            --------

            mean_marginals - mean marginal group for each intermediate cell (as a Pandas series). """
        
        #Compute mass transport from the intermediate cells in each marginal:
        mu_df = self.marginals().loc[self.intermediate_cells]

        #Normalize so all cells send a total of 1 mass over all marginals:
        mu_df_normed = np.diag(1/mu_df.sum(axis=1))@mu_df 

        mean_marginals_df = pd.Series(mu_df_normed @ np.array([k for k in range(1,mu_df.shape[1]+1)]), name='mean_marginal_group', index=self.initial_cells+self.intermediate_cells+self.terminal_cells).sort_index()
        
        return mean_marginals_df



    def pseudotemporal_order(self):
        """ Compute a temporal ordering of based on the mean marginal group for each cell.
         
            Returns:
            --------

            pseudotime - A pseudotemporal ordering of all cells in the snapshot (as a Pandas series). """
        
        #Compute mass transport to terminals in each marginal:
        nu_df = self.mass_to_terminals()

        #Normalize so all cells receive a total of 1 mass over all marginals:
        nu_df_normed = np.diag(1/nu_df.sum(axis=1))@nu_df 

        #Compute mass transport from the intermediate cells in each marginal:
        mu_df = self.marginals().loc[self.intermediate_cells]

        #Normalize so all cells send a total of 1 mass over all marginals:
        mu_df_normed = np.diag(1/mu_df.sum(axis=1))@mu_df 


        #Compute pseudotime for terminal cells and intermediate cells:
        pseudotime_mean_terminal = nu_df_normed @ np.array([k for k in range(1,nu_df.shape[1]+1)])
        pseudotime_mean_intermediate = mu_df_normed @ np.array([k for k in range(1,mu_df.shape[1]+1)])

        pseudotime_mean = pd.concat((pseudotime_mean_intermediate,pseudotime_mean_terminal)).reset_index(drop=True)

        temporal_ordering_normed_mid_end = pd.DataFrame(pseudotime_mean.sort_values())
        temporal_ordering_normed_mid_end['order'] = [k for k in range(1, temporal_ordering_normed_mid_end.shape[0]+1)]
        temporal_ordering_normed_mid_end = temporal_ordering_normed_mid_end.sort_index()['order']/(temporal_ordering_normed_mid_end.shape[0])

        temporal_ordering_normed = np.concatenate((np.zeros(len(self.initial_cells)),temporal_ordering_normed_mid_end.values))

        temporal_ordering_normed_df = pd.Series(temporal_ordering_normed, name='pseudotime',index=self.initial_cells+self.intermediate_cells+self.terminal_cells).sort_index()

        return temporal_ordering_normed_df
    
    def _get_global_to_local_index_dict(self):

        global_indices = self.initial_cells + self.intermediate_cells + self.terminal_cells

        return {global_indices[k] : k for k in range(len(self.initial_cells + self.intermediate_cells + self.terminal_cells))}

    def cell_fate_probabilities(self, fate_groups = None, transition_matrix=None):
        """ Compute cell fate probabilities as the absorption probabilities in a Markov chain model
            based on the optimal transport plans.

            Parameters:
            ----------
            fate_groups   : dict
                            dictionary of key:value pairs of the form <fate_label> : <index_array>, where <fate_label> is a name (string) of
                            a terminal fate (i.e., 'erythroid'), and <index_array> is the indices corresponding to that class of cells.
            
            transition_matrix (optional) : numpy array
                                           Pre-computed N x N matrix (N = number of cells in the snapshot) of elements t_{ij}, encoding the transition probabilities
                                           between every cell i and every other cell j in the snapshot.
            Returns:
            ----------
             
              cell_fate_probabilities : numpy array
                                        S x F matrix of cell fate probabilities (likelihood of absorption in each class of terminal fates) where 
                                        S is the number of states (cells) and F the number of terminal fate classes."""
        
        I  = np.eye(self._n0+self._n)
        if transition_matrix is None:
            transition_matrix = self.transition_matrix()

        P  = transition_matrix[:self._n0+self._n,:self._n0+self._n]
        S  = transition_matrix[:self._n0+self._n,self._n0+self._n:]

        A  = np.linalg.solve(I-P,S) #Solves a linear system of equations for the absoprtion probabilities A_{ij}. 
                                    #Where i is a cell index, and j a terminal cell state.
        
        
        A  = np.concatenate((A, np.eye(self._nF)),axis=0)
        A_df = pd.DataFrame(A)
        if self._AUXILIARY_CELL_COST is not None:
            A_df.columns = self.terminal_cells + ['auxiliary_terminal']
        else:
            A_df.columns = self.terminal_cells

        absorption_probabilities = pd.DataFrame()

        for fate_label in fate_groups:
            absorption_probabilities[fate_label] = A_df.loc[:,fate_groups[fate_label]].sum(axis=1)

        if self._AUXILIARY_CELL_COST is not None:
            absorption_probabilities['Fate unknown'] = A_df.loc[:,'auxiliary_terminal'].values

        if self._AUXILIARY_CELL_COST is not None:
            absorption_probabilities.index = self.initial_cells+['auxiliary_initial']+self.intermediate_cells+['auxiliary_intermediate']+self.terminal_cells + ['auxiliary_terminal'] 
        else: 
            absorption_probabilities.index = self.initial_cells+self.intermediate_cells+self.terminal_cells
            
        return absorption_probabilities




    def proximal_sinkhorn(self, epsilon_threshold : float = None, patience : int = 100, verbose : bool = False):
        """ Decrease the model's effective regularization parameter (epsilon) via a proximal sinkhorn scheme.

        Input parameters:
            --------
            epsilon_threshold : float
                                value of the desired epsilon parameter (float); the algorithm will cease once this threshold is reached.
            patience          : int
                                Number of times that the progress should be monitored during the Sinkhorn iterations.

        """
        self.__prior_utility_variables = [copy.deepcopy(self.utility_variables)] 

 
        np.seterr(all='ignore')


        start_time = time.time()
        iterations = 0

        if self._PROXIMAL_EPSILON is None:
            self._PROXIMAL_EPSILON = self._EPSILON

            self._PROXIMAL_EPSILON_HISTORY.append(self._PROXIMAL_EPSILON) 
            self._PROXIMAL_EPSILON_HISTORY.append(self._PROXIMAL_EPSILON) # First component corresponds to iteration 0, second after one "standard" Sinkhorn solution.
            self._TOTAL_EPSILON_HISTORY.append(self._EPSILON)
            self._TOTAL_EPSILON_HISTORY.append(self._EPSILON)             # First component corresponds to iteration 0, second after one "standard" Sinkhorn solution.

            self._INNER_ITERATIONS = [0,patience*(len(self.history['infeasibility']))]
            self._OUTER_PROXIMAL_ITERATIONS = 0 


        first_epsilon = self._EPSILON

        r = 0
        while r < 1:
            
            self.fit(self.data, 
                                    prior     = self.transport_plans, 
                                    verbose   = verbose,
                                    patience  = patience,
                                    tolerance = self._TOLERANCE,
                            )
            
            self.__prior_utility_variables.append(copy.deepcopy(self.utility_variables))
            

            elapsed_time = time.time() - start_time
 
            total_epsilon = self._EPSILON
            self._TOTAL_EPSILON_HISTORY.append(total_epsilon)
            
            self._INNER_ITERATIONS.append(patience*(len(self.history['infeasibility'])))

            current_epsilon = self._PROXIMAL_EPSILON
            self._PROXIMAL_EPSILON_HISTORY.append(current_epsilon)
            
            r = (total_epsilon - first_epsilon)/(epsilon_threshold - first_epsilon)

                
            self._OUTER_PROXIMAL_ITERATIONS += 1
            iterations += 1
            
            print("\r", "[Proximal Sinkhorn] Outer iterations: {iter}".format(iter=iterations) + " | Initial epsilon: {0:.4e}".format(first_epsilon)+ " | Current epsilon: {0:.4e}".format(current_epsilon) + " | Total epsilon: {0:.4e}".format(total_epsilon) + " | Elapsed time: {time}".format(time=timedelta(seconds=elapsed_time)), end = "", flush=True)
        
        self.history['proximal_sinkhorn'] = {'iterations' : self._INNER_ITERATIONS, 'proximal_epsilon' : self._PROXIMAL_EPSILON_HISTORY, 'total_epsilon' : self._TOTAL_EPSILON_HISTORY}   

        print("\n Terminating proximal scheme.", flush=True)


        return


