import sympy as sy
import numpy as np
import math
import time
import random as rnd
import memory_profiler as mem_profile
import matplotlib.pyplot as plt
from decimal import *
from parameters import nn, pow_k, BO_num_centers
import Beq
import Func
import copy
import BO_charges
from sklearn.gaussian_process import GaussianProcessRegressor


in_mem = 'Memory (Before): {}'.format(mem_profile.memory_usage())
start_time = time.time()

# Open txt file
f_complete = open("results_complete.txt", "a")

# var_pos: list of coordinates used as genes in the EA
variab_pos = [3, 6, 7] + [10, 11, 12]
non_variab_pos = [i for i in range(3*nn)]
for i in range(len(variab_pos)):
    non_variab_pos.remove(variab_pos[i])

# Hyper parameters of the EA, tuned for a 5 centers configuration
pop_size = 2000  # Size of the population
offspring_per_gen = 50  # Number of offspring per generation
generations = 8000  # Number of generations
update_generation = 666  # After how many generation strategy parameter is updated
prec_pos = 4  # Inverse step-size of the uniform distrubution of the positions
dof = len(variab_pos)  # Number of degrees of freedom
tau = 1/sy.sqrt(3*dof)  # Defined in Section 3.2
tau_prime = 1/sy.sqrt(2*sy.sqrt(3*dof))  # Defined in Section 3.2
pow = 4  # :pow: = - log_10 (var_pos)

# Set the method for the strategy parameter update:
# True: with sigma_variance
# False: random update
update_sigma_variance = True

# Computational limits of the algorithm: the program ends after the minimum
# between: n_random_sol iteration, time_computational_limit seconds
n_random_sol = 200
time_computational_limit = 50000

# Hyper parameters of BO algorithm
BO_random_search_size = 200  # Number of iterations of random search
BO_tot_iterations = 200  # Computational limit of iterations of BO
minimum_charge = 150  # Minimum value of global charges Q_1 and Q_5

# Hyperparamters random update:
# The strategy parameter of :percentage_random_update:% of population is
# divided by :factor_random_update:
percentage_random_update = 80
factor_random_update = 3

# Decimal precision of the rounded fluxes
k_rounding = 5
# Use 6 by default, unless the program tells you to increase it
prec_temp = 6
# Decimal precision of positions
precision = 13


# Print parameters
print('PARAMETERS:')
print('pop_size=', pop_size,
      ', offspring_per_gen=', offspring_per_gen,
      ', n_random_sol=', n_random_sol,
      ', generations=', generations,
      ', update_generation=', update_generation,
      ', prec_pos=', prec_pos,
      ', pow_k=', pow_k,
      ', BO_random_search_size=', BO_random_search_size,
      ', BO_tot_iterations=', BO_tot_iterations,
      ', minimum_charge=', minimum_charge)

print('PARAMETERS:', file=f_complete)
print('pop_size=', pop_size,
      ', offspring_per_gen=', offspring_per_gen,
      ', n_random_sol=', n_random_sol,
      ', generations=', generations,
      ', update_generation=', update_generation,
      ', prec_pos=', prec_pos,
      ', pow_k=', pow_k,
      ', BO_random_search_size=', BO_random_search_size,
      ', BO_tot_iterations=', BO_tot_iterations,
      ', minimum_charge=', minimum_charge, file=f_complete)


def objective(k_list):
    '''
    This function takes a list of fluxes as input, solves the homogeneous
    bubble equation, rounds the fluxes with a precision of k_rounding and returns
    the minimum between Q_1 and Q_5 for that approximate solution
    
    k_list = list of fluxes with i=0,1 of the first BO_num_centers centers +
            the i=2 flux of the 1st center
    
        '''
    asympt_charges_pre = copy.copy(asympt_charges_pre1)
    bubbleeqm = copy.copy(bubbleeqm_pre)
 
    kval = [k_list[i] for i in range(2 * BO_num_centers)]
    k02val = k_list[-1]
    
    bubbleeqm = bubbleeqm.subs({Beq.k[0, 2]: k02val})
    for a in range(BO_num_centers):
        bubbleeqm = bubbleeqm.subs({Beq.k[a, 0]: kval[2 * a], Beq.k[a, 1]: kval[2 * a + 1]})
    bubbleeqm = bubbleeqm.subs({Beq.mu: 1, Beq.omog_var: 0})
    
    # Solve bubble equation
    varlist = []
    for a in range(nn-1):
        varlist.append(Beq.k[a+1, 2])
    temp = []
    for a in range(nn-1):
        temp.append(bubbleeqm.evalf(30))
    soldic = sy.solve(temp, tuple(varlist))
    
    # Produce a list with rounded solution
    k2_rounded = []
    for key in soldic:
        with Func.Capturing() as k2output:
            print(soldic[key].evalf(k_rounding + prec_temp))
            print(k2output, k_rounding)
        k2_rounded.append(sy.Rational(*Func.float_to_rat(k2output, k_rounding)))
    if len(k2_rounded) == nn-1:
        for key in ("Q0", "Q1"):
            asympt_charges_pre[key] = asympt_charges_pre[key].subs(Beq.k[0, 2], k02val)
            for a in range(BO_num_centers):
                asympt_charges_pre[key] = asympt_charges_pre[key].subs({Beq.k[a, 0]: kval[2 * a], Beq.k[a, 1]: kval[2 * a + 1]})
            for a in range(nn-1):
                asympt_charges_pre[key] = asympt_charges_pre[key].subs(Beq.k[a+1, 2], k2_rounded[a])
            
    else:
        for key in ("Q0", "Q1"):
            asympt_charges_pre[key] = 0
    return min([asympt_charges_pre[key] for key in ("Q0", "Q1")]), k2_rounded


class Individual:
    
    def __init__(self, var_pos, sigma_init_pop, is_offspring=False):
        '''
        Generates an individual adding gaussian noise (controlled by
        \sigma=sigma_init_pop) to the approximate solution

        Parameters
        ----------
        var_pos : List of coordinates that are genes in the EA
        sigma_init_pop : Sigma of the gaussian noise implemented for new
                        individuals
        is_offspring : True if the individual generated is an offspring
                       (in which case the attributes are implemented in the
                        offspring_gen function of the class Population), False
                       if it is an individual of the initial population

        '''
        if is_offspring is False:
            pos_temp = [pcentval[0][0], pcentval[0][1], pcentval[0][2]] +\
                       [pcentval[1][0] + sy.Rational(rnd.gauss(0, 10**precision), 10**precision)*sigma_init_pop[3],
                        pcentval[1][1], pcentval[1][2]] +\
                       [pcentval[2][0] + sy.Rational(rnd.gauss(0, 10**precision), 10**precision)*sigma_init_pop[6],
                        pcentval[2][1] + sy.Rational(rnd.gauss(0, 10**precision), 10**precision)*sigma_init_pop[7],
                        pcentval[2][2]] +\
                       [(pcentval[a][i] + sy.Rational(rnd.gauss(0, 10**precision), 10**precision)*sigma_init_pop[3*a+i])
                        for a in range(3, nn) for i in range(3)]
            
            # Implement opposition based method
            pos_temp_opp = []
            for a in range(nn):
                for i in range(3):
                    pos_temp_opp.append(-pos_temp[3*a+i]+2*pcentval[a][i])
            # Compute fitnesses
            fit_temp = fit_functions(*[pos_temp[i] for i in variab_pos])
            inv_fitness = 0
            for a in range(nn-1):
                inv_fitness = inv_fitness + abs(fit_temp[a])
            fit_temp_opp = fit_functions(*[pos_temp_opp[i] for i in variab_pos])
            inv_fitness_opp = 0
            # Compare fitnesses of this solution and the opposite one
            for a in range(nn-1):
                inv_fitness_opp = inv_fitness_opp + abs(fit_temp_opp[a])
            if inv_fitness < inv_fitness_opp:
                self.position = pos_temp
                self.inv_fitness = round(inv_fitness, 15)
                self.fitness = 1/(round(inv_fitness, 15) + 10**(-15))
            else:
                self.position = pos_temp_opp
                self.inv_fitness = round(inv_fitness_opp, 15)
                self.fitness = 1/(round(inv_fitness, 15) + 10**(-15))
                
            self.sigma = [0, 0, 0] + [sy.Rational(1, var_den), 0, 0] +\
                         [sy.Rational(1, var_den), sy.Rational(1, var_den), 0] +\
                         [sy.Rational(1, var_den) for _ in range(3*nn-9)]
        else:
            self.position = [pcentval[b//3][b%3] for b in range(3*nn)]
            self.inv_fitness = None
            self.fitness = None
            self.sigma = [0, 0, 0] + [sy.Rational(1, var_den), 0, 0] + [sy.Rational(1, var_den), sy.Rational(1, var_den), 0] +\
                         [sy.Rational(1, var_den) for _ in range(3*nn-9)]
     
    def evaluate_fitness(self):
        # Function to valuates fitness of an individual
        fit_temp = fit_functions(*[self.position[i] for i in variab_pos])
        inv_fitness = 0
        for a in range(nn-1):
            inv_fitness = inv_fitness + abs(fit_temp[a])
        self.inv_fitness = round(inv_fitness, 15)
        self.fitness = 1/(round(inv_fitness, 15) + 10**(-15))

    
    def __repr__(self):
        return "Fitness: {}".format(self.fitness)

        
        
        
class Population:

    pop_size_cl = pop_size
    
    def __init__(self, first_gen=True):
        '''
        ATTRIBUTES:
            pop: Array of position of the individuals in the population
            fitness: Array of fitnesses
            inv_fitness: Array of inverse fitnesses
            tot_fit: Sum of the fitnesses
            tot_inv_fit: Sum of the inverse fitnesses
            cum_fit, cum_inv_fit: cumulative probabilities for the Fitness
                                  proportional selection method
            cum_fit_scal, cum_inv_fit_scal: cumulative probabilities for sigma
                                            sigma scaling method

        '''
        if first_gen is True:
            self.pop = np.array([Individual(var_pos, sigma_init_pop) for a in range(self.pop_size_cl)])
            self.fitnesses = np.array([self.pop[ind].fitness for ind in range(self.pop_size_cl)])
            self.inv_fitnesses = np.array([self.pop[ind].inv_fitness for ind in range(self.pop_size_cl)])
            self.tot_fit = np.sum(self.fitnesses)
            self.tot_inv_fit = np.sum(self.inv_fitnesses)
            self.cum_fit = Population.cum_fit_func(self)[0]
            self.cum_inv_fit = Population.cum_fit_func(self)[1]
            self.cum_fit_scal = Population.cum_fit_scaling_func(self)[0]
            self.cum_inv_fit_scal = Population.cum_fit_scaling_func(self)[1]
    
    def cum_fit_func(self):
        '''
        Evaluates cumulative fitness (cum_fitness) and cumulative inverse
        fitness (cum_inv_fitness) for Fitness proportional selection method
        '''
        total_fitness = self.tot_fit
        total_inv_fitness = self.tot_inv_fit
        cum_fitness = np.cumsum(self.fitnesses)/total_fitness
        cum_inv_fitness = np.cumsum(self.inv_fitnesses)/total_inv_fitness
        return cum_fitness, cum_inv_fitness
    
    def cum_fit_scaling_func(self):
        '''
        Evaluates cumulative fitness (cum_fitness_scal) and cumulative inverse
        fitness (cum_inv_fitness_scal) for sigma scaling selection method
        '''
        fitnesses = self.fitnesses
        inv_fitnesses = self.inv_fitnesses
        mean = np.mean(fitnesses)
        sigma = sy.sqrt(np.var(fitnesses))
        mean_inv = np.mean(inv_fitnesses)
        sigma_inv = sy.sqrt(np.var(inv_fitnesses))
        list_of_fit_scal = []
        list_of_inv_fit_scal = []
        for ind in range(self.pop_size_cl):
            list_of_fit_scal.append(max((1+(fitnesses[ind]-mean)/2*sigma, 0)))
            list_of_inv_fit_scal.append(max((1+(inv_fitnesses[ind]-mean_inv)/2*sigma_inv, 0)))
        cum_fitness_scalpre = np.cumsum(list_of_fit_scal)
        cum_inv_fitness_scalpre = np.cumsum(list_of_inv_fit_scal)
        cum_fitness_scal = cum_fitness_scalpre/cum_fitness_scalpre[-1]
        cum_inv_fitness_scal = cum_inv_fitness_scalpre/cum_inv_fitness_scalpre[-1]
        return cum_fitness_scal, cum_inv_fitness_scal
    
    def best_fitness(self):
        # Finds the individual with highest fitness in the population
        # and returns its fitness and index
        best_fit = np.max(self.fitnesses)
        indx_best_fitness = np.argmax(self.fitnesses)
        return best_fit, indx_best_fitness
        
    def average_fitness(self):
        # Returns average fitness
        return self.tot_fit/self.pop_size_cl
     
    def average_position(self):
        # Computes the average position weighted by the fitness
        av_pos = [0 for _ in range(3*nn)]
        tot_fitness = self.tot_fit
        for ind in range(self.pop_size_cl):
            for _ in range(3*nn):
                av_pos[_] += self.pop[ind].position[_] * self.pop[ind].fitness
        for _ in range(3*nn):
            av_pos[_] = av_pos[_]/(tot_fitness)
        return av_pos
    
    def pos_variance(self):
        # Computes the standard deviation of the positions in the population
        # for updating the strategy parameter according to the Variance method
        posit = np.array([[self.pop[ind].position] for ind in range(self.pop_size_cl)])
        posit = posit.reshape((self.pop_size_cl, 3*nn))
        standard_dev = []
        for i in range(3*nn):
            standard_dev.append(math.sqrt(round(np.standard_dev(posit, axis=0)[i], 20)))
        return standard_dev

    def select_ind_parent(self, n_parents=2, scaling=False):
        '''
        This function selects n_parents number of parents for reproduction

        Parameters
        ----------
        parents : Number of parents for reproduction. The default is 2.
        scaling : Boole to implement scaling method in the selection process
                or not. The default is False.

        Returns: Index of the two selected parents
        
        '''
        parents_list = []
        if scaling is True:
            cum_fitness = copy.copy(self.cum_fit_scal)
        else:
            cum_fitness = copy.copy(self.cum_fit)
        for _ in range(n_parents):
            indx_par = 0
            prob_par = rnd.uniform(0, cum_fitness[-1])
            while cum_fitness[indx_par] < prob_par:
                indx_par += 1
            parents_list.append(indx_par)
            if indx_par == 0:
                diff = cum_fitness[indx_par]
            else:
                diff = cum_fitness[indx_par] - cum_fitness[indx_par-1]
            for i in range(indx_par, self.pop_size_cl):
                cum_fitness[i] -= diff
        return parents_list[0], parents_list[1]
      
    def select_ind_die(self, offspring_per_gen, scaling=False):
        '''
        This function selects offspring_per_gen number of individuals
        that are replaced in the next generation

        Parameters
        ----------
        offspring_per_gen : Number of individuals that die per generation
        scaling : Boole to implement scaling method in the selection process
                or not. The default is False.

        Returns: List with indices of the selected individuals
        
        '''
        dies_list = []
        if scaling is True:
            cum_inv_fitness = copy.copy(self.cum_inv_fit_scal)
        else:
            cum_inv_fitness = copy.copy(self.cum_inv_fit)
        for _ in range(offspring_per_gen):
            indx_die = 0
            prob_die = rnd.uniform(0, cum_inv_fitness[-1])
            while cum_inv_fitness[indx_die] < prob_die:
                indx_die += 1
            dies_list.append(indx_die)
            if indx_die == 0:
                diff = cum_inv_fitness[indx_die]
            else:
                diff = cum_inv_fitness[indx_die] - cum_inv_fitness[indx_die-1]
            for i in range(indx_die, self.pop_size_cl):
                cum_inv_fitness[i] -= diff
        return dies_list
    
    def offspring_gen(self):
        '''
        This function implements the reproduction and mutation process, running
        offspring_per_gen times. At each iteration, it calls the select_ind_parent
        function, makes the two individual reproduce and implements a mutation
        with a probability int(dof/2). It then replaces offspring_per_gen
        individuals selected with the function select_ind_die with the offspring.
        It updates the attributes of the population accordingly.

        Returns
        -------
        None.

        '''
        offspring_list = []
        for num_offspring in range(offspring_per_gen):
            indx1, indx2 = Population.select_ind_parent(self)
            offspring = Individual(var_pos, sigma_init_pop, is_offspring=True)
            mutat_parameter = sy.Rational(rnd.gauss(0, 10**precision), 10**precision)  # eq.(4.2)
            # Combine the parents to generate offspring
            for ind_pos in variab_pos:
                recomb_prob = rnd.randint(0, 2)
                if recomb_prob == 0:
                    offspring.position[ind_pos] = self.pop[indx1].position[ind_pos]
                    offspring.sigma[ind_pos] = self.pop[indx1].sigma[ind_pos]
                elif recomb_prob == 1:
                    offspring.position[ind_pos] = self.pop[indx2].position[ind_pos]
                    offspring.sigma[ind_pos] = self.pop[indx2].sigma[ind_pos]
                else:
                    offspring.position[ind_pos] = self.pop[indx1].position[ind_pos]/2 +\
                                                  self.pop[indx2].position[ind_pos]/2
                    offspring.sigma[ind_pos] = (self.pop[indx1].sigma[ind_pos]/2 +\
                                                self.pop[indx2].sigma[ind_pos]/2)
                # Implement mutation
                if rnd.randint(0, int(dof/2)) == 0:
                    mutat_parameter_i = sy.Rational(rnd.gauss(0, 10**precision), 10**precision)
                    mutat_parameter_j = sy.Rational(rnd.gauss(0, 10**precision), 10**precision)
                    temp_sigma = sy.exp(tau_prime*mutat_parameter +
                                        tau*mutat_parameter_i)*offspring.sigma[ind_pos]
                    
                    with Func.Capturing() as sigma_output:
                        print(temp_sigma.evalf(precision + prec_temp))
                    offspring.sigma[ind_pos] = sy.Rational(*Func.float_to_rat(sigma_output, precision))
                    offspring.position[ind_pos] = offspring.position[ind_pos] +\
                                                     offspring.sigma[ind_pos] * mutat_parameter_j
            # Compute offspring fitness
            Individual.evaluate_fitness(offspring)
            offspring_list.append(offspring)
        # Replace the died individuals with the offspring
        die_list = Population.select_ind_die(self, offspring_per_gen, False)
        # Update the attributes of the population
        diff_fit = 0
        diff_inv_fit = 0
        for num_offspring in range(offspring_per_gen):
            diff_fit += offspring_list[num_offspring].fitness - self.fitnesses[die_list[num_offspring]]
            diff_inv_fit += offspring_list[num_offspring].inv_fitness - self.inv_fitnesses[die_list[num_offspring]]
            self.fitnesses[die_list[num_offspring]] = offspring_list[num_offspring].fitness
            self.inv_fitnesses[die_list[num_offspring]] = offspring_list[num_offspring].inv_fitness
            self.pop[die_list[num_offspring]] = offspring_list[num_offspring]
        self.tot_fit += diff_fit
        self.tot_inv_fit += diff_inv_fit
        self.cum_fit = Population.cum_fit_func(self)[0]
        self.cum_inv_fit = Population.cum_fit_func(self)[1]
        return
            

# For cycle corresponding to the different random solutions
for solution_number in range(n_random_sol):
    
    # Set computational limit
    if time.time()-start_time > time_computational_limit:
        break
    
    # Assign values to the initial parameters and positions. We impose k^0,1_a
    # to have the same sign
    qval = [Decimal(1), Decimal(1), -Decimal(1), Decimal(-1), Decimal(1)]
    lzval = [Decimal(1) ,Decimal(1), Decimal(1)] 
    muval = Decimal(1) / Decimal(10**(5))
    pcentval = [[0, 0, 0],
                [sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos)), 0, 0],
                [sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos)),
                 sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos)), 0]] +\
               [[sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos)),
                 sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos)),
                 sy.Rational(rnd.randint(-10**prec_pos, 10**prec_pos), 10**(prec_pos))] for i in range(nn-3)]
    kval = []
    for a in range(0, nn - BO_num_centers):
        if rnd.randint(0, 2) == 0:
            kval.append([Decimal(rnd.randint(-10**pow_k, 0)), 
                         Decimal(rnd.randint(-10**pow_k, 0))])
        else:
            kval.append([Decimal(rnd.randint(1, 10**pow_k)), 
                         Decimal(rnd.randint(1, 10**pow_k))])
    print('kval: ', kval)
    
    # Import Bubble eqs. from Beq and transform it into sy.Matrix
    bubbleeqm_pre = sy.Matrix(Beq.bubbleeq)
    
    # Substitute value of the parameters in bubble equation
    bubbleeqm_pre = bubbleeqm_pre.subs({Beq.lz[i]: lzval[i] for i in range(3)})
    for a in range(nn):
        bubbleeqm_pre = bubbleeqm_pre.subs(Beq.q[a], qval[a])
        for i in range(3):
            bubbleeqm_pre = bubbleeqm_pre.subs(Beq.pcent[a, i], pcentval[a][i])

    for a in range(0, nn-BO_num_centers):
        for i in range(0, 2):
            bubbleeqm_pre = bubbleeqm_pre.subs(Beq.k[a + BO_num_centers, i], kval[a][i])
            
    # Import charges eqs. from Beq
    asympt_charges_pre1 = copy.copy(Beq.asympt_charges)
    
    # Substitute values in charges
    for key in ("Q0", "Q1"):
        for a in range(nn):
            asympt_charges_pre1[key] = asympt_charges_pre1[key].subs(Beq.q[a], qval[a])
        for a in range(0, nn-BO_num_centers):
            for i in range(0, 2):
                asympt_charges_pre1[key] = asympt_charges_pre1[key].subs(Beq.k[a+BO_num_centers, i], kval[a][i])
    
    
    
    ''' BAYESIAN OPTIMIZATION ALGORITHM
        Search for  fluxes of first :BO_num_centers: number of centers
        such that both the charges are greater than :minimum_charge:.
        If, after :BO_tot_iterations: iterations of the BO, such a value is
        not found it continues the for cycle, generating new initial
        parameters and positions.
    '''
    # Sample the domain of the objective function
    t0_BO_random = time.time()
    X = np.random.randint(-10**pow_k, 10**pow_k, (BO_random_search_size, 1 + 2 * BO_num_centers))
    y = []
    for i in range(np.shape(X)[0]):
        y_i, _ = objective(X[i])
        y.append(y_i)
    y = np.array(y)
    print('BO random search takes ', time.time()-t0_BO_random, ' s')
    # Reshape into rows and cols
    y = y.reshape(len(y), 1)
    # Define and fit the model
    model = GaussianProcessRegressor()
    model.fit(X, y)
    # Perform the optimization process
    x_list = []
    BO_iteration = 0
    while True:
        BO_iteration += 1
        # Select the next point to sample
        x = BO_charges.opt_acquisition(X, y, model)
        # Sample the point
        actual = objective(x)[0]
        x = x.reshape(1 + 2 * BO_num_centers, 1)
        # Summarize the finding
        est, _ = BO_charges.surrogate(model, np.array(x.T))
        print('x= ', x, 'f()= ', est, 'actual= ', actual)
        x_list.append(x)
        # Add the data to the dataset
        X = np.vstack((X, np.array(x.T)))
        y = np.vstack((y, [[actual]]))
        # Update the model
        model.fit(X, y)
        ix = np.argmax(y)
        if y[ix] > minimum_charge or BO_iteration == BO_tot_iterations:
            break

    # Best result: if it is less than :minimum_charge: continue to the next random solution
    ix = np.argmax(y)
    if y[ix] < minimum_charge:
        continue
    print('Best result at: ', X[ix], 'with value: ', y[ix])
    
    # Given initial parameters such that Q1,Q5>:minimum_charge:, solve the 
    # homogeneous bubble equation to obtain the 2-fluxes
    _, k2_rounded = objective(X[ix])
    print('k2_rounded: ', k2_rounded)
    
    # Collect the fluxes into kval
    kval = [[X[ix][2*a], X[ix][2*a + 1]] for a in range(BO_num_centers)] + kval
    
    
    '''
    We now define the fitness function importing the bubble equations from Beq.
    Substitute values of the local charges, set :Beq.omog_var:=1 to make
    the equation non-homogeneous, and substitute the coordinates that are not
    genes of the EA.
    '''
    bubbleeqm_fit = sy.Matrix(Beq.bubbleeq)
    
    bubbleeqm_fit = bubbleeqm_fit.subs({Beq.lz[i]: lzval[i] for i in range(3)})
    for a in range(nn):
        bubbleeqm_fit = bubbleeqm_fit.subs(Beq.q[a], qval[a])

    for a in range(0, nn):
        for i in range(0, 2):
            bubbleeqm_fit = bubbleeqm_fit.subs(Beq.k[a, i], kval[a][i])
    
    bubbleeqm_fit = bubbleeqm_fit.subs({Beq.mu: muval, Beq.omog_var: 1})
    bubbleeqm_fit = bubbleeqm_fit.subs({Beq.k[0, 2]: X[ix][-1]})
    for a in range(nn-1):
            bubbleeqm_fit=bubbleeqm_fit.subs(Beq.k[a+1, 2], k2_rounded[a])
    

    for i in non_variab_pos:
        bubbleeqm_fit=bubbleeqm_fit.subs(Beq.pcent[i//3, i%3], pcentval[i//3][i%3])
    
    
    # We lambdify the fitness function (which is actually the inverse fitness
    # function) being the Bubble equations
    variables = []
    for i in variab_pos:
        variables.append(Beq.pcent[i//3, i%3])
    fit_functions = sy.lambdify(variables, list(bubbleeqm_fit), modules="math")
    

    # Check necessary condition for AdS_2 equation
    nec_cond_list = copy.copy(Beq.nec_cond)
    for j in range(nn-1):
        for a in range(nn):
            nec_cond_list[j] = nec_cond_list[j].subs(Beq.q[a], qval[a])
            for i in range(2):
                nec_cond_list[j] = nec_cond_list[j].subs(Beq.k[a, i], kval[a][i])
        nec_cond_list[j] = nec_cond_list[j].subs({Beq.k[0, 2]: X[ix][-1]})
        for a in range(nn-1):
            nec_cond_list[j] = nec_cond_list[j].subs(Beq.k[a+1, 2], k2_rounded[a])
    print('nec_cond_list: ', nec_cond_list)
    if all(conditions > 0 for conditions in nec_cond_list) or all(conditions < 0 for conditions in nec_cond_list):
        print('NECESSARY CONDITION NOT SATISFIED')
        continue
    
    # Import M, substitute values of the parameters and check it is positive definite
    M = sy.Matrix(Beq.Mbar + Beq.mu * Beq.Mdot)
    
    M = M.subs({Beq.lz[i]: lzval[i] for i in range(3)})
    for a in range(nn):
        M = M.subs(Beq.q[a], qval[a])

    for a in range(0, nn):
        for i in range(0, 2):
            M = M.subs(Beq.k[a, i], kval[a][i])
    
    M = M.subs({Beq.mu: muval, Beq.omog_var: 1})
    M = M.subs({Beq.k[0, 2]: X[ix][-1]})
    for a in range(nn-1):
        M = M.subs(Beq.k[a+1, 2], k2_rounded[a])
    
    for i in range(3*nn):
        M = M.subs(Beq.pcent[i//3, i%3], pcentval[i//3][i%3])
    
    eig_M_dict = M.eigenvals()
    print("eig_M: ", eig_M_dict)
    eig_M = list(eig_M_dict.keys())
    if all(eigenvalues > 0 for eigenvalues in eig_M):
        print('All eig positive')
    else:
        continue
    
    # Export in results_complete.txt initial values of approximate solution
    print('', file=f_complete)
    print('', file=f_complete)
    print('', file=f_complete)
    print('', file=f_complete)
    print('Solution number: ', solution_number, file=f_complete)
    print('Initial position: ', pcentval, file=f_complete)
    print('1,2 fluxes of centers: ', kval, file=f_complete)
    print('3rd flux of centers: ', [X[ix][-1]] + k2_rounded, file=f_complete)


    # Initialize :var_pos: and sigma_init_pop
    print('Parameters: update_sigma_variance==', update_sigma_variance, ' var_den=', 10**pow)
    print('', file=f_complete)
    print('', file=f_complete)
    print('Parameters: update_sigma_variance==', update_sigma_variance, ' var_den=', 10**pow, file=f_complete)
    var_den = 10**(pow)
    var_pos = [0, 0, 0] + [sy.Rational(1, var_den), 0, 0] + [sy.Rational(1, var_den), sy.Rational(1, var_den), 0] +\
              [sy.Rational(1, var_den) for _ in range(3*nn-9)]
    
    sigma_init_pop_den = 10**(pow)
    sigma_init_pop = [0, 0, 0] + [sy.Rational(1, sigma_init_pop_den), 0, 0] + [sy.Rational(1, sigma_init_pop_den), sy.Rational(1, sigma_init_pop_den), 0] +\
                     [sy.Rational(1, sigma_init_pop_den) for _ in range(3*nn-9)]
             
    # Compute the fitness of the seed solution
    fit_rounded_sol = bubbleeqm_fit
    print("Fitness equation:", fit_rounded_sol)
    print("Fitness equation:", fit_rounded_sol, file=f_complete)
    
    for i in variab_pos:
        fit_rounded_sol = fit_rounded_sol.subs(Beq.pcent[i//3, i%3], pcentval[i//3][i%3])
    inv_fitness_rounded = 0
    for a in range(nn-1):
        inv_fitness_rounded = inv_fitness_rounded + abs(fit_rounded_sol[a])
    print('Fitness of rounded sol: {}'.format(1/inv_fitness_rounded.round(5)))
    print('Fitness of rounded sol: {}'.format(1/inv_fitness_rounded.round(5)), file=f_complete)
    
    
    '''
    EA algorithm. Initialize population, and run over :generations: generations.
    '''
    # Initialize population
    Population.pop_size_cl = pop_size
    population = Population()
    position_variance = Population.pos_variance(population)
    print('Position variance: ', position_variance)
    print('Position variance: ', position_variance, file=f_complete)
    
    # Define lists to collect the best fitness per generation and the average
    # fitness per generation
    history_best = []
    history_av = []
   
    best_fit_population = None  # Variable to record the best fitness in the population
    best_fitness_ever = 0  # Variable to record best fitness ever found
    
    '''
    :solution_to_be_recorded: is a boolean variable initialized to False, if
    fitness of a solution is higher than 10^6 it is turned to True and exports
    a plot of the EA process solution_to_be_recorded = False
    '''
    solution_to_be_recorded = False
    
    # For cycle over generations
    for gen in range(generations):

        # Run the reproduction and mutation function
        Population.offspring_gen(population)
        
        ''' Every time the fittest individual in the population changes
            its fitness and position are printed
            '''
        if best_fit_population != population.pop[Population.best_fitness(population)[1]].position:
            best_fit_population = population.pop[Population.best_fitness(population)[1]].position
            print('Fitness best: ', Population.best_fitness(population)[0])
            print('Fitness best: ', Population.best_fitness(population)[0], file=f_complete)
            print('Position best: ', [round(population.pop[Population.best_fitness(population)[1]].position[i], 13) for i in range(3*nn)])
            print('Position best: ', [round(population.pop[Population.best_fitness(population)[1]].position[i], 13) for i in range(3*nn)], file=f_complete)
        
        # Record the fittest individual ever obtained
        if Population.best_fitness(population)[0] > best_fitness_ever:
            best_fit_ever = Population.best_fitness(population)[0]
            best_pos_ever = population.pop[Population.best_fitness(population)[1]].position
        history_best.append(Population.best_fitness(population)[0])
        history_av.append(Population.average_fitness(population))
        
        # :solution_to_be_recorded: = True if fitness > 10^6
        if Population.best_fitness(population)[0] > 10**6:
            solution_to_be_recorded = True
        
        # Update sigma of the population
        if gen % update_generation == update_generation-1:
            print('Current generation: ', gen)
            print('Current generation: ', gen, file=f_complete)
            position_variance = Population.pos_variance(population)
            print('Position variance: ', position_variance)
            print('Position variance: ', position_variance, file=f_complete)
            if update_sigma_variance is True:
                '''Update sigma according to the variance in position
                   of the population'''
                new_sigma = Population.pos_variance(population)
                for ind in range(pop_size):
                    for i in variab_pos:
                        if population.pop[ind].sigma[i] > 10**(-10):
                            population.pop[ind].sigma[i] = new_sigma[i]*10
                        else:
                            population.pop[ind].sigma[i] = sigma_init_pop[i]
            else:
                '''Select randomly (100-:percentage_random_update:)/100% of the
                population whose sigma is multiplied by :factor_random_update:,
                the sigma of the remaining individuals is divided by :factor_random_update:'''
                not_update_sigma = rnd.sample(range(pop_size), int(pop_size*(100-percentage_random_update)/100))
                for ind in range(pop_size):
                    for pos in variab_pos:
                        if ind not in not_update_sigma:
                            population.pop[ind].sigma[pos] = population.pop[ind].sigma[pos].evalf(3)/factor_random_update
                        else:
                            population.pop[ind].sigma[pos] = population.pop[ind].sigma[pos].evalf(3)*factor_random_update
            print("Sigma of the most fit individual: ", population.pop[Population.best_fitness(population)[1]].sigma)
            print("Sigma of the most fit individual: ", population.pop[Population.best_fitness(population)[1]].sigma, file=f_complete)
    
    
    # Generate and save plots
    plt.plot(history_best)
    if solution_to_be_recorded is True:
        plt.savefig('history_best_{}_{}_{}.png'.format(solution_number, update_sigma_variance, 10**pow))
    plt.show()
    
    history_best_log = []
    for i in range(len(history_best)):
        history_best_log.append(np.log(float(history_best[i])))
    plt.plot(history_best_log)
    if solution_to_be_recorded is True:
        plt.savefig('history_best_log_{}_{}_{}.png'.format(solution_number, update_sigma_variance, 10**pow))
    plt.show()
        
    plt.plot(history_av)
    if solution_to_be_recorded is True:
        plt.savefig('history_best_av_{}_{}_{}.png'.format(solution_number, update_sigma_variance, 10**pow))
    plt.show()
    
    position_best_fin = [i * float(muval) for i in best_pos_ever]
    
    # Print the initial position, the best position found and the difference of the two
    in_position = []
    for a in range(3):
        for i in range(nn):
            in_position.append(round(pcentval[i][a], 13))
    print('Initial position: ', in_position)
    print('Initial position: ', in_position, file=f_complete)
    print('Best position ever: ', best_pos_ever)
    print('Best position ever: ', best_pos_ever, file=f_complete)

    # Compute asymptotic charges of the best solution found
    asympt_charges_sol = copy.copy(Beq.asympt_charges)
    for key in asympt_charges_sol:
        asympt_charges_sol[key] = asympt_charges_sol[key].subs(Beq.k[0, 2], X[ix][-1])
        for a in range(nn):
            asympt_charges_sol[key] = asympt_charges_sol[key].subs(Beq.q[a], qval[a])
            if a != nn-1:
                asympt_charges_sol[key] = asympt_charges_sol[key].subs(Beq.k[a+1, 2], k2_rounded[a])
            for i in range(3):
                asympt_charges_sol[key] = asympt_charges_sol[key].subs(Beq.pcent[a, i], position_best_fin[3*a+i])
                if i != 2:
                    asympt_charges_sol[key] = asympt_charges_sol[key].subs(Beq.k[a, i], kval[a][i])
    print(asympt_charges_sol)
    print(asympt_charges_sol, file=f_complete)
    JL = math.sqrt(asympt_charges_sol["JL1"]**2 + asympt_charges_sol["JL2"]**2 + asympt_charges_sol["JL3"]**2)
    print("JL: ", JL)
    print("JL: ", JL, file=f_complete)
    
       



fin_mem = 'Memory (After): {}'.format(mem_profile.memory_usage())
print(in_mem)
print(in_mem, file=f_complete)
print(fin_mem)
print(fin_mem, file=f_complete)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds ---" % (time.time() - start_time), file=f_complete)

f_complete.close()
