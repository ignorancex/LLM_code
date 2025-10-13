import pandas as pd
import sklearn.linear_model
from sksurv.nonparametric import kaplan_meier_estimator
'''
Multi-period inventory management
Hector Perez, Christian Hubbs, Owais Sarwar
4/14/2020
'''

import gym
import itertools
import numpy as np
from scipy.stats import *
import scipy.stats
from or_gym.utils import assign_env_config
from collections import deque
import random
from sklearn.metrics import pairwise_kernels
import pandas as pd
import numpy as np
import itertools
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

class InvManagementMasterEnv(gym.Env):
    '''
    The supply chain environment is structured as follows:

    It is a multi-period multi-echelon production-inventory system for a single non-perishable product that is sold only
    in discrete quantities. Each stage in the supply chain consists of an inventory holding area and a production area.
    The exception are the first stage (retailer: only inventory area) and the last stage (raw material transformation
    plant: only production area, with unlimited raw material availability). The inventory holding area holds the inventory
    necessary to produce the material at that stage. One unit of inventory produces one unit of product at each stage.
    There are lead times between the transfer of material from one stage to the next. The outgoing material from stage i
    is the feed material for production at stage i-1. Stages are numbered in ascending order: Stages = {0, 1, ..., M}
    (i.e. m = 0 is the retailer). Production at each stage is bounded by the stage's production capacity and the available
    inventory.

    At the beginning of each time period, the following sequence of events occurs:

    0) Stages 0 through M-1 place replenishment orders to their respective suppliers. Replenishment orders are filled
        according to available production capacity and available inventory at the respective suppliers.
    1) Stages 0 through M-1 receive incoming inventory replenishment shipments that have made it down the product pipeline
        after the stage's respective lead time.
    2) Customer demand occurs at stage 0 (retailer). It is sampled from a specified discrete probability distribution.
    3) Demand is filled according to available inventory at stage 0.
    4) Option: one of the following occurs,
        a) Unfulfilled sales and replenishment orders are backlogged at a penalty.
            Note: Backlogged sales take priority in the following period.
        b) Unfulfilled sales and replenishment orders are lost with a goodwill loss penalty.
    5) Surpluss inventory is held at each stage at a holding cost.

    '''
    def __init__(self, *args, **kwargs):
        '''
        periods = [positive integer] number of periods in simulation.
        I0 = [non-negative integer; dimension |Stages|-1] initial inventories for each stage.
        p = [positive float] unit price for final product.
        r = [non-negative float; dimension |Stages|] unit cost for replenishment orders at each stage.
        k = [non-negative float; dimension |Stages|] backlog cost or goodwill loss (per unit) for unfulfilled orders (demand or replenishment orders).
        h = [non-negative float; dimension |Stages|-1] unit holding cost for excess on-hand inventory at each stage.
            (Note: does not include pipeline inventory).
        c = [positive integer; dimension |Stages|-1] production capacities for each suppliers (stages 1 through |Stage|).
        L = [non-negative integer; dimension |Stages|-1] lead times in betwen stages.
        backlog = [boolean] are unfulfilled orders backlogged? True = backlogged, False = lost sales.
        dist = [integer] value between 1 and 4. Specifies distribution for customer demand.
            1: poisson distribution
            2: binomial distribution
            3: uniform random integer
            4: geometric distribution
            5: user supplied demand values
        dist_param = [dictionary] named values for parameters fed to statistical distribution.
            poisson: {'mu': <mean value>}
            binom: {'n': <mean value>, 'p': <probability between 0 and 1 of getting the mean value>}
            raindint: {'low' = <lower bound>, 'high': <upper bound>}
            geom: {'p': <probability. Outcome is the number of trials to success>}
        alpha = [float in range (0,1]] discount factor to account for the time value of money
        seed_int = [integer] seed for random state.
        user_D = [list] user specified demand for each time period in simulation
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)
        self.num_periods = 50
        self.init_inv = 15
        self.unit_price = 2
        self.unit_cost_ordering = 3
        self.demand_cost = 2
        self.holding_cost = 1
        self.supply_capacity = 15
        self.L = [0]
        self.num_stages=2
        self.dist = 1
        self.dist_param = {'mu': [5,5]
                           }
        self.discount = 0.99
        self.seed_int = 0

        self._max_rewards = 2000
        self.allowed_censoring = 1

        # Mapping of the pricing action
        self.custom_mapping = {0: 2, 1: 4, 2: 6}


        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)
        #  parameters
        #  dictionary with options for demand distributions


        # check inputs
        assert self.init_inv >=0, "The initial inventory cannot be negative"
        try:
            assert self.num_periods > 0, "The number of periods must be positive. Num Periods = {}".format(self.num_periods)
        except TypeError:
            print('\n{}\n'.format(self.num_periods))
        assert self.unit_price >= 0, "The sales prices cannot be negative."
        assert self.unit_cost_ordering >= 0, "The procurement costs cannot be negative."
        assert self.demand_cost >= 0, "The unfulfilled demand costs cannot be negative."
        assert self.holding_cost >= 0, "The inventory holding costs cannot be negative."
        assert self.supply_capacity > 0, "The supply capacities must be positive."
        assert (self.discount>0) & (self.discount<=1), "alpha must be in the range (0,1]."


        # intialize
        self.reset()


        self.action_space=gym.spaces.multi_discrete.MultiDiscrete(np.array([16,3]))
        self.observation_space = gym.spaces.Box(
             low=np.array([0,0,0,0,0,0],dtype=np.int32),
             high=np.array([25,1,30,1,1,15],dtype=np.int32))



    def _RESET(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        '''

        periods = self.num_periods



        # simulation result lists
        self.I=np.zeros(periods + 1)
        self.state_of_economy = np.zeros(periods+1)
        self.interest_rate = np.zeros(periods+1)

        self.R=np.zeros(periods) # replenishment order
        self.D=np.zeros(periods) # demand at retailer
        self.Sales=np.zeros(periods) # units sold
        self.LS=np.zeros(periods) # lost sales
        self.P=np.zeros(periods) # real profit
        self.P_censored=np.zeros(periods) #observed profit
        self.Indicator_censor=np.zeros(periods)


        # initializetion
        self.period = 0 # initialize time
        self.I[0]=self.init_inv # initial inventory

        self.state_log=[np.nan for _ in range(periods+1)]
        self.ordering_log = np.zeros(periods)
        self.price_log = np.zeros(periods)

        # set state
        self._update_state()

        return self.state

    def _update_state(self):

        t = self.period

        state = np.zeros(5)

        if self.state_of_economy[t-1]==1 or t==0:

            self.state_of_economy[t] = bernoulli.rvs(0.8,size=1)
        else:
            self.state_of_economy[t] = bernoulli.rvs(0.3,size=1)

        if t ==0:
            self.interest_rate[t] = norm.rvs(0,0.0001,size=1)
        else:
            self.interest_rate[t] = 0.25*self.interest_rate[t-1]+ norm.rvs(0,0.0001,size=1)



        if t == 0:
            state[0] = self.init_inv
        else:
            state[0] = self.I[t]

        if t == 0:
            state[1]=1
        else:
            state[1] = self.Indicator_censor[t-1]


        if self.Indicator_censor[t-1]==0:
            if t==0:
                state[2]=5  #start of the period there is no censoring and demand is the historical average
            else:
                state[2] = self.Sales[t-1]
        else:
            state[2] = self.D[t-1]

        state[3] = self.state_of_economy[t]

        state[4] = self.interest_rate[t]

        self.state = state.copy()
        self.state_log[t]=state



    def _STEP(self,action):
        '''
        Take a step in time in the multiperiod inventory management problem.
        action = [integer; dimension |Stages|-1] number of units to request from suppliers (last stage makes no requests)
        '''


        # Transform the second action
        action[1] = self.custom_mapping[action[1]]
        # get inventory at hand and pipeline inventory at beginning of the period
        n = self.period

        order=action[0]
        price=action[1]

        self.ordering_log[n] = order # save which ordering action taken
        self.price_log[n] = price # save which pricing action taken

        I = self.I[n].copy() # inventory at start of period n


        c = self.supply_capacity # maximum ordering capacity
        CurrentInventory  = self.I[n] + order # available inventory after order is made


        self.dist_param['mu'][0] = 3.5
        self.dist_param['mu'][1] = 1


        price_elasticity= -1.5
        economic_impact=5
        interest_rate_impact = -5

        adjusted_price_effect=price_elasticity*price
        adjusted_economy_effect = economic_impact*self.state_of_economy[n]
        adjusted_interest_effect = interest_rate_impact*self.interest_rate[n]
        lag_demand=1.2* self.D[n-1]
        random_disturbance=truncnorm.rvs(0,10,self.dist_param['mu'][0],self.dist_param['mu'][1])
        # demand is realized
        if n==0:
            Demand=max(round(5 +
                        adjusted_price_effect +
                        adjusted_economy_effect +
                        adjusted_interest_effect +
                        random_disturbance),0)
        else:
            Demand=max(round(lag_demand +
                        adjusted_price_effect +
                        adjusted_economy_effect +
                        adjusted_interest_effect +
                        random_disturbance),0)





        if Demand > CurrentInventory and n >=self.allowed_censoring and (self.Indicator_censor[n-self.allowed_censoring:n] == np.zeros(self.allowed_censoring)).all(): #if there is censoring in the current time and allowed censoring is already met then demand is current inventory
            Demand = CurrentInventory
        else:
            Demand = Demand
        self.D[n] = Demand # store D[n]
        Sales = min(CurrentInventory,Demand) # Demand at retailer
        self.Sales[n] = Sales # Save sales


        self.I[n+1] = CurrentInventory - Sales  # update inventory on hand
        LostSales = max(Demand - Sales ,0)# unfulfilled demand and replenishment orders
        self.LS[n] = LostSales # store Lost Sales

        # calculate true profit
        P = price*Sales - self.unit_cost_ordering*min(order,self.supply_capacity) - self.demand_cost*max(Demand - CurrentInventory,0) - self.holding_cost*max(CurrentInventory-Demand,0)
        # calculate observed profit
        if Demand > CurrentInventory:
            P_censored =price*Sales - self.unit_cost_ordering*min(order,self.supply_capacity) - self.holding_cost*max(CurrentInventory-Demand,0)
            self.Indicator_censor[n] = 0
        else:
            P_censored = price*Sales - self.unit_cost_ordering*min(order,self.supply_capacity) - self.demand_cost*max(Demand - CurrentInventory,0) - self.holding_cost*max(CurrentInventory-Demand,0)
            self.Indicator_censor[n] = 1


        self.P[n] = P # store true profit
        self.P_censored[n] = P_censored #stoe observed profit

        # update period
        self.period += 1

        # update stae
        self._update_state()



        # determine if simulation should terminate
        if self.period >= self.num_periods:
            done = True
        else:
            done = False

        return self.state,P_censored, P,Demand ,done

    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        return self.action_space.sample()



    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
    def data_generator(self,policy,episode):
        dic_data ={}
        for i in range(episode):
            self.reset()
            data = np.empty((self.num_periods,15))
            for j in range(self.num_periods):
                state = self.state
                if policy == None:
                    action = self.sample_action()
                else:
                    action= policy.predict(self.state)
                next_state,profit_c,profit,Demand, _  = self.step(action)

                data[j]=np.concatenate([state,action,np.array([profit_c]),np.array([profit]),
                                np.array([Demand]),next_state],axis=0).reshape(1,-1)

            dic_data[i] =pd.DataFrame(data,columns=["Inv_0",
                           "Censor_0",
                          "Sales_0",
                          "State_0",
                          "IRate_0",
                           "Ordering_0",
                           "Pricing_0",
                           "Observed Profit",
                            "Real Profit",
                           "Demand",
                          "Inv_1",
                          "Censor_1",
                          "Sales_1",
                           "State_1",
                          "IRate_1"])

        return pd.concat([dic_data[i] for i in range(episode)],axis=0).reset_index().drop("index",axis=1)

    def prepare_data(self,df):
        """
        Prepare the data by converting 'Next Censor' to boolean.
        """
        df = df.copy()
        df['Censor_1'] = df['Censor_1'].apply(lambda x: True if x==1 else False)
        return df

    def calculate_survival(self,df, x):
        """
        Calculate the conditional and integral survival probabilities.
        """
        level, survival_prob = kaplan_meier_estimator(df["Censor_1"], df["Sales_1"])
        survival_df = pd.DataFrame({'level': level, 'survival_prob': survival_prob})
        con_survival = survival_df.loc[survival_df['level'] == x, 'survival_prob'].values[0]
        integ_survival = survival_df.loc[survival_df['level'] >= x, 'survival_prob'].sum()
        return con_survival, integ_survival

    def estimate_demand(self,x, df):
        """
        Estimate demand based on Kaplan-Meier survival probabilities.

        Args:
        x (int): The current level or period for which to estimate demand.
        df (DataFrame): The DataFrame containing 'Next Censor' and 'Next Sales' data.

        Returns:
        float: Estimated demand value.
        """
        try:
            # Prepare the data
            prepared_df = self.prepare_data(df[['Censor_1', 'Sales_1']])

            # Calculate survival probabilities
            con_survival, integ_survival = self.calculate_survival(prepared_df, int(x))

            # Calculate demand
            estimated_demand = int(x) + (1 / con_survival) * integ_survival

            return estimated_demand

        except KeyError as e:
            print(f"Missing column in DataFrame: {e}")
            return None
        except IndexError as e:
            print(f"Data error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


    def impute_reward(self,df):
        df_1 = df.copy() # get rid of multiple index
        df_1["Estimated Demand"]= df_1["Demand"] # when demand is observed, estimated demand is observed demand
        df_1["Imputed Profit"]= df_1["Observed Profit"] # when demand is observed, imputed profit is observed profit
        indexes=df_1.loc[df_1["Censor_1"]==0].index.to_list()
        df_1.loc[indexes,"Estimated Demand"]=df_1.loc[indexes].apply(lambda df1: self.estimate_demand(df1["Sales_1"],df),axis=1)
        df_1.loc[indexes,"Imputed Profit"]=df_1.loc[indexes].apply(lambda df1: df1["Observed Profit"]-2*(df1["Estimated Demand"]-df1["Sales_1"]),axis=1)
        return df_1


def generate_partition(data, i):
    """
    Generates partition O^{(i)}_N for a given i.

    Parameters:
    data (pd.DataFrame): Input DataFrame with columns for Inventory, Censor, Sales, State of Economy,
                         Interest Rate, Ordering, Pricing, and Imputed Profit.
    i (int): The lag step to define partitions based on Delta (Censor) conditions.

    Returns:
    pd.DataFrame: The resulting partitioned DataFrame according to the given i.
    """
    # Ensure i is at least 1
    if i < 1:
        raise ValueError("i must be at least 1")

    # Add shifted columns for past and future values
    for j in range(-i, 2):  # from t-i to t+1
        data[f'Inv_{j}'] = data['Inv_0'].shift(-j)
        data[f'Censor_{j}'] = data['Censor_0'].shift(-j)
        data[f'Sales_{j}'] = data['Sales_0'].shift(-j)
        data[f'State_{j}'] = data['State_0'].shift(-j)
        data[f'IRate_{j}'] = data['IRate_0'].shift(-j)
        if j != 1:  # Ordering and Pricing are not needed for t+1
            data[f'Ordering_{j}'] = data['Ordering_0'].shift(-j)
            data[f'Pricing_{j}'] = data['Pricing_0'].shift(-j)

    # Create a filter for the DataFrame based on Censor conditions
    conditions = (data[f'Censor_0'] == 0)
    for j in range(1, i):
        conditions &= (data[f'Censor_{-j}'] == 0)
    conditions &= (data[f'Censor_{-i}'] == 1)

    # Filter data
    filtered_data = data[conditions]

    # Select and rename columns as needed
    columns = []
    for j in range(-i, 2):  # append columns for each time frame
        columns.extend([f'Inv_{j}', f'Censor_{j}', f'Sales_{j}', f'State_{j}', f'IRate_{j}'])
        if j != 1:  # Exclude Ordering and Pricing for t+1
            columns.extend([f'Ordering_{j}', f'Pricing_{j}'])
    columns.append('Imputed Profit')  # Add Imputed Profit only once at t

    return filtered_data[columns]
def expand_and_predict_max(df, ordering, pricing, model):
    """
    Expands the DataFrame by appending each combination of actions and ordering to each row, predicts using a model,
    and returns the maximum prediction for each original row group.

    Args:
    df (pd.DataFrame): Original DataFrame to expand.
    actions (list): List of action values to include in the grid.
    ordering (list): List of ordering indices to include in the grid.
    model: A machine learning model with a predict method.

    Returns:
    pd.DataFrame: DataFrame with original columns and the maximum predictions.
    """
    # Creating the grid as a DataFrame
    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
    df=df.reset_index()
    # Adding a temporary key for merging
    df['key'] = 1
    grid['key'] = 1
    # Change columns names

    # Perform a cross join
    expanded_df = pd.merge(df, grid, on='key').drop('key', axis=1)

    # Ensure the expanded DataFrame is formatted correctly for the model prediction
    # This step may need customization based on model requirements
    expanded_df.columns=["index"] + list(model.feature_names_in_)
    # Predicting
    predictions = model.predict(expanded_df.iloc[:,1:])
    expanded_df['predictions'] = predictions

    # Group by original identifiers and find the maximum prediction for each group
    max_predictions = expanded_df.groupby(["index"])['predictions'].max()

    return max_predictions
def generalized_stack(env, log_names, counter, n=1):
    reshaped_logs = []

    # Iterate over each time step from 'counter-n' to 'counter'
    for j in range(counter - n, counter + 1):
        for log_name in log_names:
            log = getattr(env, log_name)  # Access the log from the env object

            if j < counter:  # Process as usual until the last step
                if log_name == 'state_log':
                    # Since state_log returns a 2D array (5x1), flatten this to a single row
                    log_entry = log[j].reshape(1,-1)
                else:
                    # For scalar logs, wrap the scalar in an array and flatten
                    log_entry = np.array([log[j]]).reshape(1,-1)

                # Append the processed log entry
                reshaped_logs.append(log_entry)
            else:  # When j == counter, handle only state_log
                if log_name == 'state_log':
                    # Handle only state_log at the last index
                    log_entry = log[j].reshape(1,-1)  # Flatten to ensure it is a single row
                    reshaped_logs.append(log_entry)
                # If log_name is not 'state_log', skip appending for this iteration

    # Stack all the reshaped logs horizontally
    result = np.hstack(reshaped_logs)
    return pd.DataFrame(result).reset_index()
def action_map(x):
    x=x.reset_index(drop=True)
    if x.iloc[1]==2:
        return np.array([x.iloc[0],0])
    elif x.iloc[1]==4:
        return np.array([x.iloc[0],1])
    else:
        return np.array([x.iloc[0],2])
def count_zeros_from_last(data):
    # Find the index of the last 1 and reverse the list up to that index
    data = list(data)
    last_one_index = len(data) - 1 - data[::-1].index(1)
    return data[last_one_index + 1:].count(0)
def max_consecutive_zeros(df, column_name):
    """
    Calculate the maximum number of consecutive zeros in a specified column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): The name of the column to analyze.

    Returns:
    int: Maximum number of consecutive zeros in the column.
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError("Column not found in DataFrame")

    # Identify changes from 0 to 1 or 1 to 0
    df['shifted'] = df[column_name].shift(1)  # Shift the column to compare with previous row
    df['change'] = df[column_name] != df['shifted']  # Check if there's a change in value

    # Cumulatively sum the 'change' column to group consecutive identical values
    df['groups'] = df['change'].cumsum()

    # Filter for zeros and get the length of each group of zeros
    max_consecutive_zeros = df[df[column_name] == 0].groupby('groups').size().max()

    # Clean up temporary columns to leave the DataFrame as it was
    df.drop(['shifted', 'change', 'groups'], axis=1, inplace=True)

    return max_consecutive_zeros
def expand_and_predict_max_pess(df, ordering, pricing, model,index_dataset):
    """
    Expands the DataFrame by appending each combination of actions and ordering to each row, predicts using a model,
    and returns the maximum prediction for each original row group.

    Args:
    df (pd.DataFrame): Original DataFrame to expand.
    actions (list): List of action values to include in the grid.
    ordering (list): List of ordering indices to include in the grid.
    model: A machine learning model with a predict method.

    Returns:
    pd.DataFrame: DataFrame with original columns and the maximum predictions.
    """
    # Creating the grid as a DataFrame
    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
    df=df.reset_index()
    # Adding a temporary key for merging
    df['key'] = 1
    grid['key'] = 1
    # Change columns names

    # Perform a cross join
    expanded_df = pd.merge(df, grid, on='key').drop('key', axis=1)

    # Ensure the expanded DataFrame is formatted correctly for the model prediction
    # This step may need customization based on model requirements
    expanded_df.columns=["index"] + list(model.feature_names_in_)
    # Predicting
    predictions = model.predict(expanded_df.iloc[:,1:])
    #Uncertainty Quant
    k=7*(index_dataset+1)
    if model.get_params(['kernel_ridge'])['kernel_ridge__kernel']=='rbf':
        alpha =model.get_params(['kernel_ridge'])['kernel_ridge__alpha']
        gamma =model.get_params(['kernel_ridge'])['kernel_ridge__gamma']
        UQ=compute_sigma_for_dataframe(expanded_df.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='rbf' ,gamma=gamma, zeta=alpha, beta_no=1)
    elif model.get_params(['kernel_ridge'])['kernel_ridge__kernel']=='laplacian':
        alpha =model.get_params(['kernel_ridge'])['kernel_ridge__alpha']
        gamma =model.get_params(['kernel_ridge'])['kernel_ridge__gamma']
        UQ=compute_sigma_for_dataframe(expanded_df.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='laplacian' ,gamma=gamma, zeta=alpha, beta_no=1)

    # Combined expression
    #Combine Predictions
    expanded_df['predictions'] = predictions - UQ

    # Group by original identifiers and find the maximum prediction for each group
    max_predictions = expanded_df.groupby(["index"])['predictions'].max()

    return max_predictions
def compute_sigma_for_dataframe(X, x_no,kernel='rbf' ,gamma=1, zeta=2, beta_no=1):
    """
    Computes sigma(x) for each row in DataFrame X based on provided kernel evaluations and parameters.

    Parameters:
        X (pd.DataFrame): DataFrame where each row is a point x for which sigma is computed.
        x_no (np.array): Array of n_o points used in the kernel computations.
        gamma (float): The gamma parameter for the RBF kernel.
        zeta (float): Regularization parameter added to the kernel matrix diagonal.
        beta_no (float): Scaling factor for the sigma function.

    Returns:
        pd.Series: Series containing the computed sigma values for each row in X.
    """
    # Compute kernel evaluations for n_o points using pairwise_kernels
    K_no = pairwise_kernels(x_no, x_no, metric=kernel, gamma=gamma)
    inverse_term = np.linalg.inv(K_no + zeta * np.eye(x_no.shape[0]))

    # Function to compute sigma for a single row
    def compute_single_sigma(x):
        k_no_x = pairwise_kernels(x_no, x.reshape(1, -1), metric=kernel, gamma=gamma)
        k_no_x_prime_x = pairwise_kernels(x.reshape(1, -1), x.reshape(1, -1), metric=kernel, gamma=gamma) - k_no_x.T @ inverse_term @ k_no_x
        sigma_x = beta_no * (k_no_x_prime_x / zeta)
        return sigma_x[0, 0]

    # Apply to each row in the DataFrame
    sigma_values = X.apply(lambda row: compute_single_sigma(row.values), axis=1)
    return sigma_values
allowed_c =3
pricing=[2,4,6]
log_names = ['state_log', 'ordering_log', 'price_log']

#FQI anc Pessımısm combined running parallel --- uniform behaviour
rewards_dic = {}
rewards_dic_pess = {}
pricing=[2,4,6]
for n_episode in [150]:
    for ind_run in range(1):
        #print(n_episode,ind_run)
        allowed_c =3
        env = InvManagementMasterEnv(allowed_censoring=allowed_c) #4
        ordering=[i for i in range(env.supply_capacity+1)]
        df1 =env.data_generator(None,n_episode)
        df1=env.impute_reward(df1)

        n = allowed_c+1
        models = [np.nan for i in range(n)]
        gamma = 0.9
        O0_N=df1[df1["Censor_0"]==1].drop(["Observed Profit","Real Profit","Demand","Estimated Demand"],axis=1)
        datasets = [O0_N if i == 0 else generate_partition(df1, i).dropna()  for i in range(n)]

        pipelines = [Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_ridge', KernelRidge())
        ]) for _ in range(n)]

        param_grid = {
            'kernel_ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel_ridge__kernel': [ 'rbf','laplacian'],
            'kernel_ridge__gamma': [0.01, 0.1, 1],  # Only relevant for rbf and poly

        }
        models_pess = [np.nan for i in range(n)]
        pipelines_pess = [Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_ridge', KernelRidge())
        ]) for _ in range(n)]
        cv=KFold(n_splits=5)
        for i in range(10):  # Number of iterations
            for j in range(n):  # Iterate over each model
                #print(i,j)
                data = datasets[j]  # Get the dataset for the j-th model
                if i == 0:
                    target = data.loc[:, "Imputed Profit"]
                    target_pess = data.loc[:, "Imputed Profit"]
                else:
                    rewards = data.loc[:, "Imputed Profit"]
                    index_model_1 = data["Censor_1"] == 0
                    index_model_0 = data["Censor_1"] == 1
                    if j<n-1:

                        combined_predictions = pd.Series(0, index=rewards.index)
                        if len(data.iloc[np.where(index_model_1)[0], :-1]) == 0:
                            # Do nothing for model 1
                            pass
                        else:
                            # Perform the prediction for model 1
                            predictions_model_1 = expand_and_predict_max(
                                data.iloc[np.where(index_model_1)[0], :-1],
                                ordering,
                                pricing,
                                models[j + 1]
                            )
                            combined_predictions.loc[index_model_1] = predictions_model_1
                        if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                            # Do nothing
                            pass
                        else:
                            # Perform the prediction
                            predictions_model_0 = expand_and_predict_max(
                                data.iloc[np.where(index_model_0)[0], -6:-1],
                                ordering,
                                pricing,
                                models[0]
                            )
                            combined_predictions.loc[index_model_0] = predictions_model_0

                        target = rewards + gamma * combined_predictions



                        #----Pessimism-----
                        combined_predictions_pess = pd.Series(0, index=rewards.index)

                        if len(data.iloc[np.where(index_model_1)[0], :-1]) == 0:
                            # Do nothing for model 1
                            pass
                        else:
                            # Perform the prediction for model 1
                            predictions_model_1_pess = expand_and_predict_max_pess(
                                data.iloc[np.where(index_model_1)[0], :-1], ordering, pricing, models_pess[j + 1],
                                j + 1)
                            combined_predictions_pess.loc[index_model_1] = predictions_model_1_pess
                        if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                            # Do nothing
                            pass
                        else:
                            # Perform the prediction
                            predictions_model_0_pess = expand_and_predict_max_pess(
                                data.iloc[np.where(index_model_0)[0], -6:-1], ordering, pricing, models_pess[0], 0)
                            combined_predictions_pess.loc[index_model_0] = predictions_model_0_pess

                        target_pess = rewards + gamma * combined_predictions_pess


                    else:
                        combined_predictions = pd.Series(0, index=rewards.index)

                        if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                            # Do nothing
                            pass
                        else:
                            # Perform the prediction
                            predictions_model_0 = expand_and_predict_max(
                                data.iloc[np.where(index_model_0)[0], -6:-1],
                                ordering,
                                pricing,
                                models[0]
                            )
                            combined_predictions.loc[index_model_0] = predictions_model_0

                        target = rewards + gamma * combined_predictions


                        ####-----Pessimism-----
                        combined_predictions_pess = pd.Series(0, index=rewards.index)

                        if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                            # Do nothing
                            pass
                        else:
                            # Perform the prediction
                            predictions_model_0_pess = expand_and_predict_max_pess(
                                data.iloc[np.where(index_model_0)[0], -6:-1], ordering, pricing, models_pess[0], 0)
                            combined_predictions_pess.loc[index_model_0] = predictions_model_0_pess

                        target_pess = rewards + gamma * combined_predictions_pess

                # Fit the model
                k=7*(j+1)

                models[j] =GridSearchCV(pipelines[j],param_grid=param_grid,cv=cv,scoring="neg_mean_squared_error").fit(data.iloc[:, :k], target)

                dic_param_pess = {'kernel_ridge__alpha':models[j].best_estimator_.get_params(['kernel_ridge'])['kernel_ridge__alpha'],
'kernel_ridge__degree':models[j].best_estimator_.get_params(['kernel_ridge'])['kernel_ridge__degree'],
'kernel_ridge__gamma':models[j].best_estimator_.get_params(['kernel_ridge'])['kernel_ridge__gamma'],
'kernel_ridge__kernel':models[j].best_estimator_.get_params(['kernel_ridge'])['kernel_ridge__kernel']}

                pipelines_pess[j].set_params(**dic_param_pess)
                models_pess[j]=pipelines_pess[j].fit(data.iloc[:, :k], target_pess)

        env = InvManagementMasterEnv(allowed_censoring=allowed_c) #4
        env_pess=InvManagementMasterEnv(allowed_censoring=allowed_c)
        eps = 100
        rewards = []
        rewards_pess = []
        rewards_real_pess=[]
        rewards_observed_pess=[]
        censoring_observed_pess=[]

        rewards_real=[]
        rewards_observed=[]
        censoring_observed=[]
        #Pessimism Test
        for v in range(eps):
            inital_state_pess = env_pess.reset()
            reward_pess = 0
            counter = 0
            while True:

                if counter == 0:

                    ##----Pessimismm-----------
                    inital_state_pess = pd.DataFrame(inital_state_pess.reshape(1,-1)).reset_index()
                    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
                    inital_state_pess['key'] = 1
                    grid['key'] = 1
                    # Perform a cross join
                    expanded_df_pess = pd.merge(inital_state_pess, grid, on='key').drop('key', axis=1)
                    # Ensure the expanded DataFrame is formatted correctly for the model prediction
                    # This step may need customization based on model requirements
                    expanded_df_pess.columns=["index"] + list(models_pess[0].feature_names_in_)

                    index_dataset=0
                    k=7*(index_dataset+1)
                    Q_predicted_pess = models_pess[0].predict(expanded_df_pess.iloc[:,1:])
                    if models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='rbf':
                        alpha =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                        gamma =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                        UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='rbf' ,gamma=gamma, zeta=alpha, beta_no=1)
                    elif models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='laplacian':
                        alpha =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                        gamma =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                        UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='laplacian' ,gamma=gamma, zeta=alpha, beta_no=1)
                    action_taken_pess = np.argmax(Q_predicted_pess-UQ)
                    action_pess = action_map(grid.iloc[action_taken_pess,0:2])

                else:
                    if count_zeros_from_last(env_pess.Indicator_censor[:counter])==0:
                        states_actions_pess = generalized_stack(env_pess, log_names, counter,count_zeros_from_last(env_pess.Indicator_censor[:counter]))
                        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])

                        states_actions_pess['key'] = 1
                        grid['key'] = 1
                        # Perform a cross join
                        expanded_df_pess = pd.merge(states_actions_pess, grid, on='key').drop('key', axis=1)
                        # Ensure the expanded DataFrame is formatted correctly for the model prediction
                        # This step may need customization based on model requirements
                        expanded_df_pess.columns=["index"] + list(models_pess[0].feature_names_in_)
                        index_dataset=0
                        k=7*(index_dataset+1)
                        Q_predicted_pess = models_pess[0].predict(expanded_df_pess.iloc[:,1:])
                        if models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='rbf':
                            alpha =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                            gamma =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                            UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='rbf' ,gamma=gamma, zeta=alpha, beta_no=1)
                        elif models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='laplacian':
                            alpha =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                            gamma =models_pess[0].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                            UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='laplacian' ,gamma=gamma, zeta=alpha, beta_no=1)
                        action_taken_pess = np.argmax(Q_predicted_pess-UQ)
                        action_pess = action_map(grid.iloc[action_taken_pess,0:2])

                    else:
                        ##----Pessimism
                        censor=count_zeros_from_last(env_pess.Indicator_censor[:counter])
                        states_actions_pess = generalized_stack(env_pess, log_names, counter,censor)
                        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])

                        states_actions_pess['key'] = 1
                        grid['key'] = 1
                        # Perform a cross join
                        expanded_df_pess = pd.merge(states_actions_pess, grid, on='key').drop('key', axis=1)
                        # Ensure the expanded DataFrame is formatted correctly for the model prediction
                        # This step may need customization based on model requirements
                        expanded_df_pess.columns=["index"] + list(models_pess[censor].feature_names_in_)

                        index_dataset=censor
                        k=7*(index_dataset+1)
                        Q_predicted_pess = models_pess[censor].predict(expanded_df_pess.iloc[:,1:])
                        if models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='rbf':
                            alpha =models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                            gamma =models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                            UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='rbf' ,gamma=gamma, zeta=alpha, beta_no=1)
                        elif models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__kernel']=='laplacian':
                            alpha =models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__alpha']
                            gamma =models_pess[censor].get_params(['kernel_ridge'])['kernel_ridge__gamma']
                            UQ=compute_sigma_for_dataframe(expanded_df_pess.iloc[:, 1:], datasets[index_dataset].iloc[:, :k],kernel='laplacian' ,gamma=gamma, zeta=alpha, beta_no=1)
                        action_taken_pess = np.argmax(Q_predicted_pess-UQ)
                        action_pess = action_map(grid.iloc[action_taken_pess,0:2])

                counter = counter +1
                s, r, reward_real, d, done = env_pess.step(action_pess)


                if done:
                    rewards_real_pess.append(env_pess.P.mean())
                    rewards_observed_pess.append(env_pess.P_censored.mean())
                    censoring_observed_pess.append(env_pess.Indicator_censor.mean())
                    break
        rewards_dic_pess[f"{n_episode},{ind_run}"]=[np.mean(rewards_real_pess),np.mean(rewards_observed_pess),np.mean(censoring_observed_pess)]

        #FQI test
        for v in range(eps):
            inital_state = env.reset()
            reward = 0
            counter = 0
            while True:
                #print(counter)
                if counter == 0:
                    inital_state = pd.DataFrame(inital_state.reshape(1,-1)).reset_index()
                    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
                    inital_state['key'] = 1
                    grid['key'] = 1
                    # Perform a cross join
                    expanded_df = pd.merge(inital_state, grid, on='key').drop('key', axis=1)
                    # Ensure the expanded DataFrame is formatted correctly for the model prediction
                    # This step may need customization based on model requirements
                    expanded_df.columns=["index"] + list(models[0].feature_names_in_)
                    Q_predicted = models[0].predict(expanded_df.iloc[:,1:])
                    action_taken = np.argmax(Q_predicted)
                    action = action_map(grid.iloc[action_taken,0:2])

                else:
                    if count_zeros_from_last(env.Indicator_censor[:counter])==0:
                        states_actions = generalized_stack(env, log_names, counter,count_zeros_from_last(env.Indicator_censor[:counter]))
                        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])

                        states_actions['key'] = 1
                        grid['key'] = 1
                        # Perform a cross join
                        expanded_df = pd.merge(states_actions, grid, on='key').drop('key', axis=1)
                        # Ensure the expanded DataFrame is formatted correctly for the model prediction
                        # This step may need customization based on model requirements
                        expanded_df.columns=["index"] + list(models[0].feature_names_in_)
                        Q_predicted = models[0].predict(expanded_df.iloc[:,1:])
                        action_taken = np.argmax(Q_predicted)
                        action = action_map(grid.iloc[action_taken,0:2])
                    else:
                        censor=count_zeros_from_last(env.Indicator_censor[:counter])
                        states_actions = generalized_stack(env, log_names, counter,censor)
                        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])

                        states_actions['key'] = 1
                        grid['key'] = 1
                        # Perform a cross join
                        expanded_df = pd.merge(states_actions, grid, on='key').drop('key', axis=1)
                        # Ensure the expanded DataFrame is formatted correctly for the model prediction
                        # This step may need customization based on model requirements
                        expanded_df.columns=["index"] + list(models[censor].feature_names_in_)
                        Q_predicted = models[censor].predict(expanded_df.iloc[:,1:])
                        action_taken = np.argmax(Q_predicted)
                        action = action_map(grid.iloc[action_taken,0:2])

                counter = counter +1
                s, r, reward_real, d, done = env.step(action)

                reward += reward_real
                if done:
                    rewards_real.append(env.P.mean())
                    rewards_observed.append(env.P_censored.mean())
                    censoring_observed.append(env.Indicator_censor.mean())
                    break
        rewards_dic[f"{n_episode},{ind_run}"]=[np.mean(rewards_real),np.mean(rewards_observed),np.mean(censoring_observed)]
df_fqi = pd.DataFrame(rewards_dic)

# Specify the output Excel file path
output_file = 'outputs/output_estimated_optimal_150.xlsx'

# Save the DataFrame to Excel
df_fqi.to_excel(output_file, index=False)



