from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.neighbors import KNeighborsRegressor

dict_models = {
    'Linear' : {'fingerprints' : LinearRegression(), 'rdkit': LinearRegression() , 
            'quantum' : LinearRegression(), 'hammett': LinearRegression()},
    
    "LR" : {'fingerprints' : Ridge(alpha=6, max_iter=None, random_state=1, tol=0.01, solver = 'saga'), 'rdkit': Ridge(alpha=5, max_iter=500, random_state=1, tol=0.0001, solver = 'sparse_cg') , 
            'quantum' : Ridge(alpha=5, max_iter=500, random_state=1, tol=0.0001, solver = 'sparse_cg'), 'hammett': Ridge(alpha=0.01, max_iter=None, random_state=1, tol=0.0001, solver = 'auto')},
    
    "Bayes. Ridge": {'fingerprints' : BayesianRidge(alpha_1=1e-07, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-07,
              tol=0.01), 'rdkit': BayesianRidge(alpha_1=0.0001, alpha_2=1e-07, lambda_1=1e-07, lambda_2=0.0001,
              tol=0.01), 
            'quantum' : BayesianRidge(alpha_1=1e-07, alpha_2=1e-07, lambda_1=0.0001, lambda_2=0.0001,
             tol=0.01), 'hammett': BayesianRidge(alpha_1=0.0001, alpha_2=1e-07, lambda_1=1e-07, lambda_2=0.0001,
             tol=0.01)},
    
    "LASSO" :{'fingerprints' : Lasso(alpha=1, max_iter=1000, precompute=False, random_state=1,
      selection='random', tol=0.001), 'rdkit': Lasso(alpha=0.1, max_iter=10000, precompute=False, selection = 'cyclic', random_state=1, tol=0.0001), 
            'quantum' : Lasso(alpha=0.001, max_iter=1000000, precompute=True, selection = 'cyclic', random_state=1, tol=0.01), 'hammett': Lasso(alpha=0.1, max_iter=10000, precompute=False, selection = 'cyclic', random_state=1, tol=0.00001)},
    
    "SVR": {'fingerprints' : SVR(C=1.0, epsilon=1.0, kernel='linear', shrinking=False), 'rdkit': SVR(epsilon=0.01, kernel='linear', tol=0.0001) , 
            'quantum' : SVR(epsilon=0.01, kernel='linear', tol=0.0001), 'hammett': SVR(epsilon=0.01, kernel='linear', tol=0.0001)},
    
   "Tree" : {'fingerprints' :DecisionTreeRegressor(criterion='absolute_error', max_depth=10,
                      max_features=1.0, min_samples_split=5, random_state=1), 'rdkit': DecisionTreeRegressor(random_state=1) ,
                    
            'quantum' : DecisionTreeRegressor(random_state=1), 'hammett': DecisionTreeRegressor(criterion='poisson', max_depth=10, max_features=1.0,
                      random_state=1)},
    
    'RF' : {'fingerprints' : RandomForestRegressor(), 'rdkit': RandomForestRegressor(n_estimators = 100,
 max_features = 'sqrt',
 max_depth = 15,
 criterion = 'friedman_mse',
 bootstrap = False, min_samples_split = 3) , 
            'quantum' : RandomForestRegressor(criterion='friedman_mse', max_depth=10, max_features=None,
                      n_estimators=200), 'hammett': RandomForestRegressor()#params par défaut les mieux
           }  ,
    
    'Grad. Boost.': {'fingerprints' : GradientBoostingRegressor(loss='huber', max_features='sqrt',
                                           max_leaf_nodes=5,
                                           min_impurity_decrease=1.0,
                                           min_samples_leaf=3,
                                           min_samples_split=4,
                                           n_estimators=500, subsample=0.5), 'rdkit': GradientBoostingRegressor(max_features='sqrt', max_leaf_nodes=2,
                          min_samples_leaf=2, n_estimators=500, random_state=1,
                          subsample=0.75), 
            'quantum' : GradientBoostingRegressor(learning_rate=0.1, loss='squared_error', max_leaf_nodes=15,
                          n_estimators=500, random_state=1, subsample=0.5, 
                                  max_depth = 2, min_samples_split = 2), 
           'hammett': GradientBoostingRegressor(loss='squared_error', max_leaf_nodes=2, n_estimators=500,
                          random_state=1, subsample=0.75, learning_rate = 0.1)},
    
    'GPR' : {'fingerprints' : GaussianProcessRegressor(alpha=1e-11, kernel=RBF(length_scale=1.0),
                         normalize_y=True, random_state=1), 'rdkit': GaussianProcessRegressor(alpha=1e-09, kernel=RBF(length_scale=1),
                         normalize_y=True, random_state=1) , 
            'quantum' : GaussianProcessRegressor(alpha=1e-09, kernel=RBF(length_scale=1),
                         normalize_y=True, random_state=1), 'hammett': GaussianProcessRegressor(alpha=1e-05, kernel=RBF(length_scale=1), n_restarts_optimizer = 0,
                         normalize_y=True)},
    
    'KNN' :{'fingerprints' : KNeighborsRegressor(p=1), 'rdkit': KNeighborsRegressor(n_neighbors=10, p=1, weights='distance') , 
            'quantum' : KNeighborsRegressor(n_neighbors=10, p=1, weights='distance'), 'hammett': KNeighborsRegressor(p=1, weights='distance')},
    
    'MLP' : {'fingerprints' : MLPRegressor(activation='logistic', alpha=0.001, hidden_layer_sizes=(100,),
             learning_rate='adaptive', max_iter=20000, n_iter_no_change=5,
             solver='lbfgs', tol=0.0001), 
            'rdkit': MLPRegressor(activation='identity', alpha=0.001, hidden_layer_sizes=(30,40),
             learning_rate='adaptive', max_iter=200000, n_iter_no_change = 5, tol=1e-3, solver='lbfgs'),
            'quantum' :  MLPRegressor(activation='identity', alpha=1e-05, early_stopping=True,
             hidden_layer_sizes=(100,), learning_rate='invscaling',
             max_iter=200000, solver='lbfgs'),
            'hammett' : MLPRegressor(max_iter = 20000, hidden_layer_sizes=(100,), 
                     activation = 'identity', alpha = 0.001, solver = 'lbfgs', learning_rate = 'invscaling')}
}