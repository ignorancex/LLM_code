import treatment_prediction.baseline_methods.baseline as baseline
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from econml.inference import BootstrapInference
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter

from econml.metalearners import XLearner, TLearner, SLearner
from econml.cate_interpreter import SingleTreePolicyInterpreter

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from econml.dml import CausalForestDML
import shap

from econml.policy import DRPolicyTree, DRPolicyForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import matplotlib.pyplot as plt
import os
from ganite import Ganite
from ganite.datasets import load
from ganite.utils.metrics import sqrt_PEHE_with_diff

import numpy as np
import matplotlib.pyplot as plt

from econml.panel.dml import DynamicDML
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
from sklearn.preprocessing import PolynomialFeatures
from econml.orf import DMLOrthoForest, DROrthoForest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
import torch

# The following are for the average treatment effect estimation
def linear_model(X, T, Y):
    # est = SparseLinearDML(model_y=RandomForestRegressor(),
    #                    model_t=RandomForestRegressor(),
    #                 #    featurizer=PolynomialFeatures(degree=3),
    #                    random_state=123)
    # est.fit(Y, T, X=X)
    # te_pred1 = est1.effect(X_test)
    est = LinearDML(model_y=RandomForestRegressor(),
                model_t=RandomForestRegressor(),
                random_state=123)
    ### Estimate with OLS confidence intervals
    est.fit(Y, T, X=X) # W -> high-dimensional confounders, X -> features
    return est

def dynamic_dml(X, T, Y):
    est = DynamicDML()
    # Or specify hyperparameters
    est = DynamicDML(model_y=LassoCV(cv=3), 
                    model_t=LassoCV(cv=3), 
                    cv=3)
    est.fit(Y, T, X=X, W=None, inference="auto")
    return est

def causal_forest_main(train_dataset, valid_dataset, test_dataset, max_depth=6, n_estimators=1):
    train_valid_features = torch.cat([train_dataset.features, valid_dataset.features])
    train_valid_treatment = torch.cat([train_dataset.treatment_array.view(-1), valid_dataset.treatment_array.view(-1)])
    train_valid_outcome = torch.cat([train_dataset.outcome_array.view(-1), valid_dataset.outcome_array.view(-1)])
    if train_dataset.y_scaler is not None:
        train_valid_outcome = torch.from_numpy(train_dataset.y_scaler.inverse_transform(train_valid_outcome.numpy().reshape(-1,1)))
    
    
    # est = CausalForestDML(max_depth=max_depth, n_estimators=2, subforest_size=2, discrete_treatment=True, model_t=DecisionTreeClassifier(max_depth=max_depth), model_y=DecisionTreeRegressor(max_depth=max_depth))
    est = CausalForestDML(max_depth=max_depth, n_estimators=2, subforest_size=2, discrete_treatment=True, model_t=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators), model_y=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators))
    est.fit(train_valid_outcome.numpy(), train_valid_treatment.long().numpy(), X=train_valid_features.numpy())
    
    pred_treatment_effect = est.effect(test_dataset.features.numpy())
    
    
    if test_dataset.count_outcome_array is not None:
        pos_outcome = test_dataset.outcome_array.numpy()*(test_dataset.treatment_array.numpy() == 1).astype(float) + test_dataset.count_outcome_array.numpy()*(test_dataset.treatment_array.numpy() == 0).astype(float)
        neg_outcome = test_dataset.outcome_array.numpy()*(test_dataset.treatment_array.numpy() == 0).astype(float) + test_dataset.count_outcome_array.numpy()*(test_dataset.treatment_array.numpy() == 1).astype(float)

        if train_dataset.y_scaler is not None:
            neg_outcome = train_dataset.y_scaler.inverse_transform(neg_outcome.reshape(-1,1))
            pos_outcome = train_dataset.y_scaler.inverse_transform(pos_outcome.reshape(-1,1))
        gt_treatment_effect = pos_outcome - neg_outcome
        print("Average treatment effect error: ", np.mean(np.abs(gt_treatment_effect - pred_treatment_effect)))
        
       
    return est

def extract_decision_rules(tree, feature_names, sample, return_features =False):
    """
    Extract decision rules for an input test sample from a Decision Tree classifier.

    Parameters:
        - tree: The Decision Tree classifier model.
        - feature_names: A list of feature names.
        - sample: A single input test sample as a 1D NumPy array or list.

    Returns:
        - decision_rules: A list of strings representing the decision rules that lead to the prediction.
    """
    decision_rules = []
    if return_features:
        selected_features = []

    # Get the feature indices corresponding to feature names
    feature_indices = {feature: i for i, feature in enumerate(feature_names)}

    def _extract_rules(node, rule, features=None):
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
            # Leaf node, add the prediction
            # class_idx = tree.apply([sample])[0]
            # class_label = tree.classes_[class_idx]
            value = tree.apply([sample])[0]
            decision_rules.append(rule)
            if features is not None:
                selected_features.append(features)
            # decision_rules.append(rule.app" => outcome: {value}")
        else:
            # Non-leaf node, traverse left and right children
            feature_idx = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]

            feature_name = feature_names[feature_idx]
            sample_value = sample[feature_idx]

            if sample_value <= threshold:
                rule.append(f"{feature_name} <= {threshold}")
                if features is not None:
                    features.append(feature_idx)
                _extract_rules(tree.tree_.children_left[node], rule, features=features)
            else:
                rule.append(f"{feature_name} > {threshold}")
                if features is not None:
                    features.append(feature_idx)
                _extract_rules(tree.tree_.children_right[node], rule, features=features)

    if not return_features:
        _extract_rules(0, [])
    else:
        _extract_rules(0, [], [])

    if not return_features:
        return decision_rules
    else:
        return decision_rules, selected_features

def interpret_causal_forest_main(est, test_dataset):
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=est.model_y.max_depth, min_samples_leaf=est.model_y.min_samples_split)
    intrp.interpret(est, test_dataset.features.numpy())
    decision_rules = extract_decision_rules(intrp.tree_model_, test_dataset.num_cols + test_dataset.cat_cols, test_dataset.features.numpy()[0])
    # intrp.plot(feature_names=test_dataset.num_cols + test_dataset.cat_cols)
    
    print(decision_rules)

def causal_forest(X, T, Y):
    est = CausalForestDML(max_depth=2, n_estimators=2, subforest_size=2, discrete_treatment=True, model_t=RandomForestClassifier(n_estimators=1), model_y=RandomForestClassifier(n_estimators=1))
    est = CausalForestDML()
    # Or specify hyperparameters
    # est = CausalForestDML(criterion='het', n_estimators=10,       
    #                     min_samples_leaf=10, 
    #                     max_depth=10, max_samples=0.5,
    #                     discrete_treatment=False,
    #                     model_t=LassoCV(), model_y=LassoCV())
    
    est.fit(Y, T, X=X)
    est.effect(X)
    # est.score(Y,T,X)
    return est

def causal_forest_predict(est, X, T):
    estimators = est._models_nuisance
    sum_pred_y = 0
    count = 0
    for k in range(len(estimators)):
        for j in range(len(estimators[k])):
            pred_y = est._models_nuisance[k][j]._model_y.predict(X, None)
            sum_pred_y += pred_y
            count += 1
    avg_pred_y = sum_pred_y/count
    return avg_pred_y
def ortho_forest(X, T, Y):
    est = DMLOrthoForest(max_depth=10, n_trees=10)
    est.fit(Y, T, X=X)
    est.effect(X)
    return est


def xlearner_evaluation_by_treatment(est, X, T):
    unique_Ts = list(set(T))
    sorted_unique_Ts = sorted(unique_Ts)
    full_outcome = np.zeros_like(T)
    for t in sorted_unique_Ts:
        curr_X = X[T == t]
        # if t == 0:
        #     full_outcome[T == t] = est.models[t].predict(curr_X)
        # else:    
        full_outcome[T == t] = est.models[t].predict(curr_X)
    return full_outcome

def slearner_evaluation_by_treatment(est, X, T):
    treatment_onehot = np.zeros((X.shape[0], 2))
    treatment_onehot[np.arange(X.shape[0]), T] = 1
    full_outcome = est.overall_model.predict_proba(np.concatenate([X, treatment_onehot], axis=1))
    # unique_Ts = list(set(T))
    # sorted_unique_Ts = sorted(unique_Ts)
    # full_outcome = np.zeros_like(T)
    # for t in sorted_unique_Ts:
    #     curr_X = X[T == t]
    #     # if t == 0:
    #     #     full_outcome[T == t] = est.models[t].predict(curr_X)
    #     # else:    
    #     full_outcome[T == t] = est.models[t].predict(curr_X)
    return full_outcome[:,1]

def xlearner(X, T, Y):
    # est = XLearner(models=RandomForestClassifier(),
    #           propensity_model=RandomForestClassifier(),
    #           cate_models=RandomForestClassifier())
    
    est = XLearner(models=LogisticRegressionCV(),propensity_model=LogisticRegressionCV(), cate_models=LogisticRegressionCV())
    
    est.fit(Y, T, X=X)
    return est
    # est.effect(X)
def Tlearner(X, T, Y):
    # est = XLearner(models=RandomForestClassifier(),
    #           propensity_model=RandomForestClassifier(),
    #           cate_models=RandomForestClassifier())
    est = TLearner(models=GradientBoostingClassifier())
    est = TLearner(models=RandomForestClassifier())
    # est = XLearner(models=LogisticRegressionCV(),propensity_model=LogisticRegressionCV(), cate_models=LogisticRegressionCV())
    
    est.fit(Y, T, X=X)
    return est  

def Slearner(X, T, Y):
    # est = XLearner(models=RandomForestClassifier(),
    #           propensity_model=RandomForestClassifier(),
    #           cate_models=RandomForestClassifier())
    # est = SLearner(overall_model=GradientBoostingClassifier())
    est = SLearner(overall_model=RandomForestClassifier(n_estimators=1, max_depth=2))
    # est = SLearner(overall_model=DecisionTreeClassifier(max_depth=2))
    # est = XLearner(models=LogisticRegressionCV(),propensity_model=LogisticRegressionCV(), cate_models=LogisticRegressionCV())
    
    est.fit(Y, T, X=X)
    return est  

def DR_policy_learning(X, T, Y):
    # fit a single binary decision tree policy
    policy = DRPolicyTree(max_depth=20, min_impurity_decrease=0.001, honest=True)
    policy.fit(Y, T, X=X)
    return policy

def DR_policy_learning_forest(X, T, Y):
    # fit a single binary decision tree policy
    policy = DRPolicyForest(n_estimators=10, max_depth=20, min_impurity_decrease=0.001, honest=True)
    policy.fit(Y, T, X=X)
    return policy

def draw_policy_fig(policy, treatment_names=["0","1"]):
    # recommended_T = policy.predict(X)
    plt.figure(figsize=(50,10))
    policy.plot(treatment_names=treatment_names)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'policy_fig.png'), bbox_inches='tight')

def draw_policy_fig_forest(policy, treatment_names=["0","1"]):
    # recommended_T = policy.predict(X)
    plt.figure(figsize=(50,10))
    for n in range(policy.n_estimators):
        policy.plot(n,treatment_names=treatment_names)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'policy_fig_forest_' + str(n) + '.png'), bbox_inches='tight')

def interpret_policy(est, X, feature_names, max_depth=10, min_samples_leaf=2):
    # We find a tree-based treatment policy based on the CATE model
    intrp = SingleTreePolicyInterpreter(risk_level=0.05, max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_impurity_decrease=.001)
    intrp.interpret(est, X, sample_treatment_costs=0.2)
    # Plot the tree
    plt.figure(figsize=(50, 10))
    intrp.plot(feature_names=feature_names, fontsize=12)
    plt.show()


def interpret(est, X):
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2, min_samples_leaf=10)
    # We interpret the CATE model's behavior based on the features used for heterogeneity
    intrp.interpret(est, X)
    
def interpret_tree(est, X, feature_names, max_depth=2, min_samples_leaf=10, output_suffix=""):
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    intrp.interpret(est, X)
    
    plt.figure(figsize=(50, 10))
    intrp.plot(feature_names=feature_names, fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'tree_interpret' + output_suffix + '.png'), bbox_inches='tight')
    print()
    
def interpret_shap(est, X):
    shap_values = est.shap_values(X)
    shap.summary_plot(shap_values['Y0']['T0_1'],show=False)
    f = plt.gcf()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'shap_interpret.png'), bbox_inches='tight')

    


# The following are for the individual treatment effect estimation
def ganite(X, T, Y):
    model = Ganite(X, T, Y, num_iterations=500)

    return model


def fit_decision_tree(feature_names, model, test_X, max_depth=4, output_suffix="", pred_test_y=None):
    if pred_test_y is None:
        pred_test_y = model.inference_nets(test_X.cuda()).detach().cpu()
    pred_discrete_y = (pred_test_y > 0.5).type(torch.int)
    dt_classifier_ls = []
    for k in range(pred_test_y.shape[1]):
        curr_y = pred_discrete_y[:,k].cpu().numpy()
        
        dt_classifier = DecisionTreeRegressor(max_depth=max_depth)
        dt_classifier.fit(test_X.numpy(), pred_test_y[:,k].numpy())
        plt.figure(figsize=(40, 10))
        tree.plot_tree(dt_classifier, feature_names=feature_names, fontsize=12, )
        plt.savefig(os.path.join(os.path.dirname(__file__), 'dt_tree' + output_suffix + "_" + str(k) + '.png'), bbox_inches='tight')

        dt_classifier_ls.append(dt_classifier)
    return dt_classifier_ls
