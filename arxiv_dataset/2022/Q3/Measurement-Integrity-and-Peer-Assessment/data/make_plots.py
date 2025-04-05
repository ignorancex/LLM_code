"""
Recreate the figures in the paper by uncommenting the associated function at the bottom of this script.

The generated figures are saved in either the "main_paper" or "appendix" folders in the "figures" folder located in the "data" directory (i.e. the same directory as this file).

Note the numbering of the figures corresponds with the full version of the paper (on arXiv).

@author: Noah Burrell <burrelln@umich.edu>
"""
from json import load
import os
from statistics import mean
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches                                                  
import numpy as np
from scipy.stats import zscore
import seaborn as sns

from model_code import *

def compare_metrics(metric):
    old_names = {value:key for key,value in mechanism_name_map.items()}
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    #r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    #r'Dominantly Truthful',
                    r'Truthful',             
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',   
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            "Strongly Truthful": 4,
            #"Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'payments-vs-mse_with-bias-in-model'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d)
        
    filename = 'payments-vs-mse_mse-p' 
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d) # Update to use up-to-date MSE_P data (improved estimate of consensus grade)
        
    json_data.pop("DMI: 4")
    
    tau_accuracy_dict = {}
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [mean(json_data[key][str(num)][metric]) for num in range(1, 16)]
        value = mean(values)
        tau_accuracy_dict[mechanism] = {"ABM": value}
        
    filename = 'payments-vs-mse_Spring17_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Spring17_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        #json_data.update(d)
        #print(json_data)
        for key, nums in d.items():
            json_data[key] = {}
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num] = {metric: []}
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num] = {metric: []}
                    json_data[key][num][metric].append(mean(nums[num][metric]))
                    
    filename = 'payments-vs-mse_Fall17_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
               
    filename = 'payments-vs-mse_Fall17_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    filename = 'payments-vs-mse_Spring19_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Spring19_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    filename = 'payments-vs-mse_Fall19_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Fall19_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [mean(json_data[key][str(num)][metric]) for num in range(1, 5)]
        #value = quantile(values, 0.9)
        value = mean(values)
        tau_accuracy_dict[mechanism]["Real"] = value
        
    return tau_accuracy_dict

def compare_metrics_max_min_mean(metric):
    
    old_names = {value:key for key,value in mechanism_name_map.items()}
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    #r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    #r'Dominantly Truthful',
                    r'Truthful',             
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',   
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',
                ]
        }
    
    tau_accuracy_dict = {}
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        tau_accuracy_dict[mechanism] = {}
      
    json_data = {}
    
    filename = 'payments-vs-mse_Spring17_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Spring17_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        #json_data.update(d)
        #print(json_data)
        for key, nums in d.items():
            json_data[key] = {}
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num] = {metric: []}
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num] = {metric: []}
                    json_data[key][num][metric].append(mean(nums[num][metric]))
                    
    filename = 'payments-vs-mse_Fall17_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
               
    filename = 'payments-vs-mse_Fall17_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    filename = 'payments-vs-mse_Spring19_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Spring19_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    filename = 'payments-vs-mse_Fall19_mse-p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        msep_d = load(file)
    
    filename = 'payments-vs-mse_Fall19_full'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for key, nums in d.items():
            if key == 'MSE_P: 0':
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(msep_d[key][num][metric]))
            else:
                for num in nums.keys():
                    json_data[key][num][metric].append(mean(nums[num][metric]))
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        arr = np.array([json_data[key][str(num)][metric] for num in range(1, 5)])
        values = []
        for i in range(4):
            values.append(mean(arr[:,i]))
        avg = mean(values)
        mini = min(values)
        maxi = max(values)
        tau_accuracy_dict[mechanism]["Max"] = maxi
        tau_accuracy_dict[mechanism]["Min"] = mini
        tau_accuracy_dict[mechanism]["Mean"] = avg
        
    #print(json.dumps(tau_accuracy_dict, indent=2))
        
    return tau_accuracy_dict

def figure_1a(poster_version=False):
    old_names = {value:key for key,value in mechanism_name_map.items()}
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    #r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    #r'Dominantly Truthful',
                    r'Truthful',             
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',  
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',
                ],
            "Mechanism Type":
                [
                    # Non-Parametric Mechanisms
                    r'Non-Parametric',
                    #r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    # Parametric Mechanisms
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            r'$\epsilon$-Strongly Truthful': 4,
            #"Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'payments-vs-mse_with-bias-in-model'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d)
        
    json_data.pop("DMI: 4")
    
    filename = 'payments-vs-mse_mse-p' 
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d) # Update to use up-to-date MSE_P data (improved estimate of consensus grade)
    
    be_accuracy_list = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        #values = [mean(json_data[key][str(num)]["Binary AUCs"]) for num in range(1, 16)] # Current arXiv Version 1/8/23
        values = [mean(json_data[key][str(num)]["Taus"]) for num in range(1, 16)] # Current arXiv Version 1/8/23
        value = mean(values)
        be_accuracy_list.append(value)
    
        
    data["Accuracy"] = be_accuracy_list
    
    robustness_list = []
    strategies = [
                    "NOISE",
                    "FIX-BIAS",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    json_data = {}
        
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'incentives_for_deviating-ce-bias-parametric-bias_correct_false'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            json_data[strategy][num].update(d[strategy][num])
    
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [-1*json_data[strategy][str(num)][key]["Mean Gain"] for strategy in strategies for num in range(10, 100, 10)]
        value = mean(values)
        robustness_list.append(value)
        
    data["Robustness"] = robustness_list
    
    """
    # This code block can be uncommented to set up (with a few additional adjustments below) 
    # an analogous plot considering only a single strategy, e.g. PRIOR as below, 
    # instead of aggregating performance over all considered strategies.
    
    robustness_list_prior = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        value = -1*json_data["PRIOR"]["50"][key]["Mean Gain"]
        robustness_list_prior.append(value)
        
    data["Robustness (Prior)"] = robustness_list_prior
    """
    if poster_version:
        
        data["Robustness"] = zscore(robustness_list)
        data["Accuracy"] = zscore(be_accuracy_list)
        
        _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Mechanism Type", data=data)
        #handles, labels = plt.gca().get_legend_handles_labels()
        
        # order = [0,4,2,3,5,1] # When Informed Truthful is included (6 labels)
        #order=[0,3,1,2] # When Informed Truthful is not included
        #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=r'Equilibrium Concept')
        
        # label points on the plot
        for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
            if s == r'$\Phi$-Div: $H^2$' or s == r'MSE' or s == r'MSE$_P$':
                x_val = x - 0.1
                y_val = y - 0.2
                
            elif s == r'$\Phi$-Div$_P$: $H^2$':
                x_val = x - 0.2
                y_val = y + 0.1
                
            elif s == r'$\Phi$-Div$_P$: TVD':
                x_val = x - 0.375
                y_val = y + 0.1
                
            elif s == r'$\Phi$-Div: TVD':
                x_val = x - 0.2
                y_val = y + 0.1
                
            elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_P$: $\chi^2$':
                x_val = x - 0.225
                y_val = y + 0.1
                
            elif s == r'$\Phi$-Div: $\chi^2$':
                x_val = x - 0.15
                y_val = y + 0.1
                
            elif s == r'OA':
                x_val = x - 0.05
                y_val = y + 0.1
                
            else:
                x_val = x - 0.075
                y_val = y + 0.1
            
                
            plt.text(x = x_val, # x-coordinate position of data label
            y = y_val, # y-coordinate position of data label
            s = s, # data label
            color = 'black') # set colour of line     
        
        ax = plt.gca()
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.set_facecolor('gainsboro')
        
        ax.set_ylabel(r'Measurement Integrity ($z$-score)')
        ax.set_xlabel(r'Robustness Against Strategic Reporting ($z$-score)')
        plt.title(r'Apparent Trade-off Between Integrity and Robustness')
        plt.tight_layout()
        filename = "standardized-2D-tradeoff"
        figure_file = "figures/poster/" + filename + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
    else:
        _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Theoretical Guarantee", palette=guarantee_color_map, data=data)
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # order = [0,4,2,3,5,1] # When Informed Truthful is included (6 labels)
        order=[0,3,1,2] # When Informed Truthful is not included
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=r'Equilibrium Concept', loc='upper right', bbox_to_anchor=(1.005, 1.02))
        
        # label points on the plot
        for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
            if s == r'MSE$_P$':
                x_val = x - 1.6
                y_val = y - 0.03
                
            elif s == r'MSE':
                x_val = x - 1.4
                y_val = y - 0.035
                
            elif s == r'PTS':
                x_val = x - 1.35
                y_val = y + 0.015
                
            elif s == r'OA':
                x_val = x - 0.75
                y_val = y + 0.015
                
            elif s == r'$\Phi$-Div: $H^2$':
                x_val = x - 2
                y_val = y - 0.03
                
            elif s == r'$\Phi$-Div: KL':
                x_val = x - 2
                y_val = y + 0.015
                
            elif s == r'$\Phi$-Div$_P$: TVD':
                x_val = x - 5.5
                y_val = y + 0.015
                
            elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_P$: $\chi^2$' or s == r'$\Phi$-Div$_P$: $H^2$':
                x_val = x - 2.5
                y_val = y + 0.015
                
            else:
                x_val = x - 2.5
                y_val = y + 0.015
            
                
            plt.text(x = x_val, # x-coordinate position of data label
            y = y_val, # y-coordinate position of data label
            s = s, # data label
            color = 'black') # set colour of line     
        
        ax = plt.gca()
        ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.set_facecolor('gainsboro')
        
        ax.set_ylabel(r'Measurement Integrity')
        ax.set_xlabel(r'Robustness Against Strategic Reporting')
        plt.title(r'Apparent Trade-off Between Integrity and Robustness (ABM)')
        plt.tight_layout()
        filename = "2D-tradeoff-updated-strategic"
        figure_file = "figures/main_paper/" + filename + ".pdf"
        # figure_file = "figures/poster/" + filename + ".png"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()

def figure_1b():
    old_names = {value:key for key,value in mechanism_name_map.items()}
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    #r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    #r'Dominantly Truthful',
                    r'Truthful',             
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',  
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            r'$\epsilon$-Strongly Truthful': 4,
            #"Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    #accuracy_dict = compare_metrics_max_min_mean("Binary AUCs")
    accuracy_dict = compare_metrics_max_min_mean("Taus")
    coordinate_dict = {mech:[0, 0, 0, d["Mean"], d["Min"], d["Max"]] for mech, d in accuracy_dict.items()}
        
    strategies = [
                    "NOISE",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    with open('individual-robustness_Spring17_full.json', 'r') as file:
        results1 = load(file)
        
    with open('individual-robustness_Fall17_full.json', 'r') as file:
        results2 = load(file)
        
    with open('individual-robustness_Spring19_full.json', 'r') as file:
        results3 = load(file)
        
    with open('individual-robustness_Fall19_full.json', 'r') as file:
        results4 = load(file)
        
    with open('individual-robustness_Spring17_parametric-bias_correct_false.json', 'r') as file:
        results1p = load(file)
        
    with open('individual-robustness_Fall17_parametric-bias_correct_false.json', 'r') as file:
        results2p = load(file)
        
    with open('individual-robustness_Spring19_parametric-bias_correct_false.json', 'r') as file:
        results3p = load(file)
        
    with open('individual-robustness_Fall19_parametric-bias_correct_false.json', 'r') as file:
        results4p = load(file)
        
    for name, formatted_name in mechanism_name_map.items():
        if name == "DMI: 4" or name[:2] == "SC":
            continue
        m_list = []
        for strategy in strategies:
            if name in ["MSE_P: 0", "Phi-DIV_P: CHI_SQUARED", "Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER", "Phi-DIV_P: TVD"]:
                v1 = results1p[strategy]["1"][name]["Mean Gain"]
                v2 = results2p[strategy]["1"][name]["Mean Gain"]
                v3 = results3p[strategy]["1"][name]["Mean Gain"]
                v4 = results4p[strategy]["1"][name]["Mean Gain"]
            else:
                v1 = results1[strategy]["1"][name]["Mean Gain"]
                v2 = results2[strategy]["1"][name]["Mean Gain"]
                v3 = results3[strategy]["1"][name]["Mean Gain"]
                v4 = results4[strategy]["1"][name]["Mean Gain"]
            m_list += [v1, v2, v3, v4]
        s17 = mean([m_list[0], m_list[4], m_list[8], m_list[12]])
        f17 = mean([m_list[1], m_list[5], m_list[9], m_list[13]])
        s19 = mean([m_list[2], m_list[6], m_list[10], m_list[14]])
        f19 = mean([m_list[3], m_list[7], m_list[11], m_list[15]])
        means = [s17, f17, s19, f19]
        val = mean(means)
        mini = min(means)
        maxi = max(means)
        coordinate_dict[formatted_name][0] = -1*val # Since robustness value is negated for the figure (i.e. higher coordinate should mean more robust)
        coordinate_dict[formatted_name][2] = -1*mini
        coordinate_dict[formatted_name][1] = -1*maxi
        
    data["Accuracy"] = [d[3] for k, d in coordinate_dict.items()]
    data["Robustness"] = [d[0] for k, d in coordinate_dict.items()]
    
    data["ymin"] = [d[4] for k, d in coordinate_dict.items()]
    data["ymax"] = [d[5] for k, d in coordinate_dict.items()]
    
    data["xmin"] = [d[1] for k, d in coordinate_dict.items()]
    data["xmax"] = [d[2] for k, d in coordinate_dict.items()]
    
    """
    # This code block can be uncommented to set up (with a few additional adjustments below) 
    # an analogous plot considering only a single strategy, e.g. PRIOR as below, 
    # instead of aggregating performance over all considered strategies.
    
    robustness_list_prior = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        value = -1*json_data["PRIOR"]["50"][key]["Mean Gain"]
        robustness_list_prior.append(value)
        
    data["Robustness (Prior)"] = robustness_list_prior
    """
    
    ax = plt.gca()
    for x, y, xmin, xmax, ymin, ymax in zip(data["Robustness"], data["Accuracy"], data["xmin"], data["xmax"], data["ymin"], data["ymax"]):
        xlen = xmax - xmin
        ylen = ymax - ymin
        #x_val = 0.5*xlen + xmin
        #y_val = 0.5*ylen + ymin
        el = mpatches.Rectangle((xmin, ymin), xlen, ylen, angle=0, alpha=0.2)
        ax.add_artist(el)
    
    _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Theoretical Guarantee", palette=guarantee_color_map, data=data)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # order = [0,4,2,3,5,1] # When Informed Truthful is included (6 labels)
    order=[0,3,1,2] # When Informed Truthful is not included
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=r'Equilibrium Concept', loc='upper right', bbox_to_anchor=(1.005, 1.02))
    
    
    #ax.errorbar(data["Robustness"], data["Accuracy"], yerr=[data["ymin"], data["ymax"]], xerr=[data["xmin"], data["xmax"]], fmt='')
    
    # label points on the plot
    for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
        if s == r'MSE':
            x_val = x - 1.45
            y_val = y - 0.03
            
        elif s == r'MSE$_P$':
            x_val = x - 1.5
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div$_P$: $H^2$':
            x_val = x - 3.5
            y_val = y + 0.02
            
        elif s == r'$\Phi$-Div$_P$: KL':
            x_val = x - 4
            y_val = y - 0.025
            
        elif s == r'$\Phi$-Div: KL':
            x_val = x - 3.25
            y_val = y - 0.035
            
        elif s == r'$\Phi$-Div: $H^2$':
            x_val = x - 3.25
            y_val = y + 0.0225
        
        elif s == r'$\Phi$-Div$_P$: TVD':
            x_val = x - 3.5
            y_val = y + 0.02
            
        elif s == r'$\Phi$-Div: TVD':
            x_val = x - 3.5
            y_val = y + 0.0175
            
        elif s == r'$\Phi$-Div$_P$: $\chi^2$':
            x_val = x - 3.0
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div: $\chi^2$':
            x_val = x - 3.5
            y_val = y + 0.0175
            
        elif s == r'PTS':
            x_val = x - 1.4
            y_val = y + 0.015
            
        elif s == r'OA':
            x_val = x - 1
            y_val = y + 0.015
            
        else:
            x_val = x - 1
            y_val = y + 0.015
            
        plt.text(x = x_val, # x-coordinate position of data label
        y = y_val, # y-coordinate position of data label
        s = s, # data label
        color = 'black') # set colour of line     

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_facecolor('gainsboro')
    #print(data["ymax"])
    ax.set_xlim(min(data["xmin"]), max(data["xmax"]))
    ax.set_ylim(min(data["ymin"]), max(data["ymax"]))
    
    #plt.legend(bbox_to_anchor=(1.5,0), loc="upper left")
    
    ax.set_ylabel(r'Measurement Integrity')
    ax.set_xlabel(r'Robustness Against Strategic Reporting')
    plt.title(r'Apparent Trade-off Between Integrity and Robustness (Real Data)')
    plt.tight_layout()
    filename = "2D-tradeoff-real-updated-strategic"
    figure_file = "figures/main_paper/" + filename + ".pdf"
    # figure_file = "figures/poster/" + filename + ".png"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()

def figure_2():
    
    filename = 'payments-vs-mse_mse-p'
    json_file = filename + '.json'
    pdf_file = 'main_paper/mi_mse_metrics_with_bias_no-dmi_updated'
    # pdf_file = 'poster/mi_mse_metrics_with_bias_no-dmi_updated'
    with open(json_file,"r") as file:
        more_data = load(file)
    
    filename = 'payments-vs-mse_with-bias-in-model'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        data = load(file)
        for key in data.keys():
            if key in more_data.keys():
                data[key] = more_data[key]
    plot_mi_mse_metrics_highlighted_no_dmi(data, pdf_file)

def figures_3_and_F_1a():
    for semester in ["Spring17", "Fall17", "Spring19", "Fall19"]:
        name = ' 1'.join(semester.split('1'))
        filename = f'payments-vs-mse_{semester}_full'
        json_file = filename + '.json'
        main_file = f'main_paper/mi_mse_metrics_{semester}_full'
        appendix_file = f'appendix/mi_mse_metrics_{semester}_full'
        with open(json_file, "r") as file:
            data = load(file)
        plot_mi_mse_tau_real_data(data, name, main_file)
        plot_mi_mse_other_metrics_real_data(data, name, appendix_file)
        
def figures_4_and_F_2a_and_F_2b():
   
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    
    main_file = f'main_paper/{filename}-mean_gain'
    appendix_file = f'appendix/{filename}-mean_gain'
    with open(json_file,"r") as file:
        data = load(file)
    filename = 'incentives_for_deviating-ce-bias-parametric-bias_correct_false'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            data[strategy][num].update(d[strategy][num])    
    
    plot_mean_rank_changes(data, main_file, main=True, appendix=False)
    
    plot_mean_rank_changes(data, appendix_file, main=False, appendix=True)
    
    plot_variance_rank_changes(data, appendix_file)

def figures_5_and_F_2d_and_F_2e():
    f1 = 'individual-robustness_Spring17'
    f2 = 'individual-robustness_Fall17'
    f3 = 'individual-robustness_Spring19'
    f4 = 'individual-robustness_Fall19'
    j1 = f1 + '_full.json'
    j2 = f2 + '_full.json'
    j3 = f3 + '_full.json'
    j4 = f4 + '_full.json'
    j1p = f1 + '_parametric-bias_correct_false.json'
    j2p = f2 + '_parametric-bias_correct_false.json'
    j3p = f3 + '_parametric-bias_correct_false.json'
    j4p = f4 + '_parametric-bias_correct_false.json'
    
    
    
    main_file = f'main_paper/incentives-for-deviating-real-bias-correct'
    appendix_file = f'appendix/incentives-for-deviating-real-bias-correct'
    with open(j1, "r") as file:
        d1 = load(file)
    with open(j1p,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            d1[strategy][num].update(d[strategy][num])
        
    with open(j2, "r") as file:
        d2 = load(file)
    with open(j2p,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            d2[strategy][num].update(d[strategy][num])
            
    with open(j3, "r") as file:
        d3 = load(file)
    with open(j3p,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            d3[strategy][num].update(d[strategy][num])
            
    with open(j4, "r") as file:
        d4 = load(file)
    with open(j4p,"r") as file:
        d = load(file)
    for strategy in d.keys():
        for num in d[strategy].keys():
            d4[strategy][num].update(d[strategy][num])
        
    plot_mean_rank_changes_real_data(d1, d2, d3, d4, main_file, main=True, appendix=False)
    
    plot_mean_rank_changes_real_data(d1, d2, d3, d4, appendix_file, main=False, appendix=True)
    
    plot_variance_rank_changes_real_data(d1, d2, d3, d4, appendix_file)

def figure_D_1a():
    old_names = {value:key for key,value in mechanism_name_map.items()}
    old_names.update(
            {y:x for x, y in {
                "AMSE_P: 0": r'AMSE$_P$',
                "CORR: 0": r'CORR',
                "MCC: 0": r'MCC',
                "Phi-DIV_P*: CHI_SQUARED": r'$\Phi$-Div$_{P}^*$: $\chi^2$',
                "Phi-DIV_P*: KL": r'$\Phi$-Div$_{P}^*$: KL',
                "Phi-DIV_P*: SQUARED_HELLINGER": r'$\Phi$-Div$_{P}^*$: $H^2$', 
                "Phi-DIV_P*: TVD": r'$\Phi$-Div$_{P}^*$: TVD',
                "R2: 0": r'$R^2$',
                }.items()
            }
        )
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD',
                    # Extensions
                    r'AMSE$_P$',
                    r'CORR',
                    r'MCC',
                    r'$\Phi$-Div$_{P}^*$: $\chi^2$',
                    r'$\Phi$-Div$_{P}^*$: KL',
                    r'$\Phi$-Div$_{P}^*$: $H^2$', 
                    r'$\Phi$-Div$_{P}^*$: TVD',
                    r'$R^2$'
                    ],
                "Mechanism Status":
                [
                    # Non-Parametric Mechanisms
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    # Parametric Mechanisms
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    r'Established Mechanism',
                    # Extensions
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism',
                    r'Novel Mechanism'
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            "Strongly Truthful": 4,
            "Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
            
    filename = 'be-no_bias-extensions'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
    
    be_accuracy_list = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [json_data[str(num)][key]["Mean ROC-AUC"] for num in range(10, 100, 10)]
        value = 2*mean(values) - 1 # Transform to correlation function
        be_accuracy_list.append(value)
        
    data["Accuracy"] = be_accuracy_list
    
    robustness_list = []
    strategies = [
                    "NOISE",
                    "FIX-BIAS",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    json_data = {}
        
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'incentives_for_deviating-ce-bias-extensions'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for strat in strategies:
            for num in range(10, 100, 10):
                json_data[strat][str(num)].update(d[strat][str(num)])
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [-1*json_data[strategy][str(num)][key]["Mean Gain"] for strategy in strategies for num in range(10, 100, 10)]
        value = mean(values)
        robustness_list.append(value)
        
    data["Robustness"] = robustness_list
    
    _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Mechanism Status", data=data)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    order=[0,1] 
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9)
    
    # label points on the plot
    for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
        if s == r'$\Phi$-Div: $H^2$' or s == r'MCC' or s == r'AMSE$_P$':
            x_val = x - 2
            y_val = y - 0.05
            
        elif s == r'$\Phi$-Div$_P$: TVD':
            x_val = x - 6.5
            y_val = y + 0.03
            
        elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_{P}^*$: KL' or s == r'$\Phi$-Div$_{P}^*$: TVD':
            x_val = x - 2.5
            y_val = y + 0.03
            
        elif s == r'$\Phi$-Div$_{P}^*$: $\chi^2$':
            x_val = x - 8.5
            y_val = y - 0.01
            
        elif s == r'$\Phi$-Div$_P$: $\chi^2$' or s == r'$\Phi$-Div$_{P}^*$: $H^2$':
            x_val = x - 2.5
            y_val = y - 0.05
            
        else:
            x_val = x - 1.5
            y_val = y + 0.03
            
        plt.text(x = x_val, # x-coordinate position of data label
        y = y_val, # y-coordinate position of data label
        s = s, # data label
        color = 'black') # set colour of line
    
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_facecolor('gainsboro')
    
    ax.set_ylabel(r'(Coarse Ordinal) Measurement Integrity')
    ax.set_xlabel(r'Robustness Against Strategic Reporting')
    plt.title(r'Apparent Trade-off Between Integrity and Robustness')
    plt.tight_layout()
    filename = "2D-extensions"
    figure_file = "figures/appendix/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def figure_D_1a_bias_correction():
    old_names = {value:key for key,value in mechanism_name_map.items()}
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    #r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    #r'Dominantly Truthful',
                    r'Truthful',             
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',  
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'$\epsilon$-Strongly Truthful',  
                    r'$\epsilon$-Strongly Truthful',
                    r'$\epsilon$-Strongly Truthful', 
                    r'$\epsilon$-Strongly Truthful',
                ],
            "Mechanism Type":
                [
                    # Non-Parametric Mechanisms
                    r'Non-Parametric',
                    #r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    r'Non-Parametric',
                    # Parametric Mechanisms
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                    r'Parametric',
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            r'$\epsilon$-Strongly Truthful': 4,
            #"Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'payments-vs-mse_with-bias-in-model'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d)
        
    json_data.pop("DMI: 4")
    
    filename = 'payments-vs-mse_parametric' 
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
    json_data.update(d) # Update to bias-corrected values
            
    be_accuracy_list = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [mean(json_data[key][str(num)]["Taus"]) for num in range(1, 16)]
        value = mean(values) # transform to correlation function
        be_accuracy_list.append(value)
        
    data["Accuracy"] = be_accuracy_list
    
    robustness_list = []
    strategies = [
                    "NOISE",
                    "FIX-BIAS",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    json_data = {}
        
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [-1*json_data[strategy][str(num)][key]["Mean Gain"] for strategy in strategies for num in range(10, 100, 10)]
        value = mean(values)
        robustness_list.append(value)
        
    data["Robustness"] = robustness_list
    
    _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Theoretical Guarantee", palette=guarantee_color_map, data=data)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # order = [0,4,2,3,5,1] # When Informed Truthful is included (6 labels)
    order=[0,3,1,2] # When Informed Truthful is not included
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=r'Equilibrium Concept', loc='upper center')
    
    # label points on the plot
    for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
        if s == r'MSE$_P$':
            x_val = x + 1
            y_val = y - 0.005
            
        elif s == r'MSE':
            x_val = x - 1.5
            y_val = y - 0.0375
            
        elif s == r'PTS':
            x_val = x - 1.35
            y_val = y + 0.015
            
        elif s == r'OA':
            x_val = x - 0.8
            y_val = y + 0.02
            
        elif s == r'$\Phi$-Div: $H^2$':
            x_val = x - 2
            y_val = y - 0.035
            
        elif s == r'$\Phi$-Div: KL':
            x_val = x - 2
            y_val = y + 0.02
            
        elif s == r'$\Phi$-Div$_P$: TVD':
            x_val = x - 6.5
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_P$: $\chi^2$' or s == r'$\Phi$-Div$_P$: $H^2$':
            x_val = x - 4
            y_val = y + 0.025
            
        else:
            x_val = x - 3.5
            y_val = y + 0.02
            
        plt.text(x = x_val, # x-coordinate position of data label
        y = y_val, # y-coordinate position of data label
        s = s, # data label
        color = 'black') # set colour of line
    
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_facecolor('gainsboro')
    
    ax.set_ylabel(r'Ordinal Measurement Integrity')
    ax.set_xlabel(r'Robustness Against Strategic Reporting')
    plt.title(r'Apparent Trade-off Between Integrity and Robustness (with Parametric Bias Correction)')
    plt.tight_layout()
    filename = "2D-tradeoff-bias-correction"
    figure_file = "figures/main_paper/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()

def figure_E_1a():
    
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    #plot_mean_aucc(data, pdf_file)

    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            data[num].update(d[num])
    #plot_mean_aucc(data, pdf_file)

    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            data[num].update(d[num])
    #plot_mean_aucc(data, pdf_file)
    
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            data[num].update(d[num])
                 
    plot_mean_aucc(data, pdf_file)
    
def figure_E_1a_no_dmi():
    
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    #plot_mean_aucc(data, pdf_file)

    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            data[num].update(d[num])
    #plot_mean_aucc(data, pdf_file)

    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            no_dmi = d[num]
            no_dmi.pop('DMI: 4')
            data[num].update(no_dmi)
    #plot_mean_aucc(data, pdf_file)
    
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename + '-no_dmi'
    with open(json_file,"r") as file:
        d = load(file)
        for num in data.keys():
            data[num].update(d[num])
                 
    plot_mean_aucc(data, pdf_file)
    
def figure_E_1b():
    filename = 'be-bias-all'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_mean_aucc(data, pdf_file)
    
def figure_E_1b_no_dmi():
    filename = 'be-bias-all'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename + '-no_dmi'
    with open(json_file,"r") as file:
        data = load(file)
    for num in data.keys():
        data[num].pop('DMI: 4')
    plot_mean_aucc(data, pdf_file)
    
def figure_E_1c():
    filename = 'ce-bias-all'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    for num in data.keys():
        data[num].pop('DMI: 4')
    plot_kendall_tau(data, pdf_file)
    
def figure_E_1c_no_dmi():
    filename = 'ce-bias-all'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename + '-no_dmi'
    with open(json_file,"r") as file:
        data = load(file)
    for num in data.keys():
        data[num].pop('DMI: 4')
    plot_kendall_tau(data, pdf_file)
    
def figure_E_2d():
    filename = 'strategic-ce-bias'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_taus(data, pdf_file)
    
def figure_E_2d_no_dmi():
    filename = 'strategic-ce-bias'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename + '-no_dmi'
    
    with open(json_file,"r") as file:
        data = load(file)
    for strategy in data.keys():
        for num in data[strategy].keys():
            data[strategy][num].pop('DMI: 4')
    plot_kendall_taus(data, pdf_file)
    
def figure_F_2c():
    filename = 'truthful_vs_strategic_payments-ce-bias'
    json_file = filename + '.json'
    
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_strategic(data, pdf_file)

def figure_F_3a():
    filename = 'true-grade-recovery_no-bias'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_estimation_mses(data, pdf_file)
    
    filename = 'true-grade-recovery'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_estimation_mses(data, pdf_file)

if __name__ == "__main__":
    """Uncomment a function below to create a .pdf of the associated figure from the paper."""

    # figure_1a(True)

    figure_1a(False)
    
    figure_1b()
    
    figure_2()
    
    # figures_3_and_F_1a()
    
    # figures_4_and_F_2a_and_F_2b()
    
    # figures_5_and_F_2d_and_F_2e()
    
    # figure_D_1a()
    
    # figure_D_1a_bias_correction()
    
    # figure_E_1a()
    
    # figure_E_1b()
    
    # figure_E_1c()
    
    # figure_E_1d()
    
    # figure_E_2f()
    
    # figure_F_2c()
    
    # figure_F_3a()
    
    # print(dumps(compare_taus(), indent=2))
    
    # metric = "Binary AUCs"
    # metric = "Taus"
    # metric = "Rhos"
    # print(metric)
    
    # d = compare_metrics(metric)
    # l = [(key, val["Real"]) for key, val in d.items()]
    # print(sorted(l, key=lambda x: x[1], reverse=True))