"""
Functions that plot the results from the simulated experiments.

@author: Noah Burrell <burrelln@umich.edu>
"""
import seaborn as sns
import matplotlib 
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean

"""
Global dictionaries used to store LaTeX formatting for labels used in plots.
"""

mechanism_name_map = {
        
            #Non-Parametric Mechanisms
            "BASELINE: MSE": r'MSE',
            "DMI: 4": r'DMI',
            "OA: 0": r'OA',
            "Phi-DIV: CHI_SQUARED": r'$\Phi$-Div: $\chi^2$',
            "Phi-DIV: KL": r'$\Phi$-Div: KL',
            "Phi-DIV: SQUARED_HELLINGER": r'$\Phi$-Div: $H^2$',
            "Phi-DIV: TVD": r'$\Phi$-Div: TVD',
            "PTS: 0": r'PTS',
            #Parametric Mechanisms
            "MSE_P: 0": r'MSE$_P$',
            "Phi-DIV_P: CHI_SQUARED": r'$\Phi$-Div$_P$: $\chi^2$',
            "Phi-DIV_P: KL": r'$\Phi$-Div$_P$: KL',
            "Phi-DIV_P: SQUARED_HELLINGER": r'$\Phi$-Div$_P$: $H^2$', 
            "Phi-DIV_P: TVD": r'$\Phi$-Div$_P$: TVD',
            #Spot Checking Mechanisms
            "SC: 10": r'Spot Checking (10\%)',
            "SC: 15": r'Spot Checking (15\%)',
            "SC: 20": r'Spot Checking (20\%)',
            "SC: 25": r'Spot Checking (25\%)',
            "SC: 50": r'Spot Checking (50\%)',
            "SC: 75": r'Spot Checking (75\%)',
            
            
    }

mechanism_color_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': '#a6cee3',
            r'DMI': '#d9ef8b', 
            r'OA': '#b15928',
            r'$\Phi$-Div: $\chi^2$': '#b2df8a',
            r'$\Phi$-Div: KL': '#fb9a99',
            r'$\Phi$-Div: $H^2$': '#cab2d6',
            r'$\Phi$-Div: TVD': '#fdbf6f',
            r'PTS': '#01665e',
            #Parametric Mechanisms
            r'MSE$_P$': '#1f78b4',
            r'$\Phi$-Div$_P$: $\chi^2$': '#33a02c',
            r'$\Phi$-Div$_P$: KL': '#e31a1c',
            r'$\Phi$-Div$_P$: $H^2$': '#6a3d9a', 
            r'$\Phi$-Div$_P$: TVD': '#ff7f00',
            #Spot Checking Mechanisms
            r'Spot Checking (10\%)':'black',
            r'Spot Checking (15\%)':'black',
            r'Spot Checking (20\%)':'black',
            r'Spot Checking (25\%)':'black',
            r'Spot Checking (50\%)':'black',
            r'Spot Checking (75\%)':'black',
            
    }

mechanism_marker_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': 'o',
            r'DMI': 'o', 
            r'OA': 'o',
            r'$\Phi$-Div: $\chi^2$': 'o',
            r'$\Phi$-Div: KL': 'o',
            r'$\Phi$-Div: $H^2$': 'o',
            r'$\Phi$-Div: TVD': 'o',
            r'PTS': 'o',
            #Parametric Mechanisms
            r'MSE$_P$': 'P',
            r'$\Phi$-Div$_P$: $\chi^2$': 'P',
            r'$\Phi$-Div$_P$: KL': 'P',
            r'$\Phi$-Div$_P$: $H^2$': 'P', 
            r'$\Phi$-Div$_P$: TVD': 'P',
            #Spot Checking Mechanisms
            r'Spot Checking (10\%)': '*',
            r'Spot Checking (15\%)': '*',
            r'Spot Checking (20\%)': '*',
            r'Spot Checking (25\%)': '*',
            r'Spot Checking (50\%)': '*',
            r'Spot Checking (75\%)': '*',
            
    }

mechanism_dash_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': (1, 1),
            r'DMI': (1, 1), 
            r'OA': (1, 1), 
            r'$\Phi$-Div: $\chi^2$': (1, 1), 
            r'$\Phi$-Div: KL': (1, 1), 
            r'$\Phi$-Div: $H^2$': (1, 1), 
            r'$\Phi$-Div: TVD': (1, 1), 
            r'PTS': (1, 1), 
            #Parametric Mechanisms
            r'MSE$_P$': (5, 5), 
            r'$\Phi$-Div$_P$: $\chi^2$': (5, 5), 
            r'$\Phi$-Div$_P$: KL': (5, 5),
            r'$\Phi$-Div$_P$: $H^2$': (5, 5),
            r'$\Phi$-Div$_P$: TVD': (5, 5),
            #Spot Checking Mechanisms
            r'Spot Checking (10\%)': (1, 1),
            r'Spot Checking (15\%)': (1, 1),
            r'Spot Checking (20\%)': (1, 1),
            r'Spot Checking (25\%)': (1, 1),
            r'Spot Checking (50\%)': (1, 1),
            r'Spot Checking (75\%)': (1, 1),
            
    }

strategy_color_map = {
            "NOISE": '#999999',
            "FIX-BIAS": '#f781bf',
            "MERGE": '#a65628',
            "PRIOR": '#ffff33', 
            "ALL10": '#ff7f00',
            "HEDGE": '#984ea3'
    }

strategy_marker_map = {
            "NOISE": 's',
            "FIX-BIAS": 's',
            "MERGE": 's',
            "PRIOR": 's', 
            "ALL10": 's',
            "HEDGE": 's'
    }

strategy_dash_map = {
            "NOISE": (3, 1, 1, 1, 1, 1),
            "FIX-BIAS": (3, 1, 1, 1, 1, 1),
            "MERGE": (3, 1, 1, 1, 1, 1),
            "PRIOR": (3, 1, 1, 1, 1, 1), 
            "ALL10": (3, 1, 1, 1, 1, 1),
            "HEDGE": (3, 1, 1, 1, 1, 1)
    }

estimation_procedure_map = {
            "Consensus-Grade": r'Consensus Grade',
            "Procedure": r'Procedure',
            "Procedure-NB": r'Procedure-NB'
    }

estimation_procedure_color_map = {
            r'Consensus Grade': '#b2df8a',
            r'Procedure': '#1f78b4',
            r'Procedure-NB': '#a6cee3'
    }

def plot_mean_aucc(results, filename):
    """
    Used for Binary Effort setting. Generates a lineplot of the mean AUCC as the number of active graders varies.

    Parameters
    ----------
    results : dict.
        { num: { mechanism: { "Median ROC-AUC": score } } },
        where num is the number of active graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Number of Active Graders": [], 
                         "AUCC": [],
                         "Mechanism": []
                         }
    
    final_y_vals = []
    record_val = False
    for key in results.keys():
        mechanisms = list(results[key].keys())
        if key == max(results.keys()):
            record_val = True
        for mechanism in mechanisms:
            val = 2*results[key][mechanism]["Mean ROC-AUC"] - 1
            formatted_results["Number of Active Graders"].append(key)
            formatted_results["AUCC"].append(val)
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            if record_val:
                final_y_vals.append(val)
            
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.lineplot(x="Number of Active Graders", y="AUCC", hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map)
    
    order = list(range(len(mechanisms)))
    order.sort(key = lambda i: final_y_vals[i], reverse=True)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_auc_strategic(results, filename):
    """
    Used with strategic agents to compare payments between strategic and truthful agents. 
    For each mechanism, generates boxplots of the AUC scores as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num: { mechanism: { "ROC-AUC Scores": [ score ] } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and each score in the list is a float.
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "AUC": [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            if int(key) in [10, 30, 50, 70, 90]:
                mechanisms = list(results[strategy][key].keys())
                for mechanism in mechanisms:
                    for s in results[strategy][key][mechanism]["ROC-AUC Scores"]:
                        formatted_results["Number of Strategic Graders"].append(key)
                        formatted_results["AUC"].append(s)
                        formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                        formatted_results["Strategy"].append(strategy)
                    
    results_df = pd.DataFrame(data=formatted_results)
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.boxplot(x="Number of Strategic Graders", y="AUC", hue="Strategy", palette=strategy_color_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
    
def plot_estimation_mses(results, filename):
    """
    Used for true grade recovery tests with the various estimation procedures. Generates a boxplot of the MSE scores for each procedure.

    Parameters
    ----------
    results : dict.
        { procedure: { "MSE Scores": [ score ] } },
        where procedure is the name of the procedure (str, one of the keys of the global estimation_procedure_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global estimation_procedure_map
    
    formatted_results = {
                         "MSE": [],
                         "Estimation Method": []
                         }
    
    mechanisms = list(results.keys())
    for mechanism in mechanisms:
        for m in results[mechanism]["MSE Scores"]:
            formatted_results["MSE"].append(m)
            formatted_results["Estimation Method"].append(estimation_procedure_map[mechanism])
        
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.boxplot(x="Estimation Method", y="MSE", palette=estimation_procedure_color_map, data=results_df)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_kendall_tau(results, filename):
    """
    Used for Continuous Effort setting. Generates a plot of the Kendall rank correlation coefficient (tau_B) scores of each mechanism.

    Parameters
    ----------
    results : dict.
        { mechanism: { "Tau Scores": [ score ] } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Number of Assignments Per Semester": [], 
                         "Tau": [],
                         "Mechanism": []
                         }
    
    final_y_vals = []
    record_val = False
    for key in results.keys():
        mechanisms = list(results[key].keys())
        if key == max(results.keys()):
            record_val = True
        for mechanism in mechanisms:
            val = mean(results[key][mechanism]["Tau Scores"])
            formatted_results["Number of Assignments Per Semester"].append(key)
            formatted_results["Tau"].append(val)
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            if record_val:
                final_y_vals.append(val)
            
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.lineplot(x="Number of Assignments Per Semester", y="Tau", hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map)
    
    order = list(range(len(mechanisms)))
    order.sort(key = lambda i: final_y_vals[i], reverse=True)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.0, 1.0))
    
    _.set_ylabel(r'$\tau_B$')
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_kendall_taus(results, filename):
    """
    Used with strategic agents to compare mechanism performance as the number of strategic agents varies. 
    For each strategy, generates a plot of the average tau scores of each mechanism as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Tau Scores": [ score ] } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and each score in the list is a float.
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    strategies = []
    
    strategies = list(results.keys())
    for strategy in results.keys():
        formatted_results = {"Number of Strategic Graders": [], 
                             "Tau": [],
                             "Mechanism": [],
                             }
        final_y_vals = []
        record_val = False
        for key in results[strategy].keys():
            mechanisms = list(results[strategy][key].keys())
            if key == max(results[strategy].keys()):
                record_val = True
            for mechanism in mechanisms:
                val = mean(results[strategy][key][mechanism]["Tau Scores"])
                formatted_results["Number of Strategic Graders"].append(key)
                formatted_results["Tau"].append(val)
                formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                if record_val:
                    final_y_vals.append(val)
            
        title = strategy
        results_df = pd.DataFrame(data=formatted_results)
        _ = sns.lineplot(x="Number of Strategic Graders", y="Tau", hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map)
        
        order = list(range(len(mechanisms)))
        order.sort(key = lambda i: final_y_vals[i], reverse=True)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.0, 1.0))
        
        _.set_ylabel(r'$\tau_B$')
        
        plt.title(title)
        plt.tight_layout()
        figure_file = f"figures/{filename}-{strategy}.pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
    
def plot_mean_rank_changes(results, filename, main=False, appendix=False):
    """
    Used with strategic agents to see how agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the average gain in payment rank of a single deviating agent as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Mean Gain": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the mean rank gain (float).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Mean Rank Gain": [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            
            if main:
                results[strategy][key].pop("DMI: 4")
                results[strategy][key].pop("Phi-DIV: CHI_SQUARED")
                results[strategy][key].pop("Phi-DIV: KL")
                results[strategy][key].pop("Phi-DIV: SQUARED_HELLINGER")
                results[strategy][key].pop("Phi-DIV: TVD")
                results[strategy][key].pop("PTS: 0")
                results[strategy][key].pop("Phi-DIV_P: CHI_SQUARED")
                results[strategy][key].pop("Phi-DIV_P: TVD")
            if appendix:
                results[strategy][key].pop("BASELINE: MSE")
                results[strategy][key].pop("OA: 0")
                results[strategy][key].pop("MSE_P: 0")
                results[strategy][key].pop("Phi-DIV_P: KL")
                results[strategy][key].pop("Phi-DIV_P: SQUARED_HELLINGER")
            
            mechanisms = list(results[strategy][key].keys())    
            
            for mechanism in mechanisms:
                formatted_results["Number of Strategic Graders"].append(key)
                formatted_results["Mean Rank Gain"].append(results[strategy][key][mechanism]["Mean Gain"])
                formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                formatted_results["Strategy"].append(strategy)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.lineplot(x="Number of Strategic Graders", y="Mean Rank Gain", hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_variance_rank_changes(results, filename):
    """
    Used with strategic agents to see the variance of how much agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the variance of the gain in payment rank of a single deviating agent as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Rank Gain Variance": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the variance of the rank gain (float).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Rank Gain Variance" : [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            mechanisms = list(results[strategy][key].keys())
            for mechanism in mechanisms:
                formatted_results["Number of Strategic Graders"].append(key)
                formatted_results["Rank Gain Variance"].append(results[strategy][key][mechanism]["Variance Gain"])
                formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                formatted_results["Strategy"].append(strategy)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.lineplot(x="Number of Strategic Graders", y="Rank Gain Variance", hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_payments_vs_mse(results, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each mechanism, generates a scatterplot of that relationship.

    Parameters
    ----------
    results : dict.
        { mechanism: [ (payment, mse) ] },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and each double in the list is two floats.
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = list(results.keys())
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        result_list = results[mechanism]
        payments, mses = list(map(list, zip(*result_list)))
        _ = sns.scatterplot(x=payments, y=mses)
        
        ax = plt.gca()
        ax.set_ylabel(r'MSE of Reports')
        ax.set_xlabel(r'Reward Assigned by Mechnanism')
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_metrics(results, filename):
    """
    Plots the releationship between number of assignments and average metric value for each mechanism.
    For each metric, generates a lineplot of that relationship.

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Binary AUCs'
            - 'Quinary AUCs'
            - 'Taus'
            - 'Rhos'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    mechanisms = list(results.keys())
    
    formatted_results = {"Mean AUC (Binary)": [], 
                         "Mean AUC (Quinary)": [],
                         "Mean Tau": [],
                         "Mean Rho": [], 
                         "Mechanism": [],
                         "Number of Assignments": []
                         }
    
    
    for mechanism in mechanisms:
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            
            formatted_results["Mean AUC (Binary)"].append(mean(metrics['Binary AUCs']))
            formatted_results["Mean AUC (Quinary)"].append(mean(metrics['Quinary AUCs']))
            formatted_results["Mean Tau"].append(mean(metrics['Taus']))
            formatted_results["Mean Rho"].append(mean(metrics['Rhos']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean AUC (Binary)": r'Mean AUC (Binary)', 
        "Mean AUC (Quinary)": r'Mean AUC (Quinary)', 
        "Mean Tau": r'Mean Rank Correlation ($\tau_B$)',
        "Mean Rho": r'Mean Correlation ($\rho$)'
               }
    
    metrics_code = {
        "Mean AUC (Binary)": "bAUC", 
        "Mean AUC (Quinary)": "qAUC",
        "Mean Tau": "tau",
        "Mean Rho": "rho",
               }
    
    for metric, name in metrics_formatted.items():
        title = name
        
        _ = sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_metrics_highlighted(results, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each metric, generates a scatterplot of that relationship.
    
    USED TO REPLICATE A FIGURE FROM THE PAPER

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Binary AUCs'
            - 'Quinary AUCs'
            - 'Taus'
            - 'Rhos'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    #Parametric Mechanisms
    #results.pop("BASELINE: MSE")
    #results.pop("DMI: 4")
    #results.pop("OA: 0")
    #results.pop("Phi-DIV: CHI_SQUARED")
    #results.pop("Phi-DIV: KL")
    #results.pop("Phi-DIV: SQUARED_HELLINGER")
    #results.pop("Phi-DIV: TVD")
    #results.pop("PTS: 0")
    #Parametric Mechanisms
    #results.pop("MSE_P: 0")
    #results.pop("Phi-DIV_P: CHI_SQUARED")
    #results.pop("Phi-DIV_P: KL")
    #results.pop("Phi-DIV_P: SQUARED_HELLINGER")
    #results.pop("Phi-DIV_P: TVD")
    
    mechanisms = list(results.keys())
    
    formatted_results = {"Mean AUC (Binary)": [], 
                         "Mean AUC (Quinary)": [],
                         "Mean Tau": [],
                         "Mean Rho": [], 
                         "Mechanism": [],
                         "Number of Assignments": [],
                         "Size": []
                         }
    
    
    for mechanism in mechanisms:
        size = 1 
        if mechanism in ("Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER"):
            size = 2
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            formatted_results["Mean AUC (Binary)"].append(mean(metrics['Binary AUCs']))
            formatted_results["Mean AUC (Quinary)"].append(mean(metrics['Quinary AUCs']))
            formatted_results["Mean Tau"].append(mean(metrics['Taus']))
            formatted_results["Mean Rho"].append(mean(metrics['Rhos']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments)
            formatted_results["Size"].append(size)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean AUC (Binary)": r'Mean AUC (Binary)', 
        "Mean AUC (Quinary)": r'Mean AUC (Quinary)',
        "Mean Tau": r'Mean Rank Correlation ($\tau_B$)',
        "Mean Rho": r'Mean Correlation ($\rho$)'
               }
    
    metrics_y = {
        "Mean AUC (Binary)": r'AUC', 
        "Mean AUC (Quinary)": r'AUC',
        "Mean Tau": r'$\tau_B$',
        "Mean Rho": r'$\rho$'
               }
    
    metrics_code = {
        "Mean AUC (Binary)": "bAUC", 
        "Mean AUC (Quinary)": "qAUC",
        "Mean Tau": "tau",
        "Mean Rho": "rho",
               }
    
    mechanisms_formatted = [mechanism_name_map[m] for m in mechanisms]
    for i, m in enumerate(mechanisms_formatted):
        if m in (r'$\Phi$-Div$_P$: KL', r'$\Phi$-Div$_P$: $H^2$'):
            mechanisms_formatted[i] = m
    for metric, name in metrics_formatted.items():
        title = name
        
        fig, ax = plt.subplots()
        
        df1 = results_df.loc[((results_df['Mechanism'] != r'$\Phi$-Div$_P$: KL') & (results_df['Mechanism'] !=r'$\Phi$-Div$_P$: $H^2$'))]
        df2 = results_df.loc[((results_df['Mechanism'] == r'$\Phi$-Div$_P$: KL') | (results_df['Mechanism'] ==r'$\Phi$-Div$_P$: $H^2$'))]
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df1, palette=mechanism_color_map, size="Size",sizes=(1,1), legend=None, ax=ax)
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df2, palette=mechanism_color_map, size="Size",sizes=(3,3), markersize=10, legend=None, ax=ax)
        
        for line, mname in zip(ax.lines, mechanisms_formatted):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1] + 0.25
            if mname == r'OA' and metric == "Mean Rho":
                y = line.get_ydata()[-1] - 0.025
            	
            if not np.isfinite(y):
            	    y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
            	    continue
            text = ax.annotate(mname,
            		       xy=(x, y),
            		       xytext=(0, 0),
            		       color=line.get_color(),
            		       xycoords=(ax.get_xaxis_transform(),
            				 ax.get_yaxis_transform()),
            		       textcoords="offset points")
            text_width = (text.get_window_extent(
            	fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
            		ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
                    
        #ax.set_xticks([2,4,6,8,10,12,14])
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        
        plt.title(title)
        ax.set_ylabel(metrics_y[metric])
        ax.set_xlabel(r'Number of Assignments Per Semester')
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_metrics_highlighted_no_dmi(results, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each metric, generates a scatterplot of that relationship.
    
    USED TO REPLICATE A FIGURE FROM THE PAPER

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Binary AUCs'
            - 'Quinary AUCs'
            - 'Taus'
            - 'Rhos'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    #Parametric Mechanisms
    #results.pop("BASELINE: MSE")
    results.pop("DMI: 4")
    #results.pop("OA: 0")
    #results.pop("Phi-DIV: CHI_SQUARED")
    #results.pop("Phi-DIV: KL")
    #results.pop("Phi-DIV: SQUARED_HELLINGER")
    #results.pop("Phi-DIV: TVD")
    #results.pop("PTS: 0")
    #Parametric Mechanisms
    #results.pop("MSE_P: 0")
    #results.pop("Phi-DIV_P: CHI_SQUARED")
    #results.pop("Phi-DIV_P: KL")
    #results.pop("Phi-DIV_P: SQUARED_HELLINGER")
    #results.pop("Phi-DIV_P: TVD")
    
    mechanisms = list(results.keys())
    
    formatted_results = {"Mean AUC (Binary)": [], 
                         "Mean AUC (Quinary)": [],
                         "Mean Tau": [],
                         "Mean Rho": [], 
                         "Mechanism": [],
                         "Number of Assignments": [],
                         "Size": []
                         }
    
    
    for mechanism in mechanisms:
        size = 1 
        if mechanism in ("Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER"):
            size = 2
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            formatted_results["Mean AUC (Binary)"].append(2*mean(metrics['Binary AUCs']) - 1) # Transform to correlation function
            formatted_results["Mean AUC (Quinary)"].append(mean(metrics['Quinary AUCs']))
            formatted_results["Mean Tau"].append(mean(metrics['Taus']))
            formatted_results["Mean Rho"].append(mean(metrics['Rhos']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments)
            formatted_results["Size"].append(size)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean AUC (Binary)": r'Coarse Ordinal Measurement Integrity', 
        #"Mean AUC (Quinary)": r'Mean AUC (Quinary)',
        "Mean Tau": r'Fine Ordinal Measurement Integrity',
        "Mean Rho": r'Interval Measurement Integrity'
               }
    
    metrics_y = {
        "Mean AUC (Binary)": r'AUCC', 
        #"Mean AUC (Quinary)": r'AUC',
        "Mean Tau": r'$\tau_B$',
        "Mean Rho": r'$\rho$'
               }
    
    metrics_code = {
        "Mean AUC (Binary)": "bAUC", 
        #"Mean AUC (Quinary)": "qAUC",
        "Mean Tau": "tau",
        "Mean Rho": "rho",
               }
    
    mechanisms_formatted = [
            
                #Non-Parametric Mechanisms
                r'MSE',
                r'OA',
                r'MSE$_P$',
                #r'DMI',
                r'$\Phi$-Div: $\chi^2$',
                r'$\Phi$-Div: KL',
                r'$\Phi$-Div: $H^2$',
                r'$\Phi$-Div: TVD',
                r'PTS',
                r'$\Phi$-Div$_P$: $\chi^2$',
                r'$\Phi$-Div$_P$: TVD',
                r'$\Phi$-Div$_P$: KL',
                r'$\Phi$-Div$_P$: $H^2$', 
                
        ]
    
    for metric, name in metrics_formatted.items():
        title = name
        
        fig, ax = plt.subplots()
        
        df1 = results_df.loc[((results_df['Mechanism'] != r'$\Phi$-Div$_P$: KL') & (results_df['Mechanism'] !=r'$\Phi$-Div$_P$: $H^2$'))]
        df2 = results_df.loc[((results_df['Mechanism'] == r'$\Phi$-Div$_P$: KL') | (results_df['Mechanism'] ==r'$\Phi$-Div$_P$: $H^2$'))]
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df1, palette=mechanism_color_map, size="Size",sizes=(1,1), legend=None, ax=ax)
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df2, palette=mechanism_color_map, size="Size",sizes=(3,3), markersize=10, legend=None, ax=ax)
        
        for line, mname in zip(ax.lines, mechanisms_formatted):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1] + 0.25
            
            if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.005
            if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] + 0.005
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.0125
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                #print("BH", y)
                y = line.get_ydata()[-1] + 0.0075
            if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Binary)":
                #print("BK", y)
                y = line.get_ydata()[-1] - 0.0075
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                #print("BC", y)
                y = line.get_ydata()[-1] - 0.0175
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.0075
            
                
            if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.005
            if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] + 0.005
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.0075
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                #print("QH", y)
                y = line.get_ydata()[-1] - 0.01
            if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Quinary)":
                #print("QK", y)
                y = line.get_ydata()[-1] + 0.005
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                #print("QC", y)
                y = line.get_ydata()[-1] - 0.0225
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.0075
                
                
            if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] - 0.005
            if mname == mechanism_name_map["OA: 0"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] + 0.005
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] - 0.0075
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                #print("TH", y)
                y = line.get_ydata()[-1] - 0.02
            if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Tau":
                #print("TK", y)
                y = line.get_ydata()[-1] + 0.0075
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Tau":
                #print("TC", y)
                y = line.get_ydata()[-1] - 0.025
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] - 0.01
                
            if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] + 0.0075
            if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] - 0.0275
            if mname == mechanism_name_map["OA: 0"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] - 0.0025
            if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] - 0.0025
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                #print("RH", y)
                y = line.get_ydata()[-1] - 0.0225
            if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Rho":
                #print("RK", y)
                y = line.get_ydata()[-1] + 0.0125
            if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] - 0.01
            if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Rho":
                #print("RPH", y)
                y = line.get_ydata()[-1] - 0.0225
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Rho":
                #print("RC", y)
                y = line.get_ydata()[-1] - 0.0025
            	
            if not np.isfinite(y):
            	    y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
            	    continue
            text = ax.annotate(mname,
            		       xy=(x, y),
            		       xytext=(0, 0),
            		       color=line.get_color(),
            		       xycoords=(ax.get_xaxis_transform(),
            				 ax.get_yaxis_transform()),
            		       textcoords="offset points")
            text_width = (text.get_window_extent(
            	fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
            		ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
                    
        #ax.set_xticks([2,4,6,8,10,12,14])
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        
        plt.title(title)
        ax.set_ylabel(metrics_y[metric])
        ax.set_xlabel(r'Number of Assignments Per Semester')
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        # figure_file = "figures/" + filename + "-" + metrics_code[metric] +".png"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_metrics_other(results, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each metric, generates a scatterplot of that relationship.
    
    USED TO REPLICATE A FIGURE FROM THE PAPER

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Binary AUCs'
            - 'Quinary AUCs'
            - 'Taus'
            - 'Rhos'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    ###Parametric Mechanisms
    results.pop("BASELINE: MSE")
    #results.pop("DMI: 4")
    #results.pop("OA: 0")
    #results.pop("Phi-DIV: CHI_SQUARED")
    #results.pop("Phi-DIV: KL")
    #results.pop("Phi-DIV: SQUARED_HELLINGER")
    #results.pop("Phi-DIV: TVD")
    #results.pop("PTS: 0")
    ###Parametric Mechanisms
    results.pop("MSE_P: 0")
    #results.pop("Phi-DIV_P: CHI_SQUARED")
    results.pop("Phi-DIV_P: KL")
    results.pop("Phi-DIV_P: SQUARED_HELLINGER")
    #results.pop("Phi-DIV_P: TVD")
    
    mechanisms = list(results.keys())
    
    formatted_results = {"Mean AUC (Binary)": [], 
                         "Mean AUC (Quinary)": [],
                         "Mean Tau": [],
                         "Mean Rho": [], 
                         "Mechanism": [],
                         "Number of Assignments": [],
                         "Size": []
                         }
    
    for mechanism in mechanisms:
        size = 1 
        if mechanism in ("Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER"):
            size = 2
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            formatted_results["Mean AUC (Binary)"].append(mean(metrics['Binary AUCs']))
            formatted_results["Mean AUC (Quinary)"].append(mean(metrics['Quinary AUCs']))
            formatted_results["Mean Tau"].append(mean(metrics['Taus']))
            formatted_results["Mean Rho"].append(mean(metrics['Rhos']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments)
            formatted_results["Size"].append(size)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean AUC (Binary)": r'Mean AUC (Binary)', 
        "Mean AUC (Quinary)": r'Mean AUC (Quinary)',
        "Mean Tau": r'Mean Rank Correlation ($\tau_B$)',
        "Mean Rho": r'Mean Correlation ($\rho$)'
               }
    
    metrics_y = {
        "Mean AUC (Binary)": r'AUC', 
        "Mean AUC (Quinary)": r'AUC',
        "Mean Tau": r'$\tau_B$',
        "Mean Rho": r'$\rho$'
               }
    
    metrics_code = {
        "Mean AUC (Binary)": "bAUC", 
        "Mean AUC (Quinary)": "qAUC",
        "Mean Tau": "tau",
        "Mean Rho": "rho",
               }
    
    mechanisms_formatted = [mechanism_name_map[m] for m in mechanisms]
    for i, m in enumerate(mechanisms_formatted):
        if m in (r'$\Phi$-Div$_P$: KL', r'$\Phi$-Div$_P$: $H^2$'):
            mechanisms_formatted[i] = m
    for metric, name in metrics_formatted.items():
        title = name
        
        fig, ax = plt.subplots()
        
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map, size="Size",sizes=(1,1), legend=None, ax=ax)
        
        for line, mname in zip(ax.lines, mechanisms_formatted):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1] + 0.25
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.0125
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] + 0.015
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.015
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                y = line.get_ydata()[-1] - 0.0025
                
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.0075
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] + 0.0125
            if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.01
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                y = line.get_ydata()[-1] - 0.0025
                
            if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] - 0.005
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] + 0.02
            if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                y = line.get_ydata()[-1] - 0.01
                
            if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                y = line.get_ydata()[-1] + 0.03
            	
            if not np.isfinite(y):
            	    y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
            	    continue
            text = ax.annotate(mname,
            		       xy=(x, y),
            		       xytext=(0, 0),
            		       color=line.get_color(),
            		       xycoords=(ax.get_xaxis_transform(),
            				 ax.get_yaxis_transform()),
            		       textcoords="offset points")
            text_width = (text.get_window_extent(
            	fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
            		ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
                    
        #ax.set_xticks([2,4,6,8,10,12,14])
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        
        plt.title(title)
        ax.set_ylabel(metrics_y[metric])
        ax.set_xlabel(r'Number of Assignments Per Semester')
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_other_metrics_real_data(results, semester, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each metric, generates a scatterplot of that relationship.
    
    USED TO REPLICATE A FIGURE FROM THE PAPER

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Binary AUCs'
            - 'Quinary AUCs'
            - 'Rhos'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    #Parametric Mechanisms
    #results.pop("BASELINE: MSE")
    #results.pop("DMI: 4")
    #results.pop("OA: 0")
    #results.pop("Phi-DIV: CHI_SQUARED")
    #results.pop("Phi-DIV: KL")
    #results.pop("Phi-DIV: SQUARED_HELLINGER")
    #results.pop("Phi-DIV: TVD")
    #results.pop("PTS: 0")
    #Parametric Mechanisms
    #results.pop("MSE_P: 0")
    #results.pop("Phi-DIV_P: CHI_SQUARED")
    #results.pop("Phi-DIV_P: KL")
    #results.pop("Phi-DIV_P: SQUARED_HELLINGER")
    #results.pop("Phi-DIV_P: TVD")
    
    mechanisms = list(results.keys())
    
    formatted_results = {"Mean AUC (Binary)": [], 
                         "Mean AUC (Quinary)": [],
                         "Mean Rho": [], 
                         "Mechanism": [],
                         "Number of Assignments": [],
                         "Size": []
                         }
    
    
    for mechanism in mechanisms:
        size = 1 
        if mechanism in ("Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER"):
            size = 2
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            formatted_results["Mean AUC (Binary)"].append(2*mean(metrics['Binary AUCs']) - 1) # Transform to correlation function
            formatted_results["Mean AUC (Quinary)"].append(mean(metrics['Quinary AUCs']))
            formatted_results["Mean Rho"].append(mean(metrics['Rhos']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments) 
            formatted_results["Size"].append(size)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean AUC (Binary)": r'Coarse Ordinal Measurement Integrity', 
        #"Mean AUC (Quinary)": r'Mean AUC (Quinary)',
        "Mean Rho": r'Interval Measurement Integrity'
               }
    
    metrics_y = {
        "Mean AUC (Binary)": r'AUCC', 
        #"Mean AUC (Quinary)": r'AUC',
        "Mean Rho": r'$\rho$'
               }
    
    metrics_code = {
        "Mean AUC (Binary)": "bAUC", 
        #"Mean AUC (Quinary)": "qAUC",
        "Mean Rho": "rho",
               }
    
    semester_map = {
            "Spring 17": r'Spring 2017: ',
            "Fall 17": r'Fall 2017: ',
            "Spring 19": r'Spring 2019: ',
            "Fall 19": r'Fall 2019: '
        }
    
    mechanisms_formatted = [
            
                #Non-Parametric Mechanisms
                r'MSE',
                r'OA',
                r'$\Phi$-Div: $\chi^2$',
                r'$\Phi$-Div: KL',
                r'$\Phi$-Div: $H^2$',
                r'$\Phi$-Div: TVD',
                r'PTS',
                r'MSE$_P$',
                r'$\Phi$-Div$_P$: $\chi^2$',
                r'$\Phi$-Div$_P$: TVD',
                r'$\Phi$-Div$_P$: KL',
                r'$\Phi$-Div$_P$: $H^2$', 
                
        ]
    
    for metric, name in metrics_formatted.items():
        title = semester_map[semester] + name
        
        fig, ax = plt.subplots()
        
        df1 = results_df.loc[((results_df['Mechanism'] != r'$\Phi$-Div$_P$: KL') & (results_df['Mechanism'] !=r'$\Phi$-Div$_P$: $H^2$'))]
        df2 = results_df.loc[((results_df['Mechanism'] == r'$\Phi$-Div$_P$: KL') | (results_df['Mechanism'] ==r'$\Phi$-Div$_P$: $H^2$'))]
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df1, palette=mechanism_color_map, size="Size",sizes=(1,1), legend=None, ax=ax)
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df2, palette=mechanism_color_map, size="Size",sizes=(3,3), markersize=10, legend=None, ax=ax)
        
        for line, mname in zip(ax.lines, mechanisms_formatted):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1] + 0.1
            
            if semester == "Spring 17":    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Binary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Binary)":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Binary)":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.03    
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.02  
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Binary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] + 0.0225
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.0175
                    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Quinary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.0125
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Quinary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("PH", y)
                    y = line.get_ydata()[-1] - 0.000
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Quinary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.0125
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.01
                
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Rho":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.0325
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Rho":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.03
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.03
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.0125
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("PH:", y)
                    y = line.get_ydata()[-1] - 0.05
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Rho":
                    print("PKL:", y)
                    y = line.get_ydata()[-1] + 0.0675
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("H:", y)
                    y = line.get_ydata()[-1] - 0.0175
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Rho":
                    print("KL:", y)
                    y = line.get_ydata()[-1] + 0.03
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.0125
            
            if semester == "Fall 17":    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Binary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Binary)":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Binary)":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Binary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.02  
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Binary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.015
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Binary)":
                    print("TVD", y)
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.005
                    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Quinary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Quinary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.000
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("PH", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Quinary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.0025
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                    
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Rho":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.035
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Rho":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Rho":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.025
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Rho":
                    print("PC:", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Rho":
                    print("C:", y)
                    y = line.get_ydata()[-1] - 0.05
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Rho":
                    print("PTVD:", y)
                    y = line.get_ydata()[-1] - 0.05
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.0125
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("PH:", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Rho":
                    print("PKL:", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("H:", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Rho":
                    print("KL:", y)
                    y = line.get_ydata()[-1] + 0.055
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.0125
                
            
            if semester == "Spring 19":    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Binary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Binary)":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.0025
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Binary)":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Binary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    print("C", y)
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Binary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.04
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    print("H", y)
                    y = line.get_ydata()[-1] - 0.0325
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Binary)":
                    print("TVD", y)
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.005
                    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Quinary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Quinary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("PH", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Quinary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("H", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.0075
                    
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Rho":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Rho":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Rho":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Rho":
                    print("PC:", y)
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Rho":
                    print("C:", y)
                    y = line.get_ydata()[-1] - 0.035
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Rho":
                    print("PTVD:", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("PH:", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Rho":
                    print("PKL:", y)
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("H:", y)
                    y = line.get_ydata()[-1] - 0.06
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Rho":
                    print("KL:", y)
                    y = line.get_ydata()[-1] - 0.08
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] + 0.0125
            
            if semester == "Fall 19":    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Binary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Binary)":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Binary)":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.0025
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.0025
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Binary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    print("PH", y)
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Binary)":
                    print("C", y)
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Binary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] + 0.0075
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Binary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.025
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Binary)":
                    print("TVD", y)
                    y = line.get_ydata()[-1] + 0.015
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Binary)":
                    y = line.get_ydata()[-1] + 0.00
                    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean AUC (Quinary)":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.0175
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    print("PC", y)
                    y = line.get_ydata()[-1] - 0.025
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean AUC (Quinary)":
                    print("PKL", y)
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("PH", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean AUC (Quinary)":
                    print("PTVD", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean AUC (Quinary)":
                    print("C", y)
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean AUC (Quinary)":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0025
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean AUC (Quinary)":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.015
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] + 0.0025
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean AUC (Quinary)":
                    y = line.get_ydata()[-1] - 0.005
                    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Rho":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.08
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Rho":
                    print("PC", y)
                    y = line.get_ydata()[-1] - 0.04
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Rho":
                    print("PKL", y)
                    y = line.get_ydata()[-1] + 0.025
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("PH", y)
                    y = line.get_ydata()[-1] - 0.0375
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Rho":
                    print("PTVD", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Rho":
                    print("C", y)
                    y = line.get_ydata()[-1] + 0.0075
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Rho":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0325
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Rho":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.02
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Rho":
                    y = line.get_ydata()[-1] - 0.01
    
            if not np.isfinite(y):
            	    y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
            	    continue
            text = ax.annotate(mname,
            		       xy=(x, y),
            		       xytext=(0, 0),
            		       color=line.get_color(),
            		       xycoords=(ax.get_xaxis_transform(),
            				 ax.get_yaxis_transform()),
            		       textcoords="offset points")
            text_width = (text.get_window_extent(
            	fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
            		ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
                    
        #ax.set_xticks([2,4,6,8,10,12,14])
        ax.set_xticks([1, 2, 3, 4])
        
        plt.title(title) 
        ax.set_ylabel(metrics_y[metric])
        ax.set_xlabel(r'Number of Assignment Blocks Graded')
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mi_mse_tau_real_data(results, semester, filename):
    """
    Used to visually inspect relationship between Student payments and the MSE of their reports. 
    For each metric, generates a scatterplot of that relationship.
    
    USED TO REPLICATE A FIGURE FROM THE PAPER

    Parameters
    ----------
    results : dict.
        { mechanism: { num_assignments: { metric : [ scores ] } } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), num_assignments is an int,
        metric is one of 
            - 'Taus'
            
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    
    global mechanism_name_map
    
    #Parametric Mechanisms
    #results.pop("BASELINE: MSE")
    #results.pop("DMI: 4")
    #results.pop("OA: 0")
    #results.pop("Phi-DIV: CHI_SQUARED")
    #results.pop("Phi-DIV: KL")
    #results.pop("Phi-DIV: SQUARED_HELLINGER")
    #results.pop("Phi-DIV: TVD")
    #results.pop("PTS: 0")
    #Parametric Mechanisms
    #results.pop("MSE_P: 0")
    #results.pop("Phi-DIV_P: CHI_SQUARED")
    #results.pop("Phi-DIV_P: KL")
    #results.pop("Phi-DIV_P: SQUARED_HELLINGER")
    #results.pop("Phi-DIV_P: TVD")
    
    mechanisms = list(results.keys())
    
    formatted_results = {
                         "Mean Tau": [],
                         "Mechanism": [],
                         "Number of Assignments": [],
                         "Size": []
                         }
    
    
    for mechanism in mechanisms:
        size = 1 
        if mechanism in ("Phi-DIV_P: KL", "Phi-DIV_P: SQUARED_HELLINGER"):
            size = 2
        for num, metrics in results[mechanism].items():
            num_assignments = int(num)
            formatted_results["Mean Tau"].append(mean(metrics['Taus']))
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            formatted_results["Number of Assignments"].append(num_assignments)
            formatted_results["Size"].append(size)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    metrics_formatted = {
        "Mean Tau": r'Fine Ordinal Measurement Integrity',
               }
    
    metrics_y = {
        "Mean Tau": r'$\tau_B$',
               }
    
    metrics_code = {
        "Mean Tau": "tau",
               }
    
    semester_map = {
            "Spring 17": r'Spring 2017: ',
            "Fall 17": r'Fall 2017: ',
            "Spring 19": r'Spring 2019: ',
            "Fall 19": r'Fall 2019: '
        }
    
    mechanisms_formatted = [
            
                #Non-Parametric Mechanisms
                r'MSE',
                r'OA',
                r'$\Phi$-Div: $\chi^2$',
                r'$\Phi$-Div: KL',
                r'$\Phi$-Div: $H^2$',
                r'$\Phi$-Div: TVD',
                r'PTS',
                r'MSE$_P$',
                r'$\Phi$-Div$_P$: $\chi^2$',
                r'$\Phi$-Div$_P$: TVD',
                r'$\Phi$-Div$_P$: KL',
                r'$\Phi$-Div$_P$: $H^2$', 
                
        ]
    
    for metric, name in metrics_formatted.items():
        title = semester_map[semester] + name
        
        fig, ax = plt.subplots()
        
        df1 = results_df.loc[((results_df['Mechanism'] != r'$\Phi$-Div$_P$: KL') & (results_df['Mechanism'] !=r'$\Phi$-Div$_P$: $H^2$'))]
        df2 = results_df.loc[((results_df['Mechanism'] == r'$\Phi$-Div$_P$: KL') | (results_df['Mechanism'] ==r'$\Phi$-Div$_P$: $H^2$'))]
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df1, palette=mechanism_color_map, size="Size",sizes=(1,1), legend=None, ax=ax)
        sns.lineplot(x="Number of Assignments", y=metric, hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=df2, palette=mechanism_color_map, size="Size",sizes=(3,3), markersize=10, legend=None, ax=ax)
        
        for line, mname in zip(ax.lines, mechanisms_formatted):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1] + 0.1
            
            if semester == "Spring 17":    
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Tau":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Tau":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.0225
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.0075
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.0125
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.02
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    print("H:", y)
                    y = line.get_ydata()[-1] + 0.0025
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Tau":
                    print("KL:", y)
                    y = line.get_ydata()[-1] - 0.0125
                    
            
            if semester == "Fall 17":    
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Tau":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.0075
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Tau":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] + 0.025
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Tau":
                    print("C:", y)
                    y = line.get_ydata()[-1] - 0.0175
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.0175
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Tau":
                    print("TVD:", y)
                    y = line.get_ydata()[-1] + 0.0125
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Tau":
                    print("KL:", y)
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.015

            
            if semester == "Spring 19":    
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Tau":
                    print("MSE", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Tau":
                    print("MSE_P", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Tau":
                    print("C:", y)
                    y = line.get_ydata()[-1] - 0.00
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.01
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Tau":
                    print("TVD:", y)
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    print("PH:", y)
                    y = line.get_ydata()[-1] - 0.02
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.00
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    print("H:", y)
                    y = line.get_ydata()[-1] + 0.0125
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Tau":
                    print("KL:", y)
                    y = line.get_ydata()[-1] - 0.0025
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.00
                    
                    
            if semester == "Fall 19":    
                if mname == mechanism_name_map["OA: 0"] and metric == "Mean Tau":
                    print("OA", y)
                    y = line.get_ydata()[-1] - 0.0225
                if mname == mechanism_name_map["BASELINE: MSE"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["MSE_P: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                if mname == mechanism_name_map["Phi-DIV_P: CHI_SQUARED"] and metric == "Mean Tau":
                    print("PC", y)
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["Phi-DIV_P: KL"] and metric == "Mean Tau":
                    print("PKL", y)
                    y = line.get_ydata()[-1] - 0.015
                if mname == mechanism_name_map["Phi-DIV_P: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    print("PH", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV_P: TVD"] and metric == "Mean Tau":
                    print("PTVD", y)
                    y = line.get_ydata()[-1] - 0.04
                if mname == mechanism_name_map["Phi-DIV: CHI_SQUARED"] and metric == "Mean Tau":
                    print("C", y)
                    y = line.get_ydata()[-1] + 0.01
                if mname == mechanism_name_map["Phi-DIV: KL"] and metric == "Mean Tau":
                    print("KL", y)
                    y = line.get_ydata()[-1] - 0.0125
                if mname == mechanism_name_map["Phi-DIV: SQUARED_HELLINGER"] and metric == "Mean Tau":
                    print("H", y)
                    y = line.get_ydata()[-1] + 0.015
                if mname == mechanism_name_map["Phi-DIV: TVD"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] + 0.005
                if mname == mechanism_name_map["PTS: 0"] and metric == "Mean Tau":
                    y = line.get_ydata()[-1] - 0.005
                    
    
            if not np.isfinite(y):
            	    y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
            	    continue
            text = ax.annotate(mname,
            		       xy=(x, y),
            		       xytext=(0, 0),
            		       color=line.get_color(),
            		       xycoords=(ax.get_xaxis_transform(),
            				 ax.get_yaxis_transform()),
            		       textcoords="offset points")
            text_width = (text.get_window_extent(
            	fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            if np.isfinite(text_width):
            		ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
                    
        #ax.set_xticks([2,4,6,8,10,12,14])
        ax.set_xticks([1, 2, 3, 4])
        
        plt.title(title) 
        ax.set_ylabel(metrics_y[metric])
        ax.set_xlabel(r'Number of Assignment Blocks Graded')
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + metrics_code[metric] + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_mean_rank_changes_real_data(results1, results2, results3, results4, filename, main=False, appendix=False):
    """
    Used with strategic agents to see how agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the average gain in payment rank of a single deviating agent for each semester.

    Parameters
    ----------
    resultsX : dict.
        { strategy: { num : { mechanism: { "Mean Gain": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the mean rank gain (float).
        The four results files here should correspond to results from the four semesters of real grading data (in chronological order).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).
    main, appendix : bool, optional
        Indicates whether certain mechanisms should be excluded (for replicating figures from the paper exactly).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    all_results = [("Spring 2017", results1), ("Fall 2017", results2), ("Spring 2019", results3), ("Fall 2019", results4)]
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Mean Rank Gain": [],
                         "Rank Gain Variance": [],
                         "Mechanism": [],
                         "Strategy": [],
                         "Semester": []
                         }
    
    for semester, results in all_results:
        for strategy in results.keys():
            for key in results[strategy].keys():
                if main:
                    results[strategy][key].pop("Phi-DIV: CHI_SQUARED")
                    results[strategy][key].pop("Phi-DIV: KL")
                    results[strategy][key].pop("Phi-DIV: SQUARED_HELLINGER")
                    results[strategy][key].pop("Phi-DIV: TVD")
                    results[strategy][key].pop("PTS: 0")
                    results[strategy][key].pop("Phi-DIV_P: CHI_SQUARED")
                    results[strategy][key].pop("Phi-DIV_P: TVD")
                if appendix:
                    results[strategy][key].pop("BASELINE: MSE")
                    results[strategy][key].pop("OA: 0")
                    results[strategy][key].pop("MSE_P: 0")
                    results[strategy][key].pop("Phi-DIV_P: KL")
                    results[strategy][key].pop("Phi-DIV_P: SQUARED_HELLINGER")
                
                mechanisms = list(results[strategy][key].keys())    
                for mechanism in mechanisms:
                    formatted_results["Number of Strategic Graders"].append(key)
                    formatted_results["Mean Rank Gain"].append(results[strategy][key][mechanism]["Mean Gain"])
                    formatted_results["Rank Gain Variance"].append(results[strategy][key][mechanism]["Variance Gain"])
                    formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                    formatted_results["Strategy"].append(strategy)
                    formatted_results["Semester"].append(semester)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    for y_val, name in [("Mean Rank Gain", "-gain")]:
        for mechanism in mechanisms:
            title = mechanism_name_map[mechanism]
            mechanism_df = results_df.loc[results_df["Mechanism"] == title]
            _ = sns.lineplot(x="Semester", y=y_val, hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
            
            plt.title(title)
            plt.tight_layout()
            figure_file = "figures/" + filename + "-" + mechanism + name + ".pdf"
            plt.savefig(figure_file, dpi=300)
            plt.show()
            plt.close()
            
def plot_variance_rank_changes_real_data(results1, results2, results3, results4, filename):
    """
    Used with strategic agents to see how agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the variance of the gain in payment rank of a single deviating agent for each semester.

    Parameters
    ----------
    resultsX : dict.
        { strategy: { num : { mechanism: { "Variance Gain": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the mean rank gain (float).
        The four results files here should correspond to results from the four semesters of real grading data (in chronological order).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    all_results = [("Spring 2017", results1), ("Fall 2017", results2), ("Spring 2019", results3), ("Fall 2019", results4)]
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Mean Rank Gain": [],
                         "Rank Gain Variance": [],
                         "Mechanism": [],
                         "Strategy": [],
                         "Semester": []
                         }
    
    for semester, results in all_results:
        for strategy in results.keys():
            for key in results[strategy].keys():
                
                mechanisms = list(results[strategy][key].keys())    
                for mechanism in mechanisms:
                    formatted_results["Number of Strategic Graders"].append(key)
                    formatted_results["Mean Rank Gain"].append(results[strategy][key][mechanism]["Mean Gain"])
                    formatted_results["Rank Gain Variance"].append(results[strategy][key][mechanism]["Variance Gain"])
                    formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                    formatted_results["Strategy"].append(strategy)
                    formatted_results["Semester"].append(semester)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    for y_val, name in [("Rank Gain Variance", "-variance")]:
        for mechanism in mechanisms:
            title = mechanism_name_map[mechanism]
            mechanism_df = results_df.loc[results_df["Mechanism"] == title]
            _ = sns.lineplot(x="Semester", y=y_val, hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
            
            plt.title(title)
            plt.tight_layout()
            figure_file = "figures/" + filename + "-" + mechanism + name + ".pdf"
            plt.savefig(figure_file, dpi=300)
            plt.show()
            plt.close()