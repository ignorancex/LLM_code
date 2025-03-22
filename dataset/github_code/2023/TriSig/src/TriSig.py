import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy.special import betainc, betaincinv
from math import comb
from decimal import *
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TriSig:

    transitions = {}
    p_pre_computed = {}
    ind_transitions = {}
    combined_data = None
    data = None

    def __init__(self, data):
        nr_transitions = len(data[0][data[0].columns[0]])*(len(data)-1)
        self.data = data
        self.combined_data = pd.concat(data).reset_index(drop=True)
        for y_i in self.combined_data.columns:
            self.transitions[y_i] = {}
            vals = self.combined_data[y_i].value_counts().to_dict()
            for key in vals.keys():
                vals[key] = vals[key] / len(self.combined_data[y_i])
            self.p_pre_computed[y_i] = vals

        z = 0
        while z < (len(self.data) - 1):
            z_0 = self.data[z]
            z+=1
            z_1 = self.data[z]
            for y_i in z_0.columns:
                temp_data = pd.concat([z_0[y_i], z_1[y_i]], axis=1)
                temp_data.columns = ["A", "B"]
                temp_data = temp_data.groupby(['A','B']).size().reset_index().rename(columns={0:'count'})
                temp_data = pd.concat([temp_data["A"].astype(str) + temp_data["B"].astype(str), temp_data["count"]], axis=1)

                for val in temp_data[0]:
                    if val not in self.transitions[y_i].keys():
                        self.transitions[y_i][val] = 0
                temp_data.columns = ["keys", "count"]
                vals = pd.Series(temp_data["count"].values,index=temp_data["keys"]).to_dict()
                for key in vals.keys():
                    self.transitions[y_i][key] += vals[key]/nr_transitions

    @staticmethod
    def bin_prob(n, p, k):
        if p == 1 and n == k:
            return 1
        ctx = Context()
        arr = math.factorial(n) // math.factorial(k) // math.factorial(n - k)
        bp = (Decimal(arr) * ctx.power(Decimal(p), Decimal(k)) * ctx.power(Decimal(1 - p), Decimal(n - k)))
        return float(bp) if sys.float_info.min < bp else sys.float_info.min

    @staticmethod
    def binom(n, p, k):
        soma = 0
        for i in range(k, n+1):
            soma += TriSig.bin_prob(n, p, i)
        return soma

    @staticmethod
    def hochberg_critical_value(p_values, false_discovery_rate=0.05):
        ordered_pvalue = sorted(p_values)

        critical_values = []
        for i in range(1,len(ordered_pvalue)+1):
            critical_values.append((i/len(ordered_pvalue)) * false_discovery_rate)

        critical_val = ordered_pvalue[len(ordered_pvalue)-1]
        for i in reversed(range(len(ordered_pvalue))):
            if ordered_pvalue[i] < critical_values[i]:
                critical_val = ordered_pvalue[i]
                break
        if critical_val > 0.05:
            critical_val = 0.05
        return critical_val

    @staticmethod
    def bonferroni_correction(alpha, m):
        return alpha / m

    def y_ind_z_dep(self, tricluster, i_d_d_vars=False, print_statistics=False):
        if print_statistics:
            print("Tricluster("+str(len(tricluster["rows"]))+","+str(len(tricluster["columns"]))+","+str(len(tricluster["times"]))+")")

        p = Decimal(1.0)
        for z_i in tricluster["times"]:
            if print_statistics:
                print("P("+str(z_i)+")")
            for col in tricluster["columns"]:
                y_i = self.data[z_i].columns[col]

                val_yi = self.data[z_i][y_i].iloc[tricluster["rows"]].value_counts().idxmax()
                #print(val_yi_aux)

                #val_yi = self.data[z_i][y_i][tricluster["rows"][0]]
                p = p * Decimal(self.p_pre_computed[y_i][val_yi])
                if print_statistics:
                    print("P(Y) = "+str(self.p_pre_computed[y_i][val_yi]))

        # Adjust p based on number of existing contexts
        p = p * Decimal(comb(len(self.data), len(tricluster["times"])))
        if p > 1:
            p = Decimal(1.0)

        pvalue = Decimal(TriSig.binom(len(self.data[0][self.data[0].columns[0]]), p, len(tricluster["rows"])))
        if i_d_d_vars:
            pvalue = float(pvalue * Decimal(comb(len(self.data[0].columns), len(tricluster["columns"]))))
        if pvalue > 1:
            pvalue = 1

        if print_statistics:
            print("pvalue = "+str(pvalue))

        return float(pvalue)

    def y_dep_z_dep(self, tricluster, i_d_d_vars=False, print_statistics=False):
        #############
        if print_statistics:
            print("Tricluster("+str(len(tricluster["rows"]))+","+str(len(tricluster["columns"]))+","+str(len(tricluster["times"]))+")")
        #############
        filtered_cols_data = self.combined_data[[col for col in self.combined_data.columns[tricluster["columns"]]]]
        combined_data_concatenated = filtered_cols_data[filtered_cols_data.columns[0]].astype(str) + filtered_cols_data[filtered_cols_data.columns[1]].astype(str)
        for i in range(2, len(filtered_cols_data.columns)):
            combined_data_concatenated += filtered_cols_data[filtered_cols_data.columns[i]].astype(str)
        combined_data_concatenated_dict = combined_data_concatenated.value_counts().to_dict()

        p = Decimal(1.0)
        for z_i in tricluster["times"]:
            if print_statistics:
                print("P("+str(z_i)+")")
            y_i = self.data[z_i].columns[tricluster["columns"]]

            val_yi = self.data[z_i][y_i].iloc[tricluster["rows"]].value_counts().idxmax()
            pattern_z = val_yi[0].astype(str) + val_yi[1].astype(str)
            for i in range(2, len(val_yi)):
                pattern_z += val_yi[i].astype(str)

            #val_yi = self.data[z_i][y_i]
            #pattern_z = val_yi[val_yi.columns[0]].astype(str) + val_yi[val_yi.columns[1]].astype(str)
            #for i in range(2, len(val_yi.columns)):
            #    pattern_z += val_yi[val_yi.columns[i]].astype(str)
            #pattern_z = pattern_z[tricluster["rows"][0]]
            temp_p = combined_data_concatenated_dict[pattern_z] / len(combined_data_concatenated)
            p = p * Decimal(temp_p)
            if print_statistics:
                print("P(Y) = "+str(temp_p))

        # Adjust p based on number of existing contexts
        p = p * Decimal(comb(len(self.data), len(tricluster["times"])))
        if p > 1:
            p = Decimal(1.0)

        pvalue = Decimal(TriSig.binom(len(self.data[0][self.data[0].columns[0]]), p, len(tricluster["rows"])))
        if i_d_d_vars:
            pvalue = float(pvalue * Decimal(comb(len(self.data[0].columns), len(tricluster["columns"]))))
        if pvalue > 1:
            pvalue = 1
        if print_statistics:
            print("pvalue = "+str(pvalue))

        return float(pvalue)

    def y_ind_z_ind(self, tricluster, i_d_d_vars=False, print_statistics=False):
        Z = len(self.data)
        Y = len(self.data[0].columns)
        cols = self.data[0].columns
        if print_statistics:
            print("Tricluster("+str(len(tricluster["rows"]))+","+str(len(tricluster["columns"]))+","+str(len(tricluster["times"]))+")")

        filtered_dataframes = [self.data[i] for i in tricluster["times"]]

        p = Decimal(1.0)
        # Para constantes
        for z in range(len(filtered_dataframes)):
            if print_statistics:
                print("Z = "+ str(z))
            p_z = 1
            for col in tricluster["columns"]:
                total_observed_samples = list(filtered_dataframes[z][cols[col]])
                pattern_observed_samples = list(filtered_dataframes[z][cols[col]].iloc[tricluster["rows"]])
                pattern_val = Counter(pattern_observed_samples).most_common(1)[0][0]
                filtered_samples = list(filter(lambda val: val == pattern_val, total_observed_samples))
                # New just for print
                p_y = Decimal((len(filtered_samples)/len(total_observed_samples)))
                if print_statistics:
                    print("P(Y) = "+str(p_y))
                p_z = p_z * p_y
                # Original p = p * Decimal((len(filtered_samples)/len(total_observed_samples)))
                p = p * p_y
            if print_statistics:
                print("P(Z = "+str(z)+")"+str(p_z))

        pvalue = Decimal(TriSig.binom(len(filtered_dataframes[0][filtered_dataframes[0].columns[0]]), p, len(tricluster["rows"])))

        # Adjust p
        if i_d_d_vars:
            pvalue = pvalue * Decimal(comb(Z, len(filtered_dataframes)))
            pvalue = pvalue * Decimal(comb(Y, len(tricluster["columns"])))
        if pvalue > 1:
            pvalue = 1.0
        if print_statistics:
            print("pvalue = "+str(pvalue))

        return float(pvalue)

    def y_ind_z_markov(self, tricluster, i_d_d_vars=False, print_statistics=False):

        if print_statistics:
            print("Tricluster("+str(len(tricluster["rows"]))+","+str(len(tricluster["columns"]))+","+str(len(tricluster["times"]))+")")
        Y = len(self.data[0].columns)

        p = Decimal(1.0)
        for col in tricluster["columns"]:
            y_i = self.data[0].columns[col]
            val = self.data[tricluster["times"][0]][y_i].value_counts().idxmax()
#            val = self.data[tricluster["times"][0]][y_i][tricluster["rows"][0]]
            p = p * Decimal(self.p_pre_computed[self.data[0].columns[col]][val])
            if print_statistics:
                print("P(k1)="+str(self.p_pre_computed[self.data[0].columns[col]][val]))

            z = 0
            while z < (len(tricluster["times"]) - 1):
                z_0 = self.data[tricluster["times"][z]]
                z+=1
                z_1 = self.data[tricluster["times"][z]]

                temp_data = z_0[y_i].astype(str) + z_1[y_i].astype(str)
                prob_antecedent = self.p_pre_computed[y_i][z_0[y_i].value_counts().idxmax()]
                #prob_antecedent = self.p_pre_computed[y_i][z_0[y_i][tricluster["rows"][0]]]
                #Most common transition
                most_t = temp_data.iloc[tricluster["rows"]].value_counts().idxmax()
                t = self.transitions[y_i][most_t]
                #t = self.transitions[y_i][temp_data[tricluster["rows"][0]]]
                if print_statistics:
                    print("P(transition-"+str(z)+")="+str(float((Decimal(t)/Decimal(prob_antecedent)))))
                p = p * (Decimal(t)/Decimal(prob_antecedent))

        # Adjust p for misalignments
        p = p * Decimal((len(self.data)-len(tricluster["times"])+1))
        if p > 1.0:
            p = 1
        pvalue = Decimal(TriSig.binom(len(self.data[0][self.data[0].columns[0]]), p, len(tricluster["rows"])))

        # Adjust p-value if variables idd
        if i_d_d_vars:
            pvalue = pvalue * Decimal(comb(Y, len(tricluster["columns"])))
        if pvalue > 1:
            pvalue = 1

        if print_statistics:
            print("pvalue = "+str(pvalue))

        return float(pvalue)

    def y_dep_z_markov(self, tricluster, i_d_d_vars=False, print_statistics=True):
        if print_statistics:
            print("Tricluster("+str(len(tricluster["rows"]))+","+str(len(tricluster["columns"]))+","+str(len(tricluster["times"]))+")")

        Y = len(self.data[0].columns)

        # P de antecedente
        filtered_cols_data = self.combined_data[[col for col in self.combined_data.columns[tricluster["columns"]]]]
        combined_data_concatenated = filtered_cols_data[filtered_cols_data.columns[0]].astype(str) + filtered_cols_data[filtered_cols_data.columns[1]].astype(str)
        for i in range(2, len(filtered_cols_data.columns)):
            combined_data_concatenated += filtered_cols_data[filtered_cols_data.columns[i]].astype(str)
        combined_data_concatenated_dict = combined_data_concatenated.value_counts().to_dict()

        #most common
        most_t = combined_data_concatenated.iloc[tricluster["rows"]].value_counts().idxmax()
        p = Decimal(combined_data_concatenated_dict[most_t]/len(combined_data_concatenated))
        #p = Decimal(combined_data_concatenated_dict[combined_data_concatenated[tricluster["rows"][0]]]/len(combined_data_concatenated))
        if print_statistics:
            print("P(k1)="+str(p))
        # Create Transitions dictionary
        transitions_combined = pd.Series()
        z = 0
        while z < (len(self.data) - 1):
            z_0 = self.data[z][self.data[0].columns[tricluster["columns"]]]
            z+=1
            z_1 = self.data[z][self.data[0].columns[tricluster["columns"]]]

            z_0_concatenated = z_0[z_0.columns[0]].astype(str) + z_0[z_0.columns[1]].astype(str)
            z_1_concatenated = z_1[z_1.columns[0]].astype(str) + z_1[z_1.columns[1]].astype(str)

            for i in range(2, len(filtered_cols_data.columns)):
                z_0_concatenated += z_0[z_0.columns[i]].astype(str)
                z_1_concatenated += z_1[z_1.columns[i]].astype(str)

            transitions_combined = transitions_combined.append(z_0_concatenated + z_1_concatenated)

        transition_dict = transitions_combined.value_counts().to_dict()

        z = 0
        while z < (len(tricluster["times"]) - 1):
            z_0 = self.data[tricluster["times"][z]][self.data[0].columns[tricluster["columns"]]]
            z+=1
            z_1 = self.data[tricluster["times"][z]][self.data[0].columns[tricluster["columns"]]]

            z_0_concatenated = z_0[z_0.columns[0]].astype(str) + z_0[z_0.columns[1]].astype(str)
            z_1_concatenated = z_1[z_1.columns[0]].astype(str) + z_1[z_1.columns[1]].astype(str)

            for i in range(2, len(filtered_cols_data.columns)):
                z_0_concatenated += z_0[z_0.columns[i]].astype(str)
                z_1_concatenated += z_1[z_1.columns[i]].astype(str)

            # Most common
            most_t_p_transition = (z_0_concatenated+z_1_concatenated).iloc[tricluster["rows"]].value_counts().idxmax()
            p_transition = Decimal(transition_dict[most_t_p_transition])/Decimal(len(transitions_combined))
            # Most common
            most_t_p_ante = z_0_concatenated.iloc[tricluster["rows"]].value_counts().idxmax()
            p_ante = Decimal(combined_data_concatenated_dict[most_t_p_ante])/Decimal(len(combined_data_concatenated))

            #p_transition = Decimal(transition_dict[(z_0_concatenated+z_1_concatenated)[tricluster["rows"][0]]])/Decimal(len(transitions_combined))
            #p_ante = Decimal(combined_data_concatenated_dict[z_0_concatenated[tricluster["rows"][0]]])/Decimal(len(combined_data_concatenated))
            if print_statistics:
                print("P(transition-"+str(z)+")="+str(float(p_transition/p_ante)))
            p = p * p_transition/p_ante

        # Adjust p for misalignments
        p = p * Decimal((len(self.data)-len(tricluster["times"])+1))
        if p > 1.0:
            p = 1
        pvalue = Decimal(TriSig.binom(len(self.data[0][self.data[0].columns[0]]), p, len(tricluster["rows"])))

        # Adjust p-value if variables idd
        if i_d_d_vars:
            pvalue = pvalue * Decimal(comb(Y, len(tricluster["columns"])))
        if pvalue > 1:
            pvalue = 1

        if print_statistics:
            print("pvalue = "+str(pvalue))
        return float(pvalue)



