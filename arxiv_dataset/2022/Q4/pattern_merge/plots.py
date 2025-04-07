import math

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats


directory = os.getcwd()

number_of_variables = [4,8,12,16,20]#,24,28,32,36,40,44,48]

'''
files = [
    str(directory)+"\\output\\4\\accident.txt",
    str(directory)+"\\output\\4\\chess.txt",
    str(directory)+"\\output\\4\\connect.txt",
    str(directory)+"\\output\\4\\mushroom.txt",
    str(directory)+"\\output\\4\\pumsb.txt"
]
file_output_names = [
    str(directory)+"\\output\\4\\accident",
    str(directory)+"\\output\\4\\chess",
    str(directory)+"\\output\\4\\connect",
    str(directory)+"\\output\\4\\mushroom",
    str(directory)+"\\output\\4\\pumsb"
]
'''

files = [
    str(directory)+"\\output\\4\\accident_with_intermediate.txt",
    str(directory)+"\\output\\4\\chess_with_intermediate.txt",
    str(directory)+"\\output\\4\\connect_with_intermediate.txt",
    str(directory)+"\\output\\4\\mushroom_with_intermediate.txt",
    str(directory)+"\\output\\4\\pumsb_with_intermediate.txt"
]
file_output_names = [
    "accident_with_intermediate",
    "chess_with_intermediate",
    "connect_with_intermediate",
    "mushroom_with_intermediate",
    "pumsb_with_intermediate"
]

for i in range(len(files)):
    file = files[i]
    counter = 0

    ##########################

    s_apriori_times = []
    d_apriori_times = []
    r_apriori_times = []

    s_apriori_times_std = []
    d_apriori_times_std = []
    r_apriori_times_std = []

    ###########################

    s_merging_times = []
    d_merging_times = []
    r_merging_times = []

    s_merging_times_std = []
    d_merging_times_std = []
    r_merging_times_std = []

    ############################

    s_tested_subsets = []
    d_tested_subsets = []
    r_tested_subsets = []

    s_tested_subsets_std = []
    d_tested_subsets_std = []
    r_tested_subsets_std = []

    ############################

    s_intersections = []
    d_intersections = []
    r_intersections = []

    s_intersections_std = []
    d_intersections_std = []
    r_intersections_std = []

    ############################

    s_observations_patterns = []
    d_observations_patterns = []
    r_observations_patterns = []

    s_observations_patterns_std = []
    d_observations_patterns_std = []
    r_observations_patterns_std = []

    #############################

    s_variables_patterns = []
    d_variables_patterns = []
    r_variables_patterns = []

    s_variables_patterns_std = []
    d_variables_patterns_std = []
    r_variables_patterns_std = []

    number_of_patterns = []
    number_of_patterns_std = []

    "Number of intersections"
    "Number of variables tested for subsets"
    "Number of observations per pattern"
    "Number of variables per pattern"

    "Number of variables:"
    #vezes 3
    "the time were:"
    "Apriori with clusters"
    "Merging of patterns"
    with open(file) as f:
        current_total = 0
        now = False
        apri = False
        merg = False
        subs = False
        inter = False
        obs = False
        vars = False
        nr_p = False
        for line in f:
            line = line.replace("\n", "")
            if "the time were:" in line:
                now = True
            elif now:
                if "Apriori with clusters" in line:
                    apri = True
                elif apri:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_apriori_times.append(sum(a)/len(a))
                        s_apriori_times_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_apriori_times.append(sum(a)/len(a))
                        d_apriori_times_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_apriori_times.append(sum(a)/len(a))
                        r_apriori_times_std.append(np.std(a, dtype=np.float64))
                    apri = False
                elif "Merging of patterns" in line:
                    merg = True
                elif merg:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_merging_times.append(sum(a)/len(a))
                        s_merging_times_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_merging_times.append(sum(a)/len(a))
                        d_merging_times_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_merging_times.append(sum(a)/len(a))
                        r_merging_times_std.append(np.std(a, dtype=np.float64))
                    merg = False
                elif "Number of intersections" in line:
                    inter = True
                elif inter:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_intersections.append(sum(a)/len(a))
                        s_intersections_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_intersections.append(sum(a)/len(a))
                        d_intersections_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_intersections.append(sum(a)/len(a))
                        r_intersections_std.append(np.std(a, dtype=np.float64))
                    inter = False
                elif "Number of variables tested for subsets" in line:
                    subs = True
                elif subs:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_tested_subsets.append(sum(a)/len(a))
                        s_tested_subsets_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_tested_subsets.append(sum(a)/len(a))
                        d_tested_subsets_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_tested_subsets.append(sum(a)/len(a))
                        r_tested_subsets_std.append(np.std(a, dtype=np.float64))
                    subs = False
                elif "Number of observations per pattern" in line:
                    obs = True
                elif obs:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_observations_patterns.append(sum(a)/len(a))
                        s_observations_patterns_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_observations_patterns.append(sum(a)/len(a))
                        d_observations_patterns_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_observations_patterns.append(sum(a)/len(a))
                        r_observations_patterns_std.append(np.std(a, dtype=np.float64))
                    obs = False
                elif "Number of variables per pattern" in line:
                    vars = True
                elif vars:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        s_variables_patterns.append(sum(a)/len(a))
                        s_variables_patterns_std.append(np.std(a, dtype=np.float64))
                    elif counter == 1:
                        d_variables_patterns.append(sum(a)/len(a))
                        d_variables_patterns_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        r_variables_patterns.append(sum(a)/len(a))
                        r_variables_patterns_std.append(np.std(a, dtype=np.float64))
                    vars = False
                elif "Number of patterns discovered" in line:
                    nr_p = True
                elif nr_p:
                    line = line.replace("[","")
                    line = line.replace("]","")
                    line_splited = line.split(",")
                    a = []
                    for val in line_splited:
                        a.append(float(val))
                    if counter == 0:
                        number_of_patterns.append(sum(a)/len(a))
                        number_of_patterns_std.append(np.std(a, dtype=np.float64))
                    elif counter == 2:
                        counter = -1
                    counter += 1
                    nr_p = False
                    now = False

    ############### Statistical Test ###############

    print(file)
    print("merge(similarity/dissimilarity) : " + str(stats.ttest_rel(s_merging_times, d_merging_times, alternative="less")[1]))
    print("merge(similarity/random) : " + str(stats.ttest_rel(s_merging_times, r_merging_times, alternative="less")[1]))

    print("execution(similarity/dissimilarity) : " + str(stats.ttest_rel(np.add(s_apriori_times, s_merging_times), np.add(d_apriori_times, d_merging_times), alternative="less")[1]))
    print("execution(similarity/random) : " + str(stats.ttest_rel(np.add(s_apriori_times, s_merging_times), np.add(r_apriori_times, r_merging_times), alternative="less")[1]))


    ############### Apriori ###############

    plt.figure()
    plt.plot(number_of_variables, d_apriori_times, color='red', linewidth=1.5, label="Dissimilarity")
    plt.fill_between(number_of_variables, np.array(d_apriori_times)-d_apriori_times_std, np.array(d_apriori_times)+d_apriori_times_std, color="red", alpha=0.2)
    plt.plot(number_of_variables, s_apriori_times, color='green', linewidth=1.5, label="Similarity")
    plt.fill_between(number_of_variables, np.array(s_apriori_times)-s_apriori_times_std, np.array(s_apriori_times)+s_apriori_times_std, color="green", alpha=0.2)
    plt.plot(number_of_variables, r_apriori_times, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(r_apriori_times)-r_apriori_times_std, np.array(r_apriori_times)+r_apriori_times_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')
    #print(d_apriori_times)
    #print(s_apriori_times)
    #print(r_apriori_times)
    #print("###########################################################")

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time of apriori")
    plt.savefig(str(directory)+"\\output\\4\\output\\apriori\\" + file_output_names[i] + "_apriori.png", bbox_inches='tight')
    plt.show()

    ############### Merging ###############

    plt.figure()
    plt.plot(number_of_variables, d_merging_times, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_merging_times, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_merging_times, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(d_merging_times)-d_merging_times_std, np.array(d_merging_times)+d_merging_times_std, color="red", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(s_merging_times)-s_merging_times_std, np.array(s_merging_times)+s_merging_times_std, color="green", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(r_merging_times)-r_merging_times_std, np.array(r_merging_times)+r_merging_times_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time of Merging")
    plt.savefig(str(directory)+"\\output\\4\\output\\merge\\" + file_output_names[i] + "_merge.png", bbox_inches='tight')
    plt.show()

    ############### Combined ###############

    plt.figure()
    plt.plot(number_of_variables, np.add(d_apriori_times, d_merging_times), color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, np.add(s_apriori_times, s_merging_times), color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, np.add(r_apriori_times, r_merging_times), color='yellow', linewidth=1.5, label="Random")

    def standard_deviation_two(x,y):
        final_array=np.zeros(len(x))
        for i in range(len(x)):
            final_array[i] = math.sqrt(pow(x[i],2) + pow(y[i],2))
        return final_array

    plt.fill_between(number_of_variables,
                     np.add(d_apriori_times, d_merging_times)-standard_deviation_two(d_apriori_times_std,d_merging_times_std),
                     np.add(d_apriori_times, d_merging_times)+standard_deviation_two(d_apriori_times_std,d_merging_times_std)
                     , color="red", alpha=0.2)
    plt.fill_between(number_of_variables,
                     np.add(s_apriori_times, s_merging_times)-standard_deviation_two(s_apriori_times_std,s_merging_times_std),
                     np.add(s_apriori_times, s_merging_times)+standard_deviation_two(s_apriori_times_std,s_merging_times_std),
                     color="green", alpha=0.2)
    plt.fill_between(number_of_variables,
                     np.add(r_apriori_times, r_merging_times)-standard_deviation_two(r_apriori_times_std,r_merging_times_std),
                     np.add(r_apriori_times, r_merging_times)+standard_deviation_two(r_apriori_times_std,r_merging_times_std),
                     color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Execution time (Seconds)")
    plt.title("Execution time combined")
    plt.savefig(str(directory)+"\\output\\4\\output\\combined\\" + file_output_names[i] + "_combined.png", bbox_inches='tight')
    plt.show()

    ############### Intersections ###############

    plt.figure()
    plt.plot(number_of_variables, d_intersections, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_intersections, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_intersections, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(d_intersections)-d_intersections_std, np.array(d_intersections)+d_intersections_std, color="red", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(s_intersections)-s_intersections_std, np.array(s_intersections)+s_intersections_std, color="green", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(r_intersections)-r_intersections_std, np.array(r_intersections)+r_intersections_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Number of intersections")
    plt.title("Number of intersections during merging")
    plt.savefig(str(directory)+"\\output\\4\\output\\intersections\\" + file_output_names[i] + "_intersections.png", bbox_inches='tight')
    plt.show()

    ############### Tested variables ###############

    plt.figure()
    plt.plot(number_of_variables, d_tested_subsets, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_tested_subsets, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_tested_subsets, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(d_tested_subsets)-d_tested_subsets_std, np.array(d_tested_subsets)+d_tested_subsets_std, color="red", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(s_tested_subsets)-s_tested_subsets_std, np.array(s_tested_subsets)+s_tested_subsets_std, color="green", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(r_tested_subsets)-r_tested_subsets_std, np.array(r_tested_subsets)+r_tested_subsets_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Number of subsets tested")
    plt.title("Number of subsets tested during merging")
    plt.savefig(str(directory)+"\\output\\4\\output\\subsets_vars\\" + file_output_names[i] + "_subset_variable.png", bbox_inches='tight')
    plt.show()

    ############### Observations per pattern ###############

    plt.figure()
    plt.plot(number_of_variables, d_observations_patterns, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_observations_patterns, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_observations_patterns, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(d_observations_patterns)-d_observations_patterns_std, np.array(d_observations_patterns)+d_observations_patterns_std, color="red", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(s_observations_patterns)-s_observations_patterns_std, np.array(s_observations_patterns)+s_observations_patterns_std, color="green", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(r_observations_patterns)-r_observations_patterns_std, np.array(r_observations_patterns)+r_observations_patterns_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Number of observations per pattern")
    plt.title("Number of observations per pattern before merge")
    plt.savefig(str(directory)+"\\output\\4\\output\\observations_per_pattern\\" + file_output_names[i] + "_observations.png", bbox_inches='tight')
    plt.show()

    ############### variables per pattern ###############

    plt.figure()
    plt.plot(number_of_variables, d_variables_patterns, color='red', linewidth=1.5, label="Dissimilarity")
    plt.plot(number_of_variables, s_variables_patterns, color='green', linewidth=1.5, label="Similarity")
    plt.plot(number_of_variables, r_variables_patterns, color='yellow', linewidth=1.5, label="Random")
    plt.fill_between(number_of_variables, np.array(d_variables_patterns)-d_variables_patterns_std, np.array(d_variables_patterns)+d_variables_patterns_std, color="red", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(s_variables_patterns)-s_variables_patterns_std, np.array(s_variables_patterns)+s_variables_patterns_std, color="green", alpha=0.2)
    plt.fill_between(number_of_variables, np.array(r_variables_patterns)-r_variables_patterns_std, np.array(r_variables_patterns)+r_variables_patterns_std, color="yellow", alpha=0.2)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Number of variables per pattern")
    plt.title("Number of variables per pattern before merge")
    plt.savefig(str(directory)+"\\output\\4\\output\\variables_per_pattern\\" + file_output_names[i] + "_variables.png", bbox_inches='tight')
    plt.show()

    ############### number of pattern ###############

    plt.figure()
    plt.plot(number_of_variables, number_of_patterns, color='blue', linewidth=1.5)
    plt.fill_between(number_of_variables, np.array(number_of_patterns)-number_of_patterns_std, np.array(number_of_patterns)+number_of_patterns_std, color="blue", alpha=0.2)
    print(number_of_patterns)
    print(number_of_patterns_std)
    plt.legend(loc='upper left')

    #print(d_merging_times)
    #print(s_merging_times)
    #print(r_merging_times)

    plt.xlabel("Number of items")
    plt.ylabel("Number of patterns")
    plt.title("Number of patterns after merge")
    plt.savefig(str(directory)+"\\output\\4\\output\\" + file_output_names[i] + "_number_of_patterns.png", bbox_inches='tight')
    plt.show()


