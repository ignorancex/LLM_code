from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import random
import statistics
import sys
from tokenize import String
import xlrd
import os
from xlwt import Workbook


def processArgs(args):
    arg = {}
    nextopt = None
    for a in args:
        if nextopt:
            arg[nextopt] = a
            nextopt = None
        elif a.startswith('--'):
            nextopt = a[2:]
    return arg


def findCorrToIncorr(path, op):
    corrProgs = defaultdict(list)
    numIncorr = defaultdict(int)
    numCorr = defaultdict(int)
    revNumCorr = defaultdict()
    corr_counter = 0
    incorr_counter = 0
    corrToIncorr = defaultdict(list)
    minPercents = []
    minCorrProgs = set()
    for probname in os.listdir(path):
        if '_'+op not in probname:
            continue

        prob_wb = xlrd.open_workbook(path+probname)
        prob_sheet = prob_wb.sheet_by_index(0)
        curr_percent = 150
        curr_prog = ''

        for i in range(1, prob_sheet.nrows):
            name_ic = prob_sheet.cell_value(i, 1)
            name_c = str(prob_sheet.cell_value(i, 0))
            if (prob_sheet.cell_value(i, 4) == 'Yes'):
                val_i = incorr_counter

                exps_corr = int(prob_sheet.cell_value(i, 18))
                exps_incorr = int(prob_sheet.cell_value(i, 19))
                if prob_sheet.cell_value(i, 20):
                    exps_incorr = int(prob_sheet.cell_value(i, 20))
                reps = int(prob_sheet.cell_value(i, 21))
                percent = (reps / (exps_corr + exps_incorr)) * 100
                if percent < curr_percent:
                    curr_percent = percent
                    curr_prog = name_c

                if name_ic in numIncorr:
                    val_i = numIncorr[name_ic]
                else:
                    numIncorr[name_ic] = incorr_counter
                    incorr_counter += 1
                val = corr_counter
                if name_c in numCorr:
                    val = numCorr[name_c]
                else:
                    numCorr[name_c] = corr_counter
                    revNumCorr[corr_counter] = name_c
                    corr_counter += 1

                corrProgs[val_i].append(val)
                corrToIncorr[name_c].append(name_ic)
        if curr_percent != 150:
            minPercents += [curr_percent]
            minCorrProgs.add(curr_prog)
    return [corrProgs, numIncorr, numCorr, revNumCorr, corr_counter, incorr_counter, corrToIncorr, minPercents, minCorrProgs]


def findMinSetNumCorr(corr_counter, revNumCorr, corrToIncorr, numIncorr):
    allCorrProgs = [i for i in range(0, corr_counter)]
    # flag = False
    ans = set()
    usedIncorr = set()
    for i in range(1, len(numIncorr)):
        for comb in itertools.combinations(allCorrProgs, i):
            allIncorr = set()
            for x in comb:
                corrProg = revNumCorr[x]
                allIncorr = allIncorr.union(set(corrToIncorr[corrProg]))
                if len(allIncorr) == len(numIncorr):
                    print("YES!")
                    ans = comb
                    # flag = True
                    usedIncorr = allIncorr
                    return [ans, usedIncorr]


def findCorrProgPercentage(usedIncorr, corrProgs, numIncorr, ans, revNumCorr, op):

    percentage = []
    for probname in os.listdir(path):
        if '_' + op not in probname:
            continue
        n = probname.split('_')[0] + '_solution.txt'

        if n not in usedIncorr:
            continue

        numsCorrs = set(corrProgs[numIncorr[n]]).intersection(ans)
        nameCorrs = [revNumCorr[i] for i in list(numsCorrs)]
        curr_percent = 150

        prob_wb = xlrd.open_workbook(path+probname)
        prob_sheet = prob_wb.sheet_by_index(0)

        for i in range(1, prob_sheet.nrows):
            name_c = str(prob_sheet.cell_value(i, 0))

            if name_c in nameCorrs:
                exps_corr = int(prob_sheet.cell_value(i, 18))
                exps_incorr = int(prob_sheet.cell_value(i, 19))
                if prob_sheet.cell_value(i, 20):
                    exps_incorr = int(prob_sheet.cell_value(i, 20))
                reps = int(prob_sheet.cell_value(i, 21))
                percent = (reps / (exps_corr + exps_incorr)) * 100
                if percent < curr_percent:
                    curr_percent = percent

        percentage += [curr_percent]
    return percentage


def makePlots(minsPer, corrPer, name):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    bp = ax.boxplot([corrPer['0'], corrPer['3'], corrPer['1'], minsPer['0'], minsPer['3'],
                    minsPer['1']], vert=1, showfliers=False, positions=[1, 2, 3, 4, 5, 6])
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5)

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(linewidth=2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color='black',
                   linewidth=2)

    # x-axis labels
    ax.set_xticklabels(['CLARA', 'AGM(L)',
                        'AGM(L+E)', 'CLARA', 'AGM(L)',
                        'AGM(L+E)'])

    s = 20
    m = 'x'
    a = 0.4
    y = [i for i in range(0, 100, 10)]
    plt.plot([3.5 for _ in range(0, 10)], y, linestyle='--', color='black')

    # plt.yticks([i for i in range(10, 70, 10)])
    plt.grid(True, axis='y', linewidth=0.5, linestyle=':')
    e3 = '#e8613c'
    e1 = '#c924a3'
    for val in minsPer['0']:
        plt.scatter(random.uniform(0.8, 1.2) + 3, val,
                    alpha=a, color=e1, s=s, marker=m)
    for val in minsPer['3']:
        plt.scatter(random.uniform(0.8, 1.2) + 4, val,
                    alpha=a, color='#0030D7', s=s, marker=m)
    for val in minsPer['1']:
        plt.scatter(random.uniform(0.8, 1.2) + 5, val,
                    alpha=a, color=e3, s=s, marker=m)
    # ----------------------------------------

    for val in corrPer['0']:
        plt.scatter(random.uniform(0.8, 1.2), val,
                    alpha=a, color=e1, s=s, marker=m)
    for val in corrPer['3']:
        plt.scatter(random.uniform(0.8, 1.2) + 1, val,
                    alpha=a, color='#0030D7', s=s, marker=m)
    for val in corrPer['1']:
        plt.scatter(random.uniform(0.8, 1.2) + 2, val,
                    alpha=a, color=e3, s=s, marker=m)
    # -------------------------------------------

    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.title(
        'Minimizing the number of correct \nprograms for all repairs', loc='left')
    plt.title(
        'Minimizing the overall modification \npercentage for all repairs', loc='right')
    plt.ylabel('Percentage (%)', rotation=90)

    # show plot
    plt.savefig("Summ_" + name + ".pdf")


opts = processArgs(sys.argv[1:])
path = opts.pop('path', '')
plots = int(opts.pop('plots', 0))
probName = str(opts.pop('prob', ''))

# percentages
minsPer = {}
corrPer = {}

# correct programs
progMin = {}
progCorr = {}

for i in ['0', '3', '1']:
    [corrProgs, numIncorr, numCorr, revNumCorr, corr_counter, incorr_counter,
        corrToIncorr, minPercents, minCorrProgs] = findCorrToIncorr(path, i)
    minsPer[i] = minPercents
    progMin[i] = minCorrProgs
    ans, usedIncorr = findMinSetNumCorr(
        corr_counter, revNumCorr, corrToIncorr, numIncorr)
    percentages = findCorrProgPercentage(
        usedIncorr, corrProgs, numIncorr, ans, revNumCorr, i)
    corrPer[i] = percentages
    progCorr[i] = len(ans)

if plots:
    makePlots(minsPer, corrPer, probName)
