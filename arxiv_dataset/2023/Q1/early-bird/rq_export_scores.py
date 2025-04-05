import pandas as pd
from multiprocessing import Process

from release_manager import *
from rq_run import *
# import plotly.graph_objs as go
# import plotly.io as pio
import copy
import shutil

"""
sk
"""

# def getMetric(df, metric):
#
#     # df['train_bugs'] = df['train_changes'] *	df['train_Bug_Per']/100
#     #
#     # df = df.sort_values(by='train_B`ug_Per', ascending=False)
#
#     metricValue = df[metric].values.tolist()
#
#     return metricValue

yearMap = {}

MAX_YEAR = 8

def computeProjectYearMap():

    projectYearMap = {}

    for p in getProjectNames():

        releaselist = release_manager.getProject(p).getReleases()

        for startYear in range(1, MAX_YEAR):

            onlyReleasesToConsider = []
            releaseObjects = getReleasesFromYear(releaselist, startYear)
            onlyReleasesToConsider += [y.getReleaseDate() for y in releaseObjects]

            projectYearMap[p+"_RELEASES_IN_YEAR_"+str(startYear)] = onlyReleasesToConsider


    return projectYearMap

projectYearMap =  None #computeProjectYearMap()

#
# print(" pyear map ",projectYearMap)

# def getSelectionRules():
#
#     # resampling = []
#     #
#     # testChanges = [1,2,3]
#     #
#     # for samples in [100]:
#     #
#     #     for split in [40]:
#     #
#     #         if samples == len(testChanges):
#     #             samplesStr = 'tCg'
#     #         else:
#     #             samplesStr = str(samples)
#     #
#     #         longLabelApprch = '_Smp_' + str(samplesStr) + "Sp" + "_" + str(
#     #             split) + "_Test"
#     #
#     #         resampling.append(longLabelApprch)
#     #
#     # return  resampling + ['Train_ALL_Test_', 'RESAMPLING_SAMPLE_FROM_Recent_Smp_', 'Train_3months', 'Train_6months']
#
#     df = pd.read_csv('./results/project_active_merchant_results.csv')
#     print(df['trainAppraoch'].unique().tolist())
#     return df['trainAppraoch'].unique().tolist()


def readable(selRule):

    if "YEAR:" in selRule:
        return  selRule.replace("YEAR:", "Y")
    elif "3MONTHS" in selRule:
        return selRule.replace('3MONTHS', 'M3')
    elif "6MONTHS" in selRule:
        return selRule.replace('6MONTHS', 'M6')
    elif "RECENT:RELEASE" in selRule:
        return selRule.replace('RECENT:RELEASE', 'RR')
    else:
        return selRule

    #
    #
    # if True:
    #     return selRule #.replace("RESAMPLE_", '')
    # else:
    #     if selRule == 'RESAMPLE_EARLY':
    #         return 'MICRO_Early'
    #     elif selRule == 'RESAMPLE_RANDOM':
    #         return 'MICRO_Random'
    #     elif selRule == 'RESAMPLE_RECENT':
    #         return 'MICRO_Recent'
    #     elif selRule == 'RESAMPLING_SAMPLE_FROM_':
    #         return 'MICRO_Specific'
    #     elif selRule == 'Train_ALL_Test_':
    #         return 'SMOTE_All'
    #     elif selRule == 'Train_3months_Test_':
    #         return 'SMOTE_Recent_3MONTH'
    #     elif selRule == 'Train_6months_Test_':
    #         return 'SMOTE_Recent_6MONTH'
    #     elif selRule == 'Early_60_Per':
    #         return "Early_60%"
    #     elif selRule == 'ALL':
    #         return 'SMOTE_ALL'
    #     else:
    #         return 'error'


def getRQ1a():

    policies = []

    for samples in [12, 25, 50]:#, 100]:

        for buggyPercentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            buggySamples = round(buggyPercentage * samples / 100)
            nonBuggySamples = abs(samples - buggySamples)

            if buggySamples > 5 and nonBuggySamples > 5:
                # print(buggySamples, nonBuggySamples)
                policies.append('RESAMPLE_'+str(buggySamples)+"_"+str(nonBuggySamples))

    policies.append('ALL')

    # winMap = {'RESAMPLE_50_50_LR': 0, 'RESAMPLE_40_60_LR': 0, 'RESAMPLE_25_25_LR': 0, 'RESAMPLE_60_40_LR': 0, 'ALL_SVM': 0,
    #  'RESAMPLE_20_30_LR': -1, 'RESAMPLE_30_70_LR': -1, 'RESAMPLE_50_50_SVM': -2, 'RESAMPLE_12_13_LR': -2, 'ALL_LR': -2,
    #  'RESAMPLE_25_25_SVM': -2, 'RESAMPLE_40_60_SVM': -2, 'RESAMPLE_25_25_KNN': -2, 'RESAMPLE_30_20_LR': -2,
    #  'ALL_NB': -2, 'RESAMPLE_15_10_LR': -2, 'RESAMPLE_30_70_SVM': -2, 'RESAMPLE_15_35_SVM': -2, 'RESAMPLE_20_80_LR': -2,}
    #
    # # winMap = {'RESAMPLE_50_50_LR': 0 }

    return policies, 'rq1a'


def getRq4():
    # Early(cfs), TCA +, Bellwether(cfs) and Early
    # Bellwether(cfs)

    return ['TCAPLUS', 'BELLWETHER', '150_25_25_random_None', '150_25_25_BELLWETHER_random_None'], 'rq4'


def getRq5():
    # Early(cfs), Early(2) and Early Bellwether(2)
    return ['150_25_25_random_None', '150_25_25_random__la_lt_',
     '150_25_25_BELLWETHER_random__la_lt_'], 'rq5'

def getrqX():

    sp = []

    for k,v in CLASSIFIER_SAMPLING_POLICY_MAP.items():
        sp += v

    return list(set(sp)), 'rqx'

def getrqfuture():

    policies = []

    policies.append('RESAMPLE_150_20_30')
    policies.append('RESAMPLE_CPDP_150_20_30')

    return policies, 'rqfuture'

def getTempRQ():
    policies = []
    policies.append('ALL')
    policies.append('150_25_25_random_None')
    policies.append('150_25_25_random__la_lt_')

    return policies, 'rqtemp'
def getRQ4():

    policies = []

    # policies.append('RESAMPLE_YEAR_1_40_60')
    # RESAMPLE_300_20_30

    # policies.append('RESAMPLE_75_20_30')
    policies.append('RESAMPLE_150_20_30')
    policies.append('RESAMPLE_300_20_30')
    policies.append('RESAMPLE_600_20_30')
    policies.append('RESAMPLE_1200_20_30')
    policies.append('ALL')


    # policies.append('RESAMPLE_2400_20_30')
    # policies.append('3MONTHS')
    # policies.append('RECENT_RELEASE')
    # policies.append('6MONTHS')



    return policies, 'rq4'


def getRQ5():

    policies = []

    # policies.append('RESAMPLE_YEAR_1_40_60')
    # RESAMPLE_300_20_30
    # policies.append('RESAMPLE_75_20_30')
    # policies.append('RESAMPLE_300_20_30')
    # policies.append('RESAMPLE_600_20_30')
    # policies.append('RESAMPLE_1200_20_30')
    # policies.append('RESAMPLE_2400_20_30')

    policies.append('RESAMPLE_150_20_30')
    policies.append('3MONTHS')
    policies.append('RECENT_RELEASE')
    policies.append('6MONTHS')
    policies.append('ALL')

    return policies, 'rq5'


def getSelectionRules():
    resamplingPolicies = []

    if True:

        # resamplingPolicies.append('RESAMPLE_EARLY')
        # resamplingPolicies.append('RESAMPLE_RANDOM')
        # resamplingPolicies.append('RESAMPLE_RECENT')

        # resamplingPolicies.append('EARLY')
        # resamplingPolicies.append('RANDOM')
        # resamplingPolicies.append('RECENT')

        # resamplingPolicies.append('RESAMPLE_EARLY_RECENT')
        # resamplingPolicies.append('RESAMPLE_EARLY_RANDOM_RECENT')
        # resamplingPolicies.append('RESAMPLE_EARLY_RANDOM')
        # resamplingPolicies.append('RESAMPLE_RANDOM_RECENT')
        #
        #
        # resamplingPolicies.append('SMOTE_EARLY_RECENT')
        # resamplingPolicies.append('SMOTE_EARLY_RANDOM_RECENT')
        # resamplingPolicies.append('SMOTE_EARLY_RANDOM')
        # resamplingPolicies.append('SMOTE_RANDOM_RECENT')


        resamplingPolicies.append('RESAMPLE_RANDOM12')
        resamplingPolicies.append('RANDOM12')
        resamplingPolicies.append('ALL')
        resamplingPolicies.append('3MONTHS')
        resamplingPolicies.append('6MONTHS')


    elif False:

        resamplingPolicies.append('ALL')
        # resamplingPolicies.append('RESAMPLE_RANDOM_50_200')
        resamplingPolicies.append('RESAMPLE_RANDOM_40_100')
        # resamplingPolicies.append('RESAMPLE_RECENT_40_100')


    # elif proof:
    #
    #     resamplingPolicies.append('ALL')
    #
    #     if year == 1:
    #         resamplingPolicies.append('RESAMPLE_RANDOM_40_200')
    #     elif year == 2:
    #         resamplingPolicies.append('RESAMPLE_RANDOM_40_100')
    #     elif year == 3:
    #         resamplingPolicies.append('RESAMPLE_RECENT_40_100')
    #     elif year == 4:
    #         resamplingPolicies.append('RESAMPLE_RANDOM_40_100')
    #     elif year == 5:
    #         resamplingPolicies.append('RESAMPLE_RANDOM_40_200')
    #     elif year == 6:
    #         resamplingPolicies.append('RESAMPLE_RANDOM_40_50')
    #     elif year == 7:
    #         resamplingPolicies.append('RESAMPLE_RECENT_50_100')
    #     else:
    #         float("opps error!!!")
    #
    # else:
    #     for split in [20, 40, 50, 60, 80]:
    #
    #         for samples in [12, 25, 50, 100, 200]:
    #
    #             info = "_" + str(split) + "_" + str(samples)
    #
    #             resamplingPolicies.append("RESAMPLE_EARLY" + info)
    #             resamplingPolicies.append("RESAMPLE_RANDOM" + info)
    #             resamplingPolicies.append("RESAMPLE_RECENT"+info)
    #
    #     resamplingPolicies.append('ALL')
    #     resamplingPolicies.append('Early60')

    return resamplingPolicies


def getRQ1b():

    # policies = []

    # policies.append('ALL')
    # policies.append('RESAMPLE_YEAR_1_20_30')
    # policies.append('RESAMPLE_YEAR_2_20_30')
    # policies.append('RESAMPLE_YEAR_3_20_30')
    #
    # # policies.append('RESAMPLE_YEAR_1_40_60')
    # # policies.append('RESAMPLE_YEAR_2_40_60')
    # # policies.append('RESAMPLE_YEAR_3_40_60')

    # return policies, 'rq1b'
    #
    # policies.append('RESAMPLE_150_20_30')
    # policies.append('RESAMPLE_300_20_30')
    # policies.append('RESAMPLE_600_20_30')
    # policies.append('RESAMPLE_1200_20_30')
    # policies.append('ALL')

    winMap = {'RESAMPLE_1200_20_30_LR': -1, 'RESAMPLE_600_20_30_LR': -1, 'RESAMPLE_300_20_30_LR': -1, 'ALL_LR': -1,
     'RESAMPLE_150_20_30_LR': -1, 'ALL_SVM': -1}
    # , 'ALL_KNN': -2, 'RESAMPLE_1200_20_30_SVM': -3,
    #  'RESAMPLE_600_20_30_SVM': -3, 'ALL_NB': -4, 'RESAMPLE_1200_20_30_KNN': -4, 'RESAMPLE_300_20_30_KNN': -4,
    #  'RESAMPLE_600_20_30_KNN': -4, 'RESAMPLE_150_20_30_KNN': -4, 'RESAMPLE_300_20_30_SVM': -4, 'ALL_RF': -4,
    #  'ALL_DT': -5, 'RESAMPLE_150_20_30_SVM': -5, 'RESAMPLE_600_20_30_RF': -5, 'RESAMPLE_300_20_30_RF': -5,
    #  'RESAMPLE_1200_20_30_NB': -6, 'RESAMPLE_1200_20_30_RF': -6, 'RESAMPLE_150_20_30_RF': -6,
    #  'RESAMPLE_1200_20_30_DT': -7, 'RESAMPLE_600_20_30_DT': -7, 'RESAMPLE_600_20_30_NB': -7,
    #  'RESAMPLE_300_20_30_DT': -7, 'RESAMPLE_150_20_30_DT': -7, 'RESAMPLE_300_20_30_NB': -7, 'RESAMPLE_150_20_30_NB': -9}
    #
    return  list(winMap.keys()), 'rq1b'

def convert(v, metric):

    return [float(vv) for vv in v]

    # vv= []
    #
    # if v is not None:
    #
    #     for vvv in v:
    #
    #         try:
    #             converted = float(vvv)
    #             if str(converted).lower() != 'nan':
    #                 vv.append(converted)
    #             elif metric in ['g-score']:
    #                 vv.append(0)
    #         except:
    #             continue
    #
    # return vv


def splitFullPolicy(samplingPolicy):

   spArr = samplingPolicy.split('_')

   classifier = spArr[len(spArr) - 1]

   rawPolicy = samplingPolicy.replace('_'+classifier, '')

   rr = rawPolicy
   if rawPolicy == 'M6':
       rr = '6MONTHS'
   elif rawPolicy == 'M3':
       rr = '3MONTHS'

   return rr, classifier


def filterDFWins(df, samplingPolicies):
    releaseList = []

    for samplingPolicy in samplingPolicies:

        rawPolicy, classifier = splitFullPolicy(samplingPolicy)


        samplingReleaseList = df[ (df['trainApproach'].str.strip() == rawPolicy) & (df['classifier'].str.strip() == classifier)]['testReleaseDate'].values.tolist()

        # print('trying ', "["+rawPolicy+"]", "["+classifier+"]" , ' found ', len(samplingReleaseList), ' releases common ', samplingReleaseList )

        if samplingReleaseList is None or len(samplingReleaseList) == 0:
            return []
        else:
            releaseList.append(samplingReleaseList)

    testReleaseSet = None

    for releases in releaseList:

        if testReleaseSet is None:

            testReleaseSet = list(set(releases))
            continue
        else:

            testReleaseSet = list(set(testReleaseSet) & set(releases))

    return testReleaseSet


def filterDF(df, samplingPolicies):

    releaseList = []


    for samplingPolicy in samplingPolicies:

        samplingReleaseList  = df[ df['trainApproach'] == samplingPolicy ]['testReleaseDate'].values.tolist()
        # print(samplingPolicy, len(samplingReleaseList))
        if samplingReleaseList is None or len(samplingReleaseList) == 0:
            return []
        else:
            releaseList.append(samplingReleaseList)

    testReleaseSet = None

    for releases in releaseList:

        if testReleaseSet is None:

            testReleaseSet = list(set(releases))
            continue
        else:

           testReleaseSet =  list( set(testReleaseSet) & set(releases) )



    return testReleaseSet


def toBoxLabel(selRule, rq):



    boxLabel = selRule.replace('RESAMPLE_', '')
    # boxLabel = boxLabel.replace('_LR', '')
    boxLabel = boxLabel.replace('_', ":")
    boxLabel = boxLabel.replace('CPDP', "C")
    boxLabel = boxLabel.replace('150:25:25', 'E[0,150]')
    boxLabel = boxLabel.replace('300:20:30', 'E[0,300]')
    boxLabel = boxLabel.replace('600:20:30', 'E[0,600]')
    boxLabel = boxLabel.replace('1200:20:30', 'E[0,1200]')

    if '25:25' == boxLabel:
        boxLabel = 'E'

    if (rq == 'rq1a' and ( 'ALL' in boxLabel or '20:30' in boxLabel)) or (rq == 'rq1b' and ('ALL' in boxLabel or  'E[0,150]' in boxLabel)) or \
            (rq == 'rq2' and  'E[0,150]' in boxLabel)   or (rq == 'rqfuture' and 'E[0,150]' in boxLabel):

        if rq == 'rq1a':
            boxLabel = '<b>'+boxLabel+' &#8592;</b>'
        else:
            boxLabel = '<b>' + boxLabel + '</b>'


    return readable(boxLabel)


# def getColor(rqText):
#
#     if rqText == 'rq1a':
#         color = green_light
#     elif rqText == 'rq1b' or rqText == 'rq4':
#         color = green_light
#     elif rqText == 'rq2':
#         color = green_light
#     else:
#         float("error")
#
#     return color


def getRank(rq, metric, selRule):


    df = pd.read_csv('./results/a_' + rq + '/sk/z' + metric + ".csv")
    # minRank = df['rank'].min()
    # maxRank = df['rank'].max()
    df = df[df['policy'].str.strip() == selRule]
    rank = int(df['rank'].values.tolist()[0])

    return rank

def getWinRank(rq, metric, fullSelectionRule):
    df = pd.read_csv('./results/a_' + rq + '/sk/z' + metric + ".csv")
    # minRank = df['rank'].min()
    # maxRank = df['rank'].max()
    df = df[df['policy'].str.strip() == fullSelectionRule]
    rank = int(df['rank'].values.tolist()[0])

    return rank

    # if maxRank == minRank:
    #     return 0
    #
    # return (rank - minRank) / (maxRank - minRank)


def getDefaultColor(metric):

    if metric.lower() in ['brier', 'ifa', 'd2h', 'pf']:
        return  dark_orange
    else:
        return dark_green

def getMediumColor(metric):

    if metric.lower() in ['brier', 'ifa', 'd2h', 'pf']:
        return  medium_orange
    else:
        return medium_green

def getLightColor(metric):

    if metric.lower() in ['brier', 'ifa', 'd2h', 'pf']:
        return  light_orange
    else:
        return light_green



# def plotWins(metric):
#
#     write = False
#
#     for rqm in [ getrq2 ]:
#
#         samplingPolicies , rq = rqm()
#
#         print(samplingPolicies)
#
#         boxplot_data = []
#
#         labelRankMap = {}
#
#         for fullSelectionRule in  samplingPolicies: #[ 'RESAMPLE_EARLY', 'ALL' , 'RESAMPLE_RANDOM','RESAMPLE_RECENT']:
#
#             rawPolicy, classifier = splitFullPolicy(fullSelectionRule)
#
#             print( ' rawPolicy, classifier ', rawPolicy, classifier)
#
#             metricValues = []
#
#             count = 0
#
#             for pType in [ 'All_projects' ]:
#
#                 projectsSkipped = 0
#
#                 for p in getProjectNames(pType):
#
#                     df = pd.read_csv('./results/a_'+rq+'/project_' + p + '_results.csv')
#                     # df = df[df['classifier'] == classifier]
#
#                     # print("sending ",p)
#                     commonReleases = filterDFWins(df, samplingPolicies)
#
#                     count += len(commonReleases)
#
#                     # print("commonReleases ",len(commonReleases))
#
#                     if len(df) > 0:
#                         sDF = df[ (df['testReleaseDate'].isin(commonReleases) ) & ( df['trainApproach'] == rawPolicy ) & (df['classifier'] == classifier)]
#                     else:
#                         projectsSkipped += 1
#                         continue
#
#                     v = sDF[metric].values.tolist()
#
#                     if len(commonReleases) != len(v):
#                         print('**** not equal \t\t', p, rawPolicy, metric, commonReleases, v,  sDF['testReleaseDate'].values.tolist())
#
#                     before = len(v)
#                     v = convert(v)
#                     if before - len(v) > 0:
#                         print("Loss = ", before - len(v))
#
#                     metricValues += v
#
#                 print(projectsSkipped, ' for ' , rawPolicy)
#
#                 boxLabel = toBoxLabel(fullSelectionRule, rq)
#
#                 print(fullSelectionRule, ' population size = ', len(metricValues))
#                 boxplot_data.append(go.Box(fillcolor=white, marker=dict(color=black),
#                                            y=metricValues, name=boxLabel, showlegend=False,
#                                            orientation="v",
#                                            line=dict(width=1.5)))
#
#                 labelRankMap[boxLabel] = getWinRank(rq, metric, fullSelectionRule)
#
#         sortByMedian(boxplot_data, metric.lower() in ['brier', 'ifa', 'd2h', 'pf'])
#
#         if metric == 'roc_auc':
#             axis = 'AUC'
#         else:
#             axis = metric.upper()
#
#         previousRank = None
#         previousColor = None
#
#         mediumUsed = False
#         for b in range(0, len(boxplot_data)):
#
#             currentRank = labelRankMap[boxplot_data[b].name]
#
#             if previousRank is None:
#                 boxplot_data[b].fillcolor = getDefaultColor(metric)
#                 previousColor = getDefaultColor(metric)
#             elif abs(previousRank - currentRank) == 0:
#                 boxplot_data[b].fillcolor = previousColor
#             elif abs(previousRank - currentRank) >= 1:
#                 if not mediumUsed:
#                     boxplot_data[b].fillcolor = getMediumColor(metric)
#                     previousColor = getMediumColor(metric)
#                     mediumUsed = True
#                 else:
#                     boxplot_data[b].fillcolor = getLightColor(metric)
#                     previousColor = getLightColor(metric)
#             else:
#                 float('error')
#
#             previousRank = currentRank
#
#             # elif currentRank != previousRank:
#             #     previousRank = currentRank
#             #     if previousColor == white:
#             #         boxplot_data[b].fillcolor = getColor(rq)
#             #         previousColor = getColor(rq)
#             #     else:
#             #         boxplot_data[b].fillcolor = white
#             #         previousColor = white
#             # elif currentRank == previousRank:
#             #     boxplot_data[b].fillcolor = previousColor
#             # else:
#             #     float('error')
#
#         if not write:
#             plot_boxes(boxplot_data, rq+"_"+metric, '', axis, rq)


def get_common_test_releases(projectDF, sampling_policies, classifiers):

    common_releases = None

    for sp in sampling_policies:

        for c in classifiers:

            if sp not in CLASSIFIER_SAMPLING_POLICY_MAP[c]:
                continue

            # if c in ['TUNED_DT', 'TUNED_LR']  and sp != 'FIRST_150':
            #     continue

            # print('Project lines # : ',len(projectDF))

            curr_common_releases = projectDF[(projectDF['trainApproach'] == sp) & (projectDF['classifier'] == c)][
                    'testReleaseDate'].values.tolist()

            # print(len(curr_common_releases), sp, c)

            if common_releases is None:
                common_releases = curr_common_releases
                # print('\t once = ', common_releases)
                continue
            else:
                common_releases = list(set(common_releases) & set(curr_common_releases))
                # print('\t reset = ', common_releases)
                # print('\t ', sp, c, len(common_releases))

    if common_releases is None:
        return []

    return common_releases


def getExpClassifiers():

    c = []
    for k, v in  CLASSIFIER_SAMPLING_POLICY_MAP.items():

        c.append(k)


    return list(set(c))


metric_srule_size = {}
def aggregate_eval_measures(metric, projectStartDateMap):

    commonReleasesMap = {}

    f = open('./results/sk/z' + metric  + '.txt', "a+")

    classifiers = getExpClassifiers()

    # classifiers = ['LogisticRegression']

    for rqm in [ getrqX ]:

        samplingPolicies , rq = rqm()
        print(' >> ', samplingPolicies,rq)

        for classifier in classifiers:

            for selRule in  samplingPolicies: #[ 'RESAMPLE_EARLY', 'ALL' , 'RESAMPLE_RANDOM','RESAMPLE_RECENT']:

                # if classifier not in ['LR_None'] and selRule == '150_25_25_random_None':
                #     continue

                metricValues = []

                count = 0

                for pType in [ 'All_projects' ]:

                    projectsSkipped = 0



                    for p in getProjectNames():

                        if p == BELLWETHER_PROJECT:
                            continue


                        try:
                            projectdf = pd.read_csv('./results/project_' + p + '_results.csv')
                            projectdf = projectdf[projectdf['d2h'] != 'CLF_ERROR']
                            # projectdf = projectdf[ (projectdf['test_changes'] > 5) ]

                            projectdf = projectdf[ (projectdf['g-score'].isnull() == False) &
                                                   ( projectdf['mcc'].isnull() == False)]


                            # if b_x != len(projectdf):
                            #     print("\t ** after ", len(projectdf))
                            #     float('error 3')



                            # if selRule.startswith('E'):
                            #     print('warn:: restricting to raw E rule ', selRule)
                            #     projectdf = projectdf[(projectdf['train_changes'] >= 50)]




                        except Exception as e:
                            print(e, p, ' does not exist')
                            continue

                        projectdf["testReleaseDate"] = pd.to_numeric(projectdf["testReleaseDate"])

                        # YEAR = 2
                        #
                        # startDate = projectdf['testReleaseDate'].min()
                        # endDate = startDate + (one_year * YEAR)
                        #
                        # projectdf = projectdf[ (projectdf['testReleaseDate'] < endDate)  ]
                        #
                        # print("warn::: only considering first YEAR years!! \n\n ")

                        # year 1 2 3 4 5...

                        if p not in commonReleasesMap:

                            # spcopy = copy.deepcopy(samplingPolicies)
                            # if p+'_ALL_CHANGES' in spcopy:
                            #     spcopy.remove(p+'_ALL_CHANGES')
                            # else:
                            #     print(p+'_ALL_CHANGES', ' not in spcopy ?>>> ', len(spcopy))

                            commonReleasesMap[p] = get_common_test_releases(projectdf, samplingPolicies, classifiers )

                        common_test_releases = commonReleasesMap[p]

                        #
                        # if rq == 'rq2':
                        #     sixmonths = projectStartDateMap[p] + (6 * one_month )
                        #     df = df[ df['testReleaseDate'] > sixmonths ]

                        classifier_df = projectdf[ projectdf['classifier'] == classifier ]

                        # print(p, classifier, selRule)
                        count += len(common_test_releases)

                        if len(classifier_df) > 0:
                            final_df = classifier_df[ (classifier_df['testReleaseDate'].isin(common_test_releases) ) & ( classifier_df['trainApproach'] == selRule ) ]
                        else:
                            projectsSkipped += 1
                            continue

                        v = final_df[metric].values.tolist()

                        # before = len(v)
                        v = convert(v, metric)

                        # if before - len(v) > 0:
                        #     print("Loss = ", before - len(v), p, metric, selRule)
                        #     float('loss error')

                        metricValues += v


                    print('Projects that did not make it ', projectsSkipped, ' for ' , selRule)

                if selRule not in CLASSIFIER_SAMPLING_POLICY_MAP[classifier]:
                    continue

                print("Releases = ", len(metricValues), selRule)

                f.write(readable(selRule) + "_" + classifier   + "\n")
                line = ''


                print("\t** Length of the values = ", metric, selRule, len(metricValues))
                metric_srule_size[metric+'_'+selRule] = len(metricValues)
                for c in metricValues:
                    line += str(c) + " "

                f.write(line.strip() + "\n\n")

                for k, v in metric_srule_size.items():

                    print( metric_srule_size[k] , '\t\t\t', k)



def sortByMedian(plotlyData, reverse):

    dataList = plotlyData

    # print(len(plotlyData), ' is length')

    i = 0
    j = 1

    while i < len(plotlyData):

        # print(dataList[i])

        j = 0
        while j < len(plotlyData):
            # print(i,j)
            # print(dataList[i])

            if reverse:
                if np.median(dataList[i].y) < np.median(dataList[j].y):
                    temp = dataList[i]
                    dataList[i] = dataList[j]
                    dataList[j] = temp

                j += 1
            else:

                if np.median(dataList[i].y) > np.median(dataList[j].y):

                    temp = dataList[i]
                    dataList[i] = dataList[j]
                    dataList[j] = temp

                j += 1

        i += 1

    return dataList


def getRange(yaxisLabel):

    if 'IFA' not in yaxisLabel:
        return [0,1]
    else:
        return None


def getTickFontSize(rq):

    if rq == 'rq1b':
        return 25
    elif rq == 'rqfuture':
        return 35
    elif rq == 'rq2':
        return 25
    else:
        return FONT_SIZE_SMALL


def getBottom(rq):
    if rq == 'rq1b':
        return 5
    elif rq == 'rq1a':
        return 150
    elif rq == 'rq2':
        return 155
    elif rq == 'rqfuture':
        return 5


def getFigureTitle(filename):

    for m in [  'roc_auc', 'recall', 'popt20']:

        if m in filename.lower():
            return '<i>larger values preferred</i>'

    for m in [ 'd2h', 'brier',  'pf', 'ifa']:

        if m in filename.lower():
            return '<i>lower values preferred</i>'

    return 'error'



# def buggySelectionRule(metric):
#
#         f = open('./results/sk/z'+metric+'.txt', "a+")
#
#         for classifier in getSimpleNames():
#
#             if classifier == 'NB':
#                 continue
#
#             buggy = []
#             all = []
#
#             for p in getProjectNames():
#
#                 # if p not in yearMap:
#                 #     releaseList = release_manager.getProject(p).getReleases()
#                 #     (releaseList[len(releaseList) - 1].getReleaseDate() - releaseList[
#                 #         0].getReleaseDate()) / one_year
#                 #     yearMap[p] = year
#                 #
#                 # if yearMap[p] < 0:
#                 #     print("skipping ",p)
#                 #     continue
#
#                 df = pd.read_csv('./results/project_'+p+'_results.csv')
#                 df = df [df['classifier'] == classifier]
#
#                 yearDF = df[df['trainReleaseDate'] == 'PAST_YEAR']
#                 v = getMetric(yearDF, metric)
#                 if v is not None and v > 0:
#                     buggy.append(v)
#
#                 allDF = df[df['trainReleaseDate'] == 'inf']
#                 v = getMetric(allDF, metric)
#                 if v is not None and v > 0:
#                     all.append(v)
#
#             f.write("Buggy_"+classifier+"\n")
#             line = ''
#             for c in buggy:
#                 line += str(c) + " "
#
#             f.write(line.strip() + "\n\n")
#
#             f.write("All_" + classifier + "\n")
#             line = ''
#             for c in all:
#                 line += str(c) + " "
#
#             f.write(line.strip() + "\n\n")


def shorten(approach):

    if len(approach.values.tolist()) == 1:
        approach = approach.values.tolist()[0]
    elif len(approach.values.tolist()) > 1:
        return 'error 1'
    else:
        return 'NA'

    if approach.startswith('Train_ALL_Test'):
        return 'ALL'
    else:
        for x in range(1, 14):

            if approach.startswith('Train_'+str(x)+'_Test'):
                return 'Train_'+str(x)

    return 'error 2'


def writePlaces():

    yearMap = {}

    for classifier in getSimpleNames():

        repoName = []
        bestApproachfirst6 = []
        worstApproachfirst6 = []
        bestApproachnext6 = []
        worstApproachnext6 = []


        for p in getProjectNames():

            if p not in yearMap:
                releaseList = release_manager.getProject(p).getReleases()
                year = (releaseList[len(releaseList)-1].getReleaseDate() - releaseList[0].getReleaseDate())/one_year
                yearMap[p] = year


            if  yearMap[p] < 3:
                continue

            df = pd.read_csv('./results/project_'+p+"_results.csv")
            df = df[ df['classifier'] == classifier]

            if len(df) <= 2:
                continue

            df = df.sort_values(by='f1', ascending=False)

            firstDf = df[ df['trainApproach'].str.endswith('first6')]
            nextDF = df[df['trainApproach'].str.endswith('next6')]

            first6BestRow = firstDf.head(1)
            first6WorstRow = firstDf.tail(1)

            next6BestRow = nextDF.head(1)
            next6WorstRow = nextDF.tail(1)

            repoName.append(p)
            bestApproachfirst6.append( shorten(first6BestRow['trainApproach']))
            bestApproachnext6.append(shorten(next6BestRow['trainApproach']))

            worstApproachfirst6.append(shorten(first6WorstRow['trainApproach']))
            worstApproachnext6.append(shorten(next6WorstRow['trainApproach']))



        newDF = pd.DataFrame()

        newDF['project'] = repoName
        newDF['best_f6'] = bestApproachfirst6
        newDF['best_n6'] = bestApproachnext6

        newDF['worst_f6'] = worstApproachfirst6
        newDF['worst_n6'] = worstApproachnext6

        newDF.to_csv('./results/infer/consolidated'+classifier+'.csv', index=False)


def getMap(df, columns):
    bestMap = {}
    for best in columns:
        values = df[best].values.tolist()
        for v in values:
            # if v == 'ALL':
            #     continue

            if v not in bestMap:
                bestMap[v] = 0

            bestMap[v] = bestMap[v] + 1

    return bestMap

def whoWon():

    for classifier in getSimpleNames():

        df = pd.read_csv('./results/infer/consolidated'+classifier+'.csv')
        bestMap = getMap(df, ['best_f6', 'best_n6'])
        worstMap = getMap(df, ['worst_f6', 'worst_n6'])

        print('\t',bestMap)
        print('\t',worstMap)

        bestTotal = 0
        for key, value in bestMap.items():

            if str(key).startswith('Train_'):
                bestTotal += value

        worstTotal = 0
        for key, value in bestMap.items():

            if str(key).startswith('Train_'):
                worstTotal += value


        print(classifier, max(bestMap.items(), key=operator.itemgetter(1))[0], round(percentage(max(bestMap.items(), key=operator.itemgetter(1))[1], bestTotal)),
              max(worstMap.items(), key=operator.itemgetter(1))[0], round(percentage( max(worstMap.items(), key=operator.itemgetter(1))[1], worstTotal)))

def unitTestHere():
    p = 'android'
    df = pd.read_csv('./results/project_' + p + '_results.csv')
    df = df[df['classifier'] == 'LR']

    selRule = 'ALL'
    sDF = df[(df['trainApproach'] == selRule)]
    sDF = sDF.groupby(['testReleaseDate']).median()
    print(sDF['precision'])


def getStr(proof=False):
    for metric in ['d2h', 'roc_auc', 'ifa']:
        for year in range(1, MAX_YEAR):
            if proof:
                print('cat z' + metric + '_proof_year_' + str(
                    year) + '.txt | python2 suv_sk.py --text 30 --latex False > z' + metric + '_proof_year_' + str(
                    year) + '.csv & ')
            else:
                print('cat z' + metric + '_year_' + str(
                    year) + '.txt | python2 suv_sk.py --text 30 --latex False > z' + metric + '_year_' + str(
                    year) + '.csv & ')




def clean():

    for f in os.listdir('./results/'):
        try:
            if f.endswith('.csv'):
                os.remove('./results/'+f)
                print('removed ',f)
        except Exception as e:
            print(e)

    for f in os.listdir('./results/sk/'):
        try:
            if f.startswith('z'):
                os.remove('./results/sk/'+f)
                print('removed ',f)
        except Exception as e:
            print(e)

def learnerSamplingMap():

    if RQ == 5:
        df = pd.read_csv('./results/project_'+TSE_SZZ_PROJECTS[0]+'_results.csv')
    else:
        try:
            df = pd.read_csv('./'+RESULTS_FOLDER+'/project_ActionBarSherlock_results.csv')
        except Exception as e:
            df = pd.read_csv('./'+ RESULTS_FOLDER+'/project_server_results.csv')


    classifiers = df['classifier'].unique()
    # classifiers = ['LogisticRegression', 'SVC']

    classifierPolicyMap = {}

    # print('\n warn : TCA plus removed!!!')

    for clf in classifiers:

        # if clf != 'LogisticRegression':
        #     continue

        ran_policies = df[df['classifier'] == clf]['trainApproach'].unique().tolist()

        # to_remove = []
        #
        # for r in ran_policies:
        #     if not r.startswith('B_'):
        #         to_remove.append(r)
        #
        # for rem in to_remove:
        #     ran_policies.remove(rem)



        # # for r in ['T_B_'+str(x) for x in range(1, len(BELLWETHER_PROJECTS) + 1) ]:
        # #     if r in ran_policies:
        # #         ran_policies.remove(r)
        #
        # for r in ['B_'+str(x) for x in range(1, len(BELLWETHER_PROJECTS) + 1) ]:
        #     if r in ran_policies:
        #         ran_policies.remove(r)



        # ran_policies.remove('E_T2_scikit-learn')

        # selected_projects = ['django-payments', 'restheart', 'apollo', 'sauce-java', 'portia', 'opendht', 'dogapi', 'midpoint',
        #  'active_merchant', 'zulip', 'woboq_codebrowser', 'pry']
        #
        # ran_policies = []
        # for s in selected_projects:
        #     ran_policies.append(s+'_E_CHANGES')
        #     ran_policies.append(s+'_ALL_CHANGES')





        classifierPolicyMap[clf] = ran_policies


        #
        # print("warning override : ", classifierPolicyMap)
        # classifierPolicyMap[clf] = ['TCAPLUS', 'BELLWETHER', 'ALL', '150_25_25_random_None',
        #  '150_25_25_head_size', '150_25_25_BELLWETHER_head_size']

    print('classifierPolicyMap = ', classifierPolicyMap)




    return classifierPolicyMap

CLASSIFIER_SAMPLING_POLICY_MAP = learnerSamplingMap()

print(CLASSIFIER_SAMPLING_POLICY_MAP)


METRICS_TO_PLOT = [ 'g-score',  'd2h', 'mcc', 'roc_auc', 'brier', 'pf', 'ifa',  'recall' ]

def cross_check():

    r = []
    release_loss = 0
    for p in ['server']:
        try:
            projectdf = pd.read_csv('./results/project_' + p + '_results.csv')
            before = len(projectdf)
            projectdf = projectdf[  projectdf['d2h'] != 'CLF_ERROR' ]

            # projectdf = projectdf[(projectdf['test_changes'] > 5)]
            # projectdf = projectdf.dropna(subset=METRICS_TO_PLOT)

            print(p, before, len(projectdf), int( percentage(len(projectdf), before)))
            r.append(int( percentage(len(projectdf), before)))

        except Exception as e:
            print(e)
            continue

    r.sort()
    print(len(r), r)


def move_results():

    try:
        sk_results_folder = 'results'
        for root, all_directories, files in os.walk(RESULTS_FOLDER):
            for csv_file in files:
                if csv_file.endswith('.csv'):
                    shutil.copy(os.path.join(root, csv_file), sk_results_folder)
                    print("Moved ",csv_file, " to ", sk_results_folder)

    except Exception as e:
        print("Error: ",e)




if __name__ == '__main__':

    print(CLASSIFIER_SAMPLING_POLICY_MAP)

    clean()

    move_results()

    projectStartDateMap = {}


    procs = []

    for metric in   METRICS_TO_PLOT:

        proc = Process(target=aggregate_eval_measures(metric, projectStartDateMap), args=(metric,))
        procs.append(proc)
        proc.start()

    # ************************************ Not validating remember ********************************************************

    # Complete the processes
    for proc in procs:
        proc.join()
#
