
import pandas as pd
import os

"""
@author : Shrikanth N C
Treatment that wins on most measures.
"""


def getRank(df, r):
    ranks = df[df['policy'] == r]['rank'].values.tolist()


    if len(ranks) != 1:
        float('error!')

    return ranks[0]


def updateMap(ruleScoreMap, df, add=True):
    selectionRule = df['policy'].values.tolist()

    for r in selectionRule:

        if r not in ruleScoreMap:
            ruleScoreMap[r] = 0

        rank = getRank(df, r)
        if add:
            ruleScoreMap[r] +=  rank
        else:
            ruleScoreMap[r] -=  rank

prefix = './results/sk/'


def getMetricMedian(metric, policy):

    print(metric, policy)
    df = pd.read_csv(prefix + 'z'+metric+'.csv')
    medianValues = df[ df['policy'].str.strip() == policy]['median'].values.tolist()
    iqrValues = df[df['policy'].str.strip() == policy]['iqr'].values.tolist()
    print(medianValues, iqrValues)


    if len(medianValues) == 1 and len(iqrValues) == 1:
        mv = medianValues[0]
        iqr = iqrValues[0].split(' (')[0].strip()

        if metric == 'ifa':
            iqr_str = ' (' + str(int(float(iqr) * 1)) + ')'
            mv = str(int(float(mv) * 1))
        else:
            iqr_str = ' (' + str(int(float(iqr) * 100)) + ')'
            mv = str(int(float(mv) * 100))



        rankValues = df[df['policy'].str.strip() == policy]['rank'].values.tolist()
        print(policy, ' policy rank ', rankValues, ' MAX RANK = ',df['rank'].max())

        if  rankValues[0] >=  df['rank'].max() - EXTRA    and metric in ['roc_auc', 'recall' , 'g-score', 'mcc', 'precision']:
            print('\t returning 1' )
            return ' '+str(mv) + iqr_str, 1
        elif rankValues[0] <=  df['rank'].min() + EXTRA     and metric in ['pf', 'd2h' , 'brier', 'ifa']:
            print('\t returning 2')
            return ' ' + str(mv) + iqr_str, 1
        else:
            print('\t returning 3')
            return  str(mv) + iqr_str , 0

    return 'error', None

def getMetricRank(metric, policy):

    print(metric, policy)
    df = pd.read_csv(prefix + 'z'+metric+'.csv')
    medianValues = df[ df['policy'].str.strip() == policy]['rank'].values.tolist()

    print(medianValues)
    if len(medianValues) == 1:
        return medianValues[0]

    return 'error'



def readable(selRule):

    if "YEAR:" in selRule:
        return  selRule.replace("YEAR:", "Y")
    elif "3MONTHS" in selRule:
        return "\\rr "+ selRule.replace('3MONTHS', 'M3')
    elif selRule in ['M3', 'M6']:
        return "\\rr " + selRule
    elif "6MONTHS" in selRule:
        return "\\rr "+selRule.replace('6MONTHS', 'M6')
    elif "RECENT:RELEASE" in selRule:
        return "\\rr "+selRule.replace('RECENT:RELEASE', 'RR')
    elif selRule in ['ALL', '20:30', '25:25']:
        # return "\\all "+selRule
        return selRule
    elif '150:25:25' in selRule:
        # return "\ "+selRule
        return  selRule.replace('150:25:25', 'EARLY')
    else:
        return selRule

def toBoxLabel(samplingPolicy):

    print("received ",samplingPolicy)
    spArr = samplingPolicy.split('_')

    classifier = spArr[len(spArr) - 1]

    selRule = samplingPolicy.replace('_' + classifier, '')


    boxLabel = selRule.replace('RESAMPLE_', '')
    # boxLabel = boxLabel.replace('_LR', '')
    boxLabel = boxLabel.replace('_', ":")
    boxLabel = boxLabel.replace('CPDP', "Cross")
    # boxLabel = boxLabel.replace('150:20:30', 'E[0,150]')
    # boxLabel = boxLabel.replace('300:20:30', 'E[0,300]')
    # boxLabel = boxLabel.replace('600:20:30', 'E[0,600]')
    # boxLabel = boxLabel.replace('1200:20:30', 'E[0,1200]')
    # boxLabel = boxLabel.replace('20:30', 'E')

    return readable(boxLabel), classifier


def shorten(classifiers):

    shortened_classifier_names = []
    for c in classifiers:
        if c  in 'LogisticRegression':
            shortened_classifier_names.append('LR')
        elif c in 'KNeighborsClassifier':
            shortened_classifier_names.append('KNN')
        elif c in 'SVC':
            shortened_classifier_names.append('SVM')
        elif c in 'RandomForestClassifier':
            shortened_classifier_names.append('RF')
        elif c in 'GaussianNB':
            shortened_classifier_names.append('NB')
        elif c in 'DecisionTreeClassifier':
            shortened_classifier_names.append('DT')
        else:
            shortened_classifier_names.append(c)

    return shortened_classifier_names


bellprojects = ['django-payments', 'restheart', 'apollo', 'sauce-java', 'portia', 'opendht', 'dogapi', 'midpoint',
         'active_merchant', 'zulip', 'woboq_codebrowser', 'pry']

def decorate(policy):

    decorated_policies = []

    for p in policy:

        added = False
        for b in bellprojects:

            if b.replace('_',':') in p:
                bellindex = bellprojects.index(b) + 1

                if 'ALL' in p:
                    decorated_policies.append('\\bellwether ' + '$Bellwether_{'+str(bellindex)+'}$')
                    added = True

                elif 'E:' in p:
                    decorated_policies.append('\\bellwether ' + '$E^{+}_{'+str(bellindex)+'}$')
                    added = True

        if not added:

            if p.startswith('E'):
                decorated_policies.append(' ' + p)
            elif p.startswith('T'):
                decorated_policies.append('\\tca ' + p)
            elif p.startswith('B'):
                decorated_policies.append('\\bellwether ' + p)
            else:
                decorated_policies.append(p)




        # if p == 'E':
        #     decorated_policies.append(' E')
        # elif p.startswith('E:B:'):
        #     decorated_policies.append('\ebell '+p)
        # elif p.startswith('B'):
        #     decorated_policies.append('\\bellwether ' + p)
        # elif p.startswith('T:'):
        #     decorated_policies.append('\\tca ' + p)



    return decorated_policies


def writeLatexCSVTable(sortedMap):

    aggRank = 1

    policy = []
    rank = []

    d2h = []
    # d2h_rank = []

    auc = []
    # auc_rank = []

    ifa = []
    # ifa_rank = []

    brier = []
    # brier_rank = []

    recall = []
    # recall_rank = []

    gscore = []
    # gscore_rank = []

    pf = []
    # pf_rank = []

    # precision = []
    mcc = []

    classifiers = []

    frequency = []

    pastV = None

    for key, v in sortedMap.items():
        f = 0

        if pastV is None:
            pastV = v
        elif pastV != v:
            pastV = v
            aggRank += 1

        rawPolicy, classifier = toBoxLabel(key)
        policy.append(rawPolicy)
        classifiers.append(classifier)
        rank.append(aggRank)

        mv, win = getMetricMedian('d2h', key)
        f += win
        d2h.append(mv)

        mv, win = getMetricMedian('roc_auc', key)
        f += win
        auc.append(mv)

        mv, win = getMetricMedian('ifa', key)
        f += win
        ifa.append(mv)

        mv, win = getMetricMedian('brier', key)
        f += win
        brier.append(mv)

        mv, win = getMetricMedian('recall', key)
        f += win
        recall.append(mv)

        mv, win = getMetricMedian('pf', key)
        f += win
        pf.append(mv)

        mv, win = getMetricMedian('g-score', key)
        f += win
        gscore.append(mv)

        # mv, win = getMetricMedian('precision', key)
        # f += win
        # precision.append(mv)

        mv, win = getMetricMedian('mcc', key)
        f += win
        mcc.append(mv)

        frequency.append(f)

    df = pd.DataFrame()

    df['Policy'] = decorate(policy)
    df['Classifier'] = shorten(classifiers)
    df['Wins'] = frequency

    df['Recall+'] = recall
    df['PF-'] = pf
    df['AUC+'] = auc
    df['D2H-'] = d2h
    df['Brier-'] = brier
    df['G-Score+'] = gscore
    df['IFA-'] = ifa

    # df['PRECISION'] = precision
    df['MCC+'] = mcc

    df = df.sort_values(['PF-'], ascending=True)
    df = df.sort_values(['Wins', 'Recall+'], ascending=False)
    df.to_csv(prefix+'zlatex_table1.csv', index=False)

    print(df)

    print("Please view results csv at :", prefix + 'zlatex_table1.csv' )




def masterOfALLMeasures():

        ruleScoreMap = {}

        d2hDF = pd.read_csv(prefix+ 'zd2h.csv')
        rocDF = pd.read_csv(prefix + 'zroc_auc.csv')
        ifaDF = pd.read_csv(prefix+'zifa.csv')
        brierDF = pd.read_csv(prefix+'zbrier.csv')
        recallDF = pd.read_csv(prefix+ 'zrecall.csv')
        pfDF = pd.read_csv(prefix+ 'zpf.csv')
        gscoreDF = pd.read_csv(prefix+'zg-score.csv')

        # precisionDF = pd.read_csv(prefix + 'zprecision.csv')
        mccDF = pd.read_csv(prefix + 'zmcc.csv')

        updateMap(ruleScoreMap, d2hDF, False)
        updateMap(ruleScoreMap, rocDF, True)
        updateMap(ruleScoreMap, ifaDF, False)
        updateMap(ruleScoreMap, brierDF, False)
        updateMap(ruleScoreMap, recallDF, True)
        updateMap(ruleScoreMap, pfDF, False)
        updateMap(ruleScoreMap, gscoreDF, True)

        # updateMap(ruleScoreMap, precisionDF, True)
        updateMap(ruleScoreMap, mccDF, True)



        sortedMap = {k.strip(): v for k, v in sorted(ruleScoreMap.items(), key=lambda item: item[1], reverse=True)}

        print(sortedMap, len(sortedMap))
        writeLatexCSVTable(sortedMap)

EXTRA = 0

if __name__ == '__main__':

    print("All measures")
    masterOfALLMeasures()
