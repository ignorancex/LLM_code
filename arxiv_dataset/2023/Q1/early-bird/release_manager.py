import operator



# from pyclustering.cluster.xmeans import xmeans
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# from pyclustering.utils import read_sample
# from pyclustering.samples.definitions import SIMPLE_SAMPLES

from Util import *
import time
import numpy as np

import feature_selector
import pandas as pd

"""
@author : Shrikanth N C
"""

import glob, os
import  subprocess
from Constants import *
from project_samples import *



def writeReleaseCSV(repos_folder_path):

    projects = os.listdir(repos_folder_path)
    print(projects)

    index = 1
    releaseMap = {}
    for p in projects:
        print(index, p)
        index += 1
        tagsFile = repos_folder_path + p + '/git_tags.txt'
        print("looking.. for ",tagsFile)
        f = open(tagsFile, "r")

        releaseList = []

        for tagEntry in f:

            try:
                tagDateArr = tagEntry[0:tagEntry.index(' (')].strip().split(' ')
                tagDate = tagDateArr[0] + " " + tagDateArr[1]
                tagVersion = tagEntry[tagEntry.index('(tag: ') + 6:tagEntry.index(')')].strip()
            except:
                continue

            releaseList.append(toEpoch(tagDate))

        releaseMap[p] = len(releaseList)

        df = pd.DataFrame()

        releaseList = list(set(releaseList))
        releaseList.sort()
        df['releases'] = releaseList
        df.to_csv('./data/release_info/'+p+".csv", index=False)

    for key, value in releaseMap.items():
        print(key, ',' , value)

def printProjectInfo(projectName):

    p = getProject(projectName)
    releases = p.getReleases()

    print("************** "+projectName+" ***********************")
    for r in releases:
        print('\t',r)
    print("******************************************************")


def getProjectDuration(releasesList):
    pass

def getBugPercentage(xDF):
    # print(xDF)
    return percentage(len(xDF[xDF['Buggy'] > 0]), len(xDF))


def renameColumns(projectNames):

    for p in projectNames:
        try:
            print("trying to rename ",p)
            df = pd.read_csv('./data/'+p+".csv")
            df.rename(columns={'entrophy': 'entropy'}, inplace=True)
            df.to_csv('./data/'+p+".csv", index=False)
            print("renamed ",p)
        except:
            continue

def getBugCount( xDF):

    # shrikanth warn 3

    return len(xDF[xDF['Buggy'] > 0])

def getCleanCount( xDF):

    return len(xDF[xDF['Buggy'] == 0])



def getStars(p):

    df = pd.read_csv('project_stars.csv')
    df = df[ df['name'] == p ]

    if len(df) == 1:
        return df['Stars'].values.tolist()[0]
    else:
        tdf = pd.read_csv('unpopular_projects_130.csv')
        tdf = tdf[tdf['repository'].str.contains('/'+p)]

        if len(tdf) == 1:
            return tdf['stars'].values.tolist()[0]
        elif len(tdf) > 1:
            return tdf['repository'].values.tolist()

        else:
            return None



def printSanitizedProjects():

    projectNames = []
    for file in os.listdir("./data/release_info/"):
        if file.endswith(".csv"):
            projectNames.append(file.replace(".csv", ''))

    """
    Sanity Check
    """
    qualityProjectsMap = {}

    qualifyingProjects = []

    defectPerList = []
    changes = []
    for p in projectNames:

            try:
                projObj = getProject(p)
                releasesList = projObj.getReleases()
            except:
                print(p, 'file not found ')
                continue

            allChanges = projObj.getAllChanges()

            if allChanges is None or len(allChanges) <=0 :
                continue

            numOfChanges = len(allChanges)
            releaseDates = [r.getStartDate() for r in releasesList]
            if releaseDates is None or len(releaseDates) <= 0:
                continue

            projectYrs = abs(min(releaseDates) - max(releaseDates))/one_year

            defect_per = percentage(getBugCount(allChanges), numOfChanges)

            if len(releasesList) > 1 and projectYrs > 1 and defect_per > 1 and getStars(p) > 1000  and numOfChanges > 1:

                changes.append(numOfChanges)

                qualifyingProjects.append(p)
                defectPerList.append(defect_per)
                qualityProjectsMap[p] = defect_per

    print("Project that we mined from commit guru ", len(qualityProjectsMap))

    print("ALL_PROJECTS = ", qualifyingProjects)

    print(len(qualifyingProjects))

    changes.sort()

    print(changes)

    #     if val < medPer:
    #         TEST.append(key)
    #     else:
    #         CROSS.append(key)
    #
    # print('Test List = ',TEST)
    # print('Cross List = ',CROSS)



RUNONARC =  True


UNPOPULAR_NONSANITY = ['ros_ethercat', 'jpush-api-csharp-client', 'metrics-plugins', 'closure-stylesheets', 'tori', 'soot-infoflow-android', 'XBeeJavaLibrary', 'SlipStream', 'seadas', 'apbs-pdb2pqr', 'giswater', 'mne-cpp', 'pressbooks', 'nodeconductor', 'vdsm', 'SuiteCRM', 'heat']

# Shrikanth

TSE_SZZ_PROJECTS = ['openstack_tse', 'qt_tse']

def getProjectNames(projectsType='All_projects'):

    # return
    if not RUNONARC:
        return [ 'numpy' ]
    else:

        if RQ == 1:
            return  UNPOPULAR_PROJECTS
        elif RQ == 5:
            return TSE_SZZ_PROJECTS
        else:
            return POPULAR_PROJECTS + UNPOPULAR_PROJECTS

        # + POPULAR_PROJECTS # UNPOPULAR_PROJECTS # + POPULAR_PROJECTS
        # UNPOPULAR_PROJECTS
        # POPULAR_PROJECTS
        # ['yii', 'vcr', 'taiga-back', 'symfony', 'snp-pipeline', 'snorocket', 'proxygen', 'mailjet-gem', 'deepin-boot-maker', 'Validation']


        # first_50 # lt_1000_stars # UNPOPULAR_PROJECTS


def printSpanRatio():

    tempDF = pd.DataFrame()
    ratio = []
    projectNam = []
    relList = []
    yrList = []
    for p in getProjectNames():
        projObj = getProject(p)

        releasesList = projObj.getReleases()
        releases = len(releasesList)
        releaseDates = [r.getReleaseDate() for r in releasesList]
        projectYrs = float(abs(min(releaseDates) - max(releaseDates))) / float(one_year)
        ratio.append( float(projectYrs)/float(releases))
        projectNam.append(p)
        relList.append(releases)
        yrList.append(projectYrs)

        print(projectNam, ratio)

    tempDF['p'] = projectNam
    tempDF['rat']= ratio
    tempDF['rel'] = relList
    tempDF['yr'] = yrList


    # return ['MaterialDrawer']

    # df = pd.read_csv('projectList.csv')
    # projects =  df[ df['status'] != 'skip' ]['projectName'].values.tolist()
    #
    # nProjects = []
    #
    # rProjects = ['wagtail','realm-java','MaterialDrawer','rasa','Signal-Android','jenkins','HikariCP','hadoop']
    #
    # for p in projects:
    #
    #     if p not in rProjects:
    #         nProjects.append(p)

    # return nProjects

    # # return ['spoon', 'synfig', 'mybatis-3', 'iotivity', 'fresco']
    # return ['rhino', 'gson', 'kafka', 'reddison' , 'thanos', 'ionic', 'apollo']

"""
Constants
"""

# data_attribute = 'author_date'
data_attribute = 'author_date_unix_timestamp'

class project(object):

    def __init__(self, name):
        self.name = name
        self.releases = getReleases(name)

        tempStartDate = math.inf
        tempEndDate = 0

        for r in self.releases:
            tempStartDate = min(tempStartDate, r.getStartDate())
            tempEndDate= max(tempEndDate, r.getReleaseDate())

        self.years = (tempEndDate - tempStartDate)/one_year


    def getYears(self):

        return self.years


    def getReleases(self):
        return self.releases

    def getName(self):
        return self.name

    def getAllChanges(self):


        changesDF = None
        changes = 0

        if len(self.releases) > 0:
            for r in self.releases:

                # print(r.getReleaseDate(), len(r.getChanges()))
                changes += len(r.getChanges())

                if changesDF is None:
                    changesDF = r.getChanges()
                else:
                    changesDF = changesDF.append(r.getChanges())
        else:
            changesDF = formatDF(pd.read_csv('./data/'+self.name+'.csv').copy(deep=True))
            print("Project with no releases but with ", len(changesDF), ' changes!')
        return changesDF


class release(object):

    def __init__(self,release_date, changes, startDate):

        self.release_date = release_date
        self.changes = changes
        self.startDate = startDate


    def getReleaseDate(self):
        return self.release_date

    def getStartDate(self):
        return self.startDate

    def getChanges(self):
        return self.changes

    def __str__(self):
        return str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.release_date)) + " : "+str(
                len(self.changes)))



def getReleasesBefore(project, releaseDate):

    pastReleases = []

    for r in project.getReleases():

        if r.getReleaseDate() < releaseDate:
            pastReleases.append(r)


    return pastReleases

def gapCalculator(project):

    projObj = getProject(project)

    projDF = projObj.getAllChanges()

    trueDF = projDF[projDF['Buggy'] == True ]

    timeDiffList = []

    correct  = 0
    error = 0

    for index, row in trueDF.iterrows():

        hashStr = row['fixes'].replace('[', '')
        hashStr = hashStr.replace(']', '')
        hashStr = hashStr.replace('"', '')
        fixList = hashStr.split(',')

        bugInducingHash = row['commit_hash']
        bugFixingHashes = fixList

        bugInducingTime = projDF[ projDF['commit_hash'] == bugInducingHash ][data_attribute].values.tolist()[0]

        for bugFixingHash in bugFixingHashes:

            blist  = projDF[projDF['commit_hash'] == bugFixingHash][data_attribute].values.tolist()

            if len(blist) != 1:

                continue



            bugFixingTime = blist[0]

            if float(bugFixingTime) - float(bugInducingTime) < 0:

                error += 1
                # print("error >>", float(bugFixingTime) - float(bugInducingTime))

            else:
                correct += 1

                timeDiffList.append(float(bugFixingTime) - float(bugInducingTime))



    import numpy as np

    return  np.median(timeDiffList)

def toDataPoint(release):


    three_D_Point = []
    tempDF = release.getChanges()

    three_D_Point.append(sum([tempDF['la'].sum(), tempDF['ld'].sum()]))
    three_D_Point.append(tempDF['nf'].sum())

    # if tempDF['age'].max() is math.nan:
    #     three_D_Point.append(0)
    # else:
    #     three_D_Point.append(tempDF['age'].max())

    return three_D_Point


def equality(dataPoint1, dataPoint2):

    # print('>>',dataPoint1, dataPoint2)
    index = 0

    for d in dataPoint1:

        if d == dataPoint2[index]:
            index += 1
            continue
        else:
            return False


    return True


def toReleaseObjects(pastReleases, closest_clusters):

    releaseObjects = []

    for pr in pastReleases:

        for cc in closest_clusters :

            if equality(toDataPoint(pr), cc):
                releaseObjects.append(pr)


    return releaseObjects



def toDataPoints(pastReleases):
    three_D_Points = []

    for r in pastReleases:
        three_D_Points.append(  toDataPoint(r)  )

    return three_D_Points


def findSimilarPastRelease(project, currentRelease):

    # print("<>", toDataPoint(currentRelease))
    pastReleases = getReleasesBefore(project, currentRelease)
    # for p in pastReleases:
    #     print(p.getReleaseDate(), toDataPoint(p))
    # #
    if len(pastReleases) > 0:
        return pastReleases
    else:
        return None

    #
    # print("\t past releases = ",pastReleases)

    if len(pastReleases) > 1:

        currentReleaseDataPoint = toDataPoint(currentRelease)
        pastReleaseDataPoints = toDataPoints(pastReleases)

        # print(currentReleaseDataPoint, 'past', pastReleaseDataPoints )

        # print(pastReleaseDataPoints)
        amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(pastReleaseDataPoints, amount_initial_centers).initialize()

        xmeans_instance = xmeans(pastReleaseDataPoints, initial_centers, len(pastReleaseDataPoints))
        result = xmeans_instance.process()
        closest_clusters = result.predict([currentReleaseDataPoint])
        # # print(len(pastReleases), 'cluster = ',closest_clusters)
        #
        ccList = []
        for c in closest_clusters:
            ccList.append(pastReleaseDataPoints[c])
        #
        specificPastReleases = toReleaseObjects(pastReleases, ccList)

        # for s in specificPastReleases:
        #     print('\t sp ',s)
        #
        # # Visualize clustering results
        # visualizer = cluster_visualizer()
        # visualizer.append_clusters(xmeans_instance.get_clusters(), pastReleaseDataPoints)
        # visualizer.append_cluster(xmeans_instance.get_centers(), None, marker='*', markersize=10)
        # visualizer.show()

        return specificPastReleases

    elif len(pastReleases) == 1:
        return [pastReleases[0]]


    return None


# """
# Need to tune this
# """
# def findPastSimilarRelease(allReleases, currentRelease, recursive=False):
#
#     if recursive == False:
#         xmeans_instance = xmeans(allReleases, None, len(allReleases))
#         xmeans_instance.process()
#         closest_clusters = xmeans_instance.predict(currentRelease)
#         return closest_clusters
#     else:
#         closest_clusters = allReleases
#         lastIteration = len(closest_clusters) + 1
#
#         while(len(closest_clusters) > 1 and len(closest_clusters) < lastIteration):
#
#             # print(closest_clusters, lastIteration)
#             lastIteration = len(closest_clusters)
#             xmeans_instance = xmeans(closest_clusters, None, len(closest_clusters))
#             xmeans_instance.process()
#             closest_clusters = xmeans_instance.predict(currentRelease)
#             print(len(closest_clusters))
#
#         return closest_clusters
#
#
#     # print(closest_clusters)
#
#     #
#     #     print(closest_clusters)
#
#     # # Visualize clustering results
#     # visualizer = cluster_visualizer()
#     # visualizer.append_clusters(xmeans_instance.get_clusters(), allReleases)
#     # centers = xmeans_instance.get_centers()
#     # visualizer.append_cluster(centers, None, marker='*', markersize=10)
#     # visualizer.show()
#
#     return closest_clusters

# likes = generateProgramableLikes()
def getReleases(p):


    # print(p)
    df = pd.read_csv('./data/'+p+'.csv')
    before = len(df)

    if p not in TSE_SZZ_PROJECTS:
        df = df[ df['classification'] != 'Merge' ]
        # print("Ignored ", len(df), " merge commits! ", before)


    releaseObjects = []


    prevPeriod = None

    releaseDates = pd.read_csv('./data/release_info/' + p + ".csv")['releases'].values.tolist()
    # print(p, len(releaseDates))

    added = CONSIDER_FIRST_X_RELEASES

    for currentPeriod in releaseDates:

        if added <= 0:
            break

        if prevPeriod is None:
            prevPeriod = currentPeriod
            continue
        else:
            period = [prevPeriod, currentPeriod]
            # print(printPeriod(period))

            tempDF = df[ (df[data_attribute] > prevPeriod) & (df[data_attribute] <= currentPeriod) ]

            # # print(period, df)
            #
            # clusterInfo = []
            # # release.append(tempDF['la'].sum())
            # # release.append(tempDF['ld'].sum())
            # clusterInfo.append(sum([tempDF['la'].sum(), tempDF['ld'].sum()]))
            # clusterInfo.append(tempDF['nf'].sum())
            # clusterInfo.append(tempDF['fixcount'].sum())
            # # release.append(tempDF['age'].mean())

            # if (len(tempDF) > 0 ):

            rDF = formatDF(tempDF)

            if len(rDF) > 1:
                releaseObjects.append(release(currentPeriod, rDF, tempDF['author_date_unix_timestamp'].min()))
                added -= 1

            prevPeriod = currentPeriod


    return releaseObjects

def get_features(df):
    fs = feature_selector.featureSelector()
    df, _feature_nums, features = fs.cfs_bfs(df)
    return df, features


def invoke_tags_sh(repos_folder_path):

    projects = os.listdir(repos_folder_path)
    for p in projects:
        cwd = p
        print('cd ', cwd)
        print('./fetch_tags.sh')
        print('cd ..')



def writeTagsFile(repos_folder_path):

    projects = os.listdir(repos_folder_path)
    pid = 1
    for p in projects:
        tagsFile = repos_folder_path + p + '/fetch_tags.sh'
        f = open(tagsFile, "w")
        f.write('git log --tags --simplify-by-decoration --pretty="format:%ai %d" > git_tags.txt')
        f.close()
        print("written , ",p, pid)
        pid +=1


def format_qt_openstack(rdf):

    releaseDF = rdf.copy(deep=True)
    # releaseDF = releaseDF[['la', 'ld', 'nf', 'nd', 'ns', 'ent', 'revd', 'nrev', 'rtime', 'tcmt', 'hcmt', 'self',
    #                        'ndev', 'age', 'nuc', 'app', 'aexp', 'rexp', 'oexp', 'arexp', 'rrexp', 'orexp', 'asexp',
    #                        'rsexp', 'osexp', 'asawr', 'rsawr', 'osawr',
    #                        'commit_id', 'author_date', 'fixcount',
    #                        'bugcount']]
    # releaseDF = releaseDF.drop(labels=['commit_id',  'fixcount'], axis=1)
    releaseDF = releaseDF.fillna(0)
    releaseDF.rename(columns={'bugcount': 'contains_bug'}, inplace=True)
    releaseDF.rename(columns={'author_date': 'author_date_unix_timestamp'}, inplace=True)
    # releaseDF.rename(columns={'fixcount': 'fix'}, inplace=True)
    releaseDF.rename(columns={'ent': 'entropy'}, inplace=True)

    # releaseDF.rename(columns={'aexp': 'exp'}, inplace=True)
    # releaseDF.rename(columns={'nd': 'lt'}, inplace=True)
    #



    return releaseDF

def formatDF(rdf):

    """
    Works for QT and OPEN-STACK
    releaseDF = rdf.copy(deep=True)
    releaseDF = releaseDF[['la','ld','nf','nd','ns','ent','revd','nrev','rtime','tcmt','hcmt','self',
                    'ndev','age','nuc','app','aexp','rexp','oexp','arexp','rrexp','orexp','asexp','rsexp','osexp','asawr','rsawr','osawr',
                    'commit_id','author_date','fixcount',
                    'bugcount']]
    releaseDF = releaseDF.drop(labels=['commit_id','author_date','fixcount'], axis=1)
    releaseDF = releaseDF.fillna(0)
    releaseDF.rename(columns={'bugcount': 'Buggy'}, inplace=True)
    """

    releaseDF = rdf.copy(deep=True)

    # print(releaseDF)


    features = []

    # releaseDF = releaseDF[[ 'la', 'ld', 'nf', 'nd', 'ns', 'ent', 'revd', 'nrev', 'rtime', 'tcmt', 'hcmt', 'self',
    #                        'ndev', 'age', 'nuc', 'app', 'aexp', 'rexp', 'oexp', 'arexp', 'rrexp', 'orexp', 'asexp',
    #                        'rsexp', 'osexp', 'asawr', 'rsawr', 'osawr',
    #                        'commit_id', 'author_date', 'fixcount',
    #                        'bugcount' ]]

    # commit_hash, author_name, author_date_unix_timestamp, author_email, author_date, commit_message, fix, classification, linked, contains_bug, fixes, ns, nd, nf, entropy, la, ld, fileschanged, lt, ndev, age, nuc, exp, rexp, sexp, glm_probability, rf_probability, repository_id, issue_id, issue_date, issue_type

    # ns, nd, nf, entropy, la, ld, fileschanged, lt, ndev, age, nuc, exp, rexp, sexp,


    try:
        releaseDF = releaseDF[
            ['author_date_unix_timestamp', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp',
             'rexp', 'sexp', 'fix', 'contains_bug']]
        dropList = ['author_date_unix_timestamp']
    except Exception as e:




        dropList = ['author_date_unix_timestamp', 'commit_id', 'revd']
        releaseDF.rename(columns={'fix': 'fixcount'}, inplace=True)

        releaseDF = releaseDF[ [col for col in releaseDF.columns if col != 'contains_bug'] + ['contains_bug']]

        releaseDF['contains_bug'] = [x > 0 for x in releaseDF['contains_bug'].values.tolist()]








    # print('Warn drop author_date_unix_timestamp')
    # dropList = []

    releaseDF = releaseDF.drop(labels=dropList, axis=1)
    releaseDF = releaseDF.fillna(0)

    # print("warning.. 2")
    releaseDF = releaseDF[ (releaseDF['contains_bug'] == True) | (releaseDF['contains_bug'] == False) ]

    # releaseDF = releaseDF[ releaseDF['la'] + releaseDF['ld'] > 0]

    # releaseDF.rename(columns={'bugcount': 'Buggy'}, inplace=True)

    releaseDF.rename(columns={'contains_bug': 'Buggy'}, inplace=True)

    releaseDF.loc[(releaseDF['Buggy'] == True) | (releaseDF['Buggy'] >= 1), 'Buggy'] = True
    releaseDF.loc[(releaseDF['Buggy'] == False) | (releaseDF['Buggy'] <= 0), 'Buggy'] = False

    # print(releaseDF.columns.values.tolist())

    # releaseDF.to_csv('test.csv', index=False)

    return releaseDF

def getUnPopularProjects():

    df = pd.read_csv('project_stars.csv')
    df = df[ df['Stars'] < 1000]

    return df['name'].values.tolist()



def getReleaseObjCopy2(projectObj, releaseDate):
    allReleases = projectObj.getReleases()

    for r in allReleases:

        if r.getReleaseDate() == releaseDate:
            return r

    return None

def getReleaseObjCopy(projectName, releaseDate):

    allReleases = getReleases(projectName)

    for r in allReleases:

        if r.getReleaseDate() == releaseDate:
            return r


    return None


def getProject(p):
    return project(p)


def split(pname, months):
    p = project(pname)


def writeReleaseInfo(projectName):
    tagsFile = './data/'+projectName+"_tags.txt"
    f = open(tagsFile, "r")

    df = pd.DataFrame()

    row = []
    for tagEntry in f:

        try:
            tagDateArr = tagEntry[0:tagEntry.index(' (')].strip().split(' ')
            tagDate = tagDateArr[0] + " " + tagDateArr[1]
            tagVersion = tagEntry[tagEntry.index('(tag: ') + 6:tagEntry.index(')')].strip()
        except:
            continue

        row.append(toEpoch(tagDate))


    row.sort()
    df['releases'] = row
    df.to_csv('./data/release_info/'+projectName+".csv", index=False)

def getCorrelations():

    correls = []

    decayProjects = []
    growProjects = []
    neutralProjects = []

    for p in getProjectNames():

        projObj = getProject(p)

        releaseList = projObj.getReleases()

        X = []
        Y = []
        for r in releaseList:

            X.append(r.getReleaseDate())
            Y.append(getBugCount(r.getChanges()))

        rhoList = computeCorrelation(Y, X)

        # if rhoList[3] < SIG_LEVEL:
        #     correls.append(rhoList[2])
        #     #
        #     print(p, rhoList[2])

        print(p, rhoList[2], len(releaseList))

        if rhoList[2] <= -0.39:
            decayProjects.append(p)
        elif rhoList[2] >= 0.39:
            growProjects.append(p)
        else:
            neutralProjects.append(p)


    print("Decay = ", ["'"+p+"'," for p in decayProjects],'')
    print("Growth =  ", ["'" + p + "'," for p in growProjects],'')
    print("Neutral =  ", ["'" + p + "'," for p in neutralProjects],'')

    return correls


def printTrend(p):

    projObj = getProject(p)

    releaseList = projObj.getReleases()

    tDF = pd.DataFrame()
    rdates = []
    bugs = []
    for r in releaseList:

        bugs.append(getBugCount(r.getChanges()))
        rdates.append(r.getReleaseDate())


    tDF['rdates'] = rdates
    tDF['bugs'] = bugs

    tDF.to_csv('ztrend.csv', index=False)


def drM():
    rhoList = []
    pvalues = []

    pnames = []

    tdf = pd.DataFrame()


    for p in getProjectNames():

        proj = getProject(p)

        releaseList = proj.getReleases()

        rDate = []
        bugCount = []



        pnames.append(p)

        for r in releaseList:

            changes = r.getChanges()

            rDate.append(r.getReleaseDate())
            bugCount.append(getBugCount(changes))


        a,b,c,d = computeCorrelation(rDate, bugCount)

        rhoList.append(c)
        pvalues.append(d)

    tdf['p'] =pnames
    tdf['rho'] = rhoList
    tdf['pvalue'] = pvalues

    tdf.to_csv("ztrend.csv", index=False)



def latexAboutProjects():

    totalCommits = 0
    totalReleases = 0
    minTime = 1583023088
    maxTime = -1

    for p in getProjectNames():

        pobj = getProject(p)

        releaseList = pobj.getReleases()

        for r in releaseList:

             totalReleases += 1

             releaseChanges = len(r.getChanges())

             if releaseChanges > 0:

                if r.getReleaseDate() == 838949040:
                    print(p , "really ?")
                    continue
                minTime = min(minTime, r.getReleaseDate())
                totalCommits += releaseChanges
                maxTime = max(maxTime, r.getReleaseDate())



    print(totalCommits, totalReleases, minTime, maxTime)




def analyze(minBug, minClean):

    bugs = []
    clean = []
    covered = 0

    for p in getProjectNames():

        projectObj = getProject(p)
        projectChanges = projectObj.getAllChanges()
        trainingRegion = projectChanges.head(150).copy(deep=True)
        buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
        nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

        if len(buggyChangesDF) >= minBug and len(nonBuggyChangesDF) > minClean:
            covered += 1

        bugs.append(len(buggyChangesDF))
        clean.append(len(nonBuggyChangesDF))
        # try:
        #     print("Bugs = ", min(bugs), np.median(bugs), max(bugs))
        #     print("Clean = ", min(clean), np.median(clean), max(clean))
        # except:
        #     print('skip')
        #     continue

    print("Overall Bugs = ", min(bugs),  np.median(bugs), max(bugs))
    print("Overall Clean = ", min(clean),  np.median(clean), max(clean))

    print("Coverage % ", minBug, minClean, '=>', percentage(covered, len(getProjectNames())))


def bugsPerRelease():

    bugsInEachRelease = []
    consideredRelease = 0
    notConsideredRElease = 0
    changesPerRelease = []

    for p in getProjectNames():

        projectObj = getProject(p)

        rels = projectObj.getReleases()

        for r in rels:

            changesPerRelease.append(len(r.getChanges()))
            bugcount = getBugCount(r.getChanges())

            if bugcount > 5:
                bugsInEachRelease.append(bugcount)
                consideredRelease += 1
            else:
                notConsideredRElease += 1

            print(consideredRelease, notConsideredRElease)

            try:
                print("Changes per release = ", min(changesPerRelease), np.median(changesPerRelease),
                      max(changesPerRelease))
            except:
                continue


    print("Overall Bugs per release = ", min(bugsInEachRelease) , np.median(bugsInEachRelease) , max(bugsInEachRelease) )
    print("Overall ",consideredRelease, notConsideredRElease)
    print("Overall changes per release = ", min(changesPerRelease), np.median(changesPerRelease), max(changesPerRelease))


def firstReleaseDuration():

    releaseDuration = []

    for p in getProjectNames():

        projectObj = getProject(p)

        firstRelease = projectObj.getReleases()[0]

        bugcount = getBugCount(firstRelease.getChanges())

        if bugcount > 5:
            releaseDuration.append(firstRelease.getReleaseDate() - firstRelease.getStartDate())

        print(releaseDuration)

    print("Overall Bugs = ", min(releaseDuration)/one_month, np.median(releaseDuration)/one_month, max(releaseDuration)/one_month)


def extractReleases(projectsFolder):
    writeTagsFile(projectsFolder)
    invoke_tags_sh(projectsFolder)
    writeReleaseCSV(projectsFolder)


def printValidProjects():

    filtered_projects = []
    for p in getProjectNames():

        projObj = getProject(p)
        bugCount = getBugCount(projObj.getAllChanges())

        if bugCount < 25:
            continue

        if len(projObj.getReleases()) < 2:
            continue

        filtered_projects.append(p)

    print(filtered_projects)



def report_goal_scores1():

    type = None

    a = []
    b = []
    c = []

    for projectName in ['ionic']:

            projObj = getProject(projectName)
            projectChanges = projObj.getAllChanges()
            trainingRegion = projectChanges.head(150).copy(deep=True)

            buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
            nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

            print(len(buggyChangesDF), len(nonBuggyChangesDF))

            print(buggyChangesDF.sample(25))

def getReleasesAfter(commits, releaseList):

    releases = []
    changes  = 0

    for r in releaseList:

        if changes >= commits:
            releases.append(r)

        changes += len(r.getChanges())

    return releases


def process_qt_openstack():

    df = format_qt_openstack(pd.read_csv('./data/szz_threat/openstack.csv'))
    df.to_csv('./data/openstack_tse.csv', index=False)

    df = format_qt_openstack(pd.read_csv('./data/szz_threat/qt.csv'))
    df.to_csv('./data/qt_tse.csv', index=False)






if __name__ == '__main__':


    p = getProject('qt_tse')
    print(len(p.getAllChanges()))

    train_limit = 0
    bc = 0
    for r in p.getReleases():

        c = r.getChanges()
        train_limit += len(c)
        bc += getBugCount(c)

        if bc >= 25:
            print(train_limit)
            break

        print(train_limit)





