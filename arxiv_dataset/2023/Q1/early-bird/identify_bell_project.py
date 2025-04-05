from rq_run import *
from release_manager import *



"""
@author: Shrikanth N C
"""

def write_bell_csv(folder, completed_projects):

    print("Trying ", folder)



    global_gscore = []
    global_recall = []
    global_pf = []
    global_population = []
    global_project = []
    type = []
    commits = []

    for cross_project in  getProjectNames():

        cross_gscore = []
        cross_recall = []
        cross_pf = []


        try:
            for target_project in   completed_projects:

                if cross_project == target_project:
                    continue

                if 'Early' in folder:
                    suffix = '_E_CHANGES'
                else:
                    suffix = '_ALL_CHANGES'

                df = pd.read_csv('./'+folder+'/project_' + target_project + '_results.csv')
                cross_gscore.append( df [ (df['trainApproach'] == cross_project+suffix) &
                                          (df['classifier'] == 'LogisticRegression')]['g-score'].median())
                cross_recall.append(df[(df['trainApproach'] == cross_project + suffix)  &
                                          (df['classifier'] == 'LogisticRegression')]['recall'].median())
                cross_pf.append(df[(df['trainApproach'] == cross_project + suffix)  &
                                          (df['classifier'] == 'LogisticRegression')]['pf'].median())


            print(cross_project, np.median(cross_gscore), np.median(cross_recall), np.median(cross_pf), len(cross_pf))

            global_gscore.append(np.median(cross_gscore))
            global_pf.append(np.median(cross_pf))
            global_recall.append(np.median(cross_recall))
            global_population.append(len(cross_pf))
            global_project.append(cross_project)

            if cross_project in POPULAR_PROJECTS:
                type.append('popular')
            else:
                type.append('unpopular')

            commits.append(len(getProject(cross_project).getAllChanges()))


        except Exception as e:
            print(cross_project, e)

    new_df = pd.DataFrame()
    new_df['repo'] = global_project
    new_df['gscore'] = global_gscore
    new_df['recall'] = global_recall
    new_df['pf'] = global_pf
    new_df['population'] = global_population
    new_df['type'] = type
    new_df['commits'] = commits

    if 'Early' in folder:
        new_df.to_csv('bell_identify_early.csv', index=False)
    else:
        new_df.to_csv('bell_identify_all.csv', index=False)












# def print_bell():
#     for p in getProjectNames():
#
#         df = pd.read_csv('./results/project_'+p+'_results.csv')
#
#         progress.append(
#             len(df['trainApproach'].unique())
#         )
#
#         print(p, len(df['trainApproach'].unique()))


# def print_progress():
#
#     progress = []
#     for p in getProjectNames():
#
#         df = pd.read_csv('./results/project_'+p+'_results.csv')
#         progress.append(
#             len(df['trainApproach'].unique())
#         )
#
#         if len(df['trainApproach'].unique()) == 1:
#             print(p, len(df['trainApproach'].unique()))
#
#     progress.sort()


def get_test_releases(cross_project, target_project):

    temp_df = pd.read_csv('./'+RESULTS_FOLDER+'/project_'+target_project+'_results.csv', error_bad_lines=False)


    if 'Early' in RESULTS_FOLDER:
        temp_df = temp_df [ (temp_df['trainApproach'] == cross_project+'_E_CHANGES')  ]
    else:
        temp_df = temp_df[(temp_df['trainApproach'] == cross_project + '_ALL_CHANGES')  ]

    return temp_df['testReleaseDate'].values.tolist()

def alreadyRan(cross_project, target_project, releaseDate):

    temp_df = pd.read_csv('./'+RESULTS_FOLDER+'/project_'+target_project+'_results.csv')

    if 'Early' in RESULTS_FOLDER:
        temp_df = temp_df [ (temp_df['trainApproach'] == cross_project+'_E_CHANGES') & (temp_df['testReleaseDate']
                                                                       == releaseDate) ]
    else:
        temp_df = temp_df[(temp_df['trainApproach'] == cross_project + '_ALL_CHANGES') & (temp_df['testReleaseDate']
                                                                         == releaseDate)]
    return len(temp_df) > 0


def print_missing(xx, param):


    y = [ x +'_E_CHANGES' for x in getProjectNames()]

    for p in y:
        if p not in param and p not in [ xx+'_E_CHANGES'] :
                print('\t missing = ', p)




def get_in_complete_projects(folder):
    pcountMap = {}
    completed_projects = []
    c = 0
    for p in getProjectNames():

        print("Checking ",p)

        try:
            projectdf = pd.read_csv('./'+folder+'/project_' + p + '_results.csv', error_bad_lines=False)
        except Exception as e:
            print(e)
            pcountMap[p] = 0
            continue

        pcountMap[p] = len(projectdf['trainApproach'].unique())
        if (len(projectdf['trainApproach'].unique())) >= 238:
            print_missing(p, projectdf['trainApproach'].unique())
            c += 1
            completed_projects.append(p)

        else:
            # completed_projects.append(p)
            print('incomplete ', p, pcountMap[p])


    s = dict(sorted(pcountMap.items(), key=lambda item: item[1]))

    print("sorted = ", s)

    print(folder, completed_projects)

    return completed_projects

def run_all_pairs(target_project):

    for cross_project in getProjectNames():

        cross_project_obj = getProject(cross_project)

        cross_all = cross_project_obj.getAllChanges().copy(deep=True)
        early_cross_changes = early_sample(cross_all.head(150))

        if cross_project == target_project:
            continue

        target_project_obj = getProject(target_project)

        start = None
        processed = 0

        already_completed = get_test_releases(cross_project, target_project)

        for test_release in  target_project_obj.getReleases():

            if test_release.getReleaseDate() in already_completed:
                print(cross_project, 'skipping')
                continue

            if start is None:
                start = test_release.getStartDate()
            else:
                if 'Early' in RESULTS_FOLDER and (test_release.getStartDate() - start) > 2 * one_year and processed > 2:
                    print("Not Breaking!")
            try:
                if 'Early' in RESULTS_FOLDER:
                    performPredictionRunner(target_project, early_cross_changes.copy(deep=True),
                                test_release.getChanges().copy(deep=True), 'NA', test_release.getReleaseDate(), cross_project+'_E_CHANGES' ,None, 'size')
                    processed += 1
                else:
                    performPredictionRunner(target_project, cross_all.copy(deep=True),
                                            test_release.getChanges().copy(deep=True), 'NA',
                                            test_release.getReleaseDate(), cross_project + '_ALL_CHANGES', None)


                # print("ALL_PAIRS NEW TIME", goodtime.time())
            except Exception as e:
                traceback.print_exc()
                print("Error processing release ", target_project, cross_project, test_release.getReleaseDate(), e)


def fix(folder):

    for p in getProjectNames():
        try:
            projectdf = pd.read_csv('./' + folder + '/project_' + p + '_results.csv', error_bad_lines=False)
            projectdf.to_csv('./' + folder + '/project_' + p + '_results.csv', index=False)

        except Exception as e:
            print(p, e)


    print("done")


def print_stats():

    early_df = pd.read_csv('bell_identify_early.csv')
    all_df = pd.read_csv('bell_identify_all.csv')

    early_df = early_df[ (early_df['recall'] > 0.69) & (early_df['pf'] < 0.31)  ]
    all_df = all_df[(all_df['recall'] > 0.69) & (all_df['pf'] < 0.31)  ]

    print('bell ', 'early = ', len(early_df), 'all = ', len(all_df))

    c = list(set(early_df['repo'].values.tolist()) & set(all_df['repo'].values.tolist()))

    print(c)

    cp = 0
    for cc in c:
        if cc in POPULAR_PROJECTS:
            cp += 1

    print("Common popular = ", cp, 'unpopular =  ', len(c) - cp)




    print(c)
    print('common = ', len( c))

    print("early popular = ", len(early_df[early_df['type'] == 'popular']), 'early unpopular = ',len(early_df[early_df['type'] == 'unpopular'])  )

    print("all popular = ", len(all_df[all_df['type'] == 'popular']), 'all unpopular = ',           len(all_df[all_df['type'] == 'unpopular']))

    print("early popular = ", early_df[early_df['type'] == 'popular']['repo'].values.tolist())
    print("early unpopular = ", early_df[early_df['type'] == 'unpopular']['repo'].values.tolist())

    print("all popular = ", all_df[all_df['type'] == 'popular']['repo'].values.tolist())
    print("all unpopular = ", all_df[all_df['type'] == 'unpopular']['repo'].values.tolist())


if __name__ == '__main__':

    print_stats()

    # fix('results_TTD_ALL_PAIRS_Early')
    # fix('results_TTD_ALL_PAIRS')

    # completed_projects = get_in_complete_projects('results_TTD_ALL_PAIRS_Early')
    # write_bell_csv('results_TTD_ALL_PAIRS_Early', completed_projects)

    # completed_projects = get_in_complete_projects('results_TTD_ALL_PAIRS')
    # write_bell_csv('results_TTD_ALL_PAIRS', completed_projects)









