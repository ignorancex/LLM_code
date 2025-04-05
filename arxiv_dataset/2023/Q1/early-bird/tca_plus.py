import tca
import DCV
from release_manager import *
import preparation

def data_extend(filename):
    train_data = []
    train_target = []

    f = open(filename,'r')
    dataReader = csv.reader(f)
    for row in dataReader:
        train_data.append(list(map(float,row[3:-1])))
        train_target.append(float(row[-1]))

    f.close()

    train_data = np.array(train_data)

    temp = []
    for i in train_target:
        if i != 0:
            temp.append(1)
        else:
            temp.append(0)

    train_target = np.array(temp)

    return [train_data, train_target]


def apply_tcaplus(train_data, test_data, dimensions):

    train_bugs = train_data.Buggy.values.tolist()
    test_bugs = test_data.Buggy.values.tolist()

    train_data = train_data.drop(labels=['Buggy'], axis=1)

    if 'fix' in train_data.columns:
        train_data = train_data.drop(labels=['fix'], axis=1)

    test_data = test_data.drop(labels=['Buggy'], axis=1)

    if 'fix' in test_data.columns:
        test_data = test_data.drop(labels=['fix'], axis=1)

    rule = DCV.DCV(train_data, test_data)

    if rule==1:
        train_data_std = train_data
        test_data_std = test_data
        print("rule 1")
    elif rule==2:
        train_data_std, test_data_std = preparation.standardization_SVL(train_data,test_data,"0-1scale")
        print("rule 2")
    elif rule==3:
        train_data_std, test_data_std = preparation.standardization_rule3(train_data,test_data)
        print("rule 3")
    elif rule==4:
        train_data_std, test_data_std = preparation.standardization_rule4(train_data,test_data)
        print("rule 4")
    else:
        train_data_std, test_data_std = preparation.standardization_SVL(train_data,test_data,"z-score")
        print("rule 5")

    flag = 0
    for train_row in train_data_std:
        for ele in train_row:
            if math.isnan(ele):
                flag = 1
                break
        if flag==1:
            break

    if flag==1:
        print('Preparation error it contains nan')
        # preparation_error_list.append(times)
        # continue

    for test_row in test_data_std:
        for ele in test_row:
            if math.isnan(ele):
                flag = 1
                break
        if flag==1:
            break
    if flag==1:
        print('Preparation error it contains nan')
        # preparation_error_list.append(times)
        # continue


    model = tca.TCA(dim=dimensions,kerneltype='linear',kernelparam=1,mu=1)
    train_data_std, test_data_std, something = model.fit_transform(train_data_std, test_data_std)

    # print("Train data", train_data_std)
    # print("Test data", train_data_std)
    # print("something", something)

    print(type(train_data_std), type(test_data_std))

    train_df = pd.DataFrame(data=train_data_std,   columns=['Col_' + str(x) for x in range(0, dimensions)])
    train_df['Buggy'] = train_bugs

    test_df = pd.DataFrame(data=test_data_std,  columns=['Col_' + str(x) for x in range(0, dimensions)])
    test_df['Buggy'] = test_bugs

    return train_df, test_df


def run(p, BELLWETHER_PROJECT, TCA_DATA_FOLDER):

    trainChanges = getProject(BELLWETHER_PROJECT).getAllChanges()

    # for p in release_manager.getProjectNames():

    project_obj = getProject(p)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    for testReleaseObj in testReleaseList:

        if path.exists('./'+TCA_DATA_FOLDER+'/train_' + BELLWETHER_PROJECT + '_' + p + '_' + str(
                testReleaseObj.getReleaseDate()) + '.csv'):
            print('continue')
            continue

        info = p + '_' + str(testReleaseObj.getReleaseDate())

        try:
            train_transformed, test_transformed = apply_tcaplus(trainChanges.copy(deep=True),
                                                                testReleaseObj.getChanges().copy(deep=True))

            train_transformed.to_csv('./'+TCA_DATA_FOLDER+'/train_' + BELLWETHER_PROJECT + '_' + p + '_' + str(
                testReleaseObj.getReleaseDate()) + '.csv', index=False)

            test_transformed.to_csv('./'+TCA_DATA_FOLDER+'/test_' + BELLWETHER_PROJECT + '_' + p + '_' + str(
                testReleaseObj.getReleaseDate()) + '.csv', index=False)

        except Exception as e:
            print('\n >>TCA ERROR<< : ', e, info)