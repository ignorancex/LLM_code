import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


#####################################
# translate Yale_Acct_Info to panel.npy data
def translate(df, N, T):
    # date = df['As_of_date'][0:T]
    feature = [	'Age_Cd',	'total_debit_amt',	'se_spend',	'total_remit_amt',
                   'total_bal_amt',	'cash_bal_amt',	'total_bal_fee_amt',
                   'total_bal_intr_amt',	'total_fees_assessed_amt',	'total_fin_chg_amt',
                   'total_disc_rev_amt', 'RAM_Amount']
    panel = np.array([[[0.0 for fea in range(len(feature))] for t in range(T)] for i in range(N)])
    df_stat = df.describe()
    stat = [[0.0 for i in range(2)] for j in range(len(feature))]
    for fea in range(len(feature)):
        # min of the feature
        stat[fea][0] = df_stat[feature[fea]][3]
        # max of the feature
        stat[fea][1] = df_stat[feature[fea]][7]

    index = 0
    num_individual = 0
    while num_individual < N:
        flag = 0
        while flag == 0:
            id = df['Accountnum'][index]
            # check whether there are T time periods for this individual
            symbol = 0
            for i in range(index, index + T):
                if df['Accountnum'][i] != id:
                    index = i
                    symbol = 1
                    break
            # check whether all features have value
            if symbol == 0:
                for i in range(index, index + T):
                    for fea in range(len(feature)-1):
                        if np.isnan(df[feature[fea]][i]):
                            symbol = 1
                            break
                    if symbol == 1:
                        index += T
                        break
            if symbol == 0:
                flag = 1

        # record the information of an individual into panel.npy data
        y_mean = np.mean(df['RAM_Amount'][index: index + 55])
        for i in range(index, index + 55):
            for fea in range(len(feature)):
                # deal with RAM_Amount
                if (feature[fea] == 'RAM_Amount') & np.isnan(df['RAM_Amount'][i]):
                    temp = y_mean
                else:
                    temp = df[feature[fea]][i]
                # normalization
                temp -= stat[fea][0]
                temp /= stat[fea][1] - stat[fea][0]
                # record into panel.npy data
                panel[num_individual][i - index][fea] = temp
        index += T
        num_individual += 1

    np.save("panel_Yale_Acct", panel)
    return panel


#########################################################################################################
# count #NaN in the data
def count_nan(df, T):
    count = {}
    times = {}
    start = {}
    count[''] = 0
    times[''] = 0
    start[''] = 0

    for i in range(T):
        count[df['As_of_date'][i]] = 0
        times[df['As_of_date'][i]] = 0
        start[df['As_of_date'][i]] = 0

    start[df['As_of_date'][0]] += 1
    for i in range(len(df)):
        times[df['As_of_date'][i]] += 1
        if np.isnan(df['RAM_Amount'][i]):
            count[df['As_of_date'][i]] += 1
        if (df['As_of_date'][i] == '') & (i < len(df)-1):
            start[df['As_of_date'][i+1]] += 1

    print(count)
    print(times)
    print(start)
    return


#####################################
# clean the data
def clean(df,T):
    # delete those rows with null data
    delete_list = []
    for i in range(len(df)):
        if np.isnan(df['RAM_Amount'][i]) or (df['As_of_date'][i] == '') or (df['As_of_date'][i] == '2006-01-01'):
            delete_list.append(i)
    np_values = df.values
    np_new_values = np.delete(np_values, delete_list, 0)
    df1 = pd.DataFrame(np_new_values, columns = df.columns)
    del df

    print('df1:', df1.shape)

    N = len(df1)

    clean_list = []
    feature = [	'Age_Cd',	'total_debit_amt',	'se_spend',	'total_remit_amt',
                   'total_bal_amt',	'cash_bal_amt',	'total_bal_fee_amt',
                   'total_bal_intr_amt',	'total_fees_assessed_amt',	'total_fin_chg_amt',
                   'total_disc_rev_amt', 'RAM_Amount']

    index = 0
    while index < N:
        id = df1['Accountnum'][index]
        # check whether there are T time periods for this individual
        symbol = 0
        for i in range(index, min(N, index + T)):
            if (df1['Accountnum'][i] != id):
                for j in range(index, i):
                    clean_list.append(j)
                index = i
                symbol = 1
                break
        # check whether all features have value
        if symbol == 0:
            for i in range(index, min(N, index + T)):
                for fea in range(len(feature)):
                    if np.isnan(df1[feature[fea]][i]):
                        symbol = 1
                        break
                if symbol == 1:
                    for j in range(index, index + T):
                        clean_list.append(j)
                    index += T
                    break
        if symbol == 0:
            index += T

    # clean individuals that does not start from '2016-02-01' or contain NaN
    np_values = df1.values
    np_new_values = np.delete(np_values, clean_list, 0)
    df2 = pd.DataFrame(np_new_values, columns=df1.columns)
    del df1

    print('df2:', df2.shape)

    df3 = df2[feature]
    df3 = df3.apply(pd.to_numeric)
    del df2
    return df3

def clean2(df3,T, sample_N, signal):
    # feature statistics before normalization
    feature = ['Age_Cd', 'total_debit_amt', 'se_spend', 'total_remit_amt',
               'total_bal_amt', 'cash_bal_amt', 'total_bal_fee_amt',
               'total_bal_intr_amt', 'total_fees_assessed_amt', 'total_fin_chg_amt',
               'total_disc_rev_amt', 'RAM_Amount']
    df3_stat = df3.describe()
    with pd.ExcelWriter('cleaned_feastatistic_Yale_Acct.xlsx', mode = 'a') as writer:
        df3_stat.to_excel(writer, sheet_name = 'before_normalization')
    print('df3_stat (before):', df3_stat[4:7])

    # if signal == 1, remove outliers
    if signal == 1:
        clean_list = []
        N = len(df3)
        index = 0
        while index < N:
            # check whether the individual is an outlier
            symbol = 0
            for i in range(index, min(index + T, N)):
                for fea in feature:
                    if (abs(df3[fea][i] - df3_stat[fea][1]) > 10*df3_stat[fea][2]):
                        symbol = 1
                        break
                if symbol == 1:
                    for j in range(index, min(index + T, N)):
                        clean_list.append(j)
                    index += T
                    break
            if symbol == 0:
                index += T

        # clean individuals that are outliers
        np_values = df3.values
        np_new_values = np.delete(np_values, clean_list, 0)
        df3 = pd.DataFrame(np_new_values, columns=df3.columns)
        # df3.to_csv('RO_Yale_Acct_Info.csv')
        print('Remaining individuals:', int(len(df3) / T))

        # feature statistic without outliers
        df3_stat = df3.describe()
        with pd.ExcelWriter('cleaned_feastatistic_Yale_Acct.xlsx', mode='a') as writer:
            df3_stat.to_excel(writer, sheet_name='before_normalization (RO)')
        print('df3_stat (RO, before):', df3_stat.shape)


    stat = [[0.0 for i in range(2)] for j in range(len(feature))]
    for fea in range(len(feature)):
        # min of the feature
        stat[fea][0] = df3_stat[feature[fea]][3]
        # max of the feature
        stat[fea][1] = df3_stat[feature[fea]][7]

    # normalization
    for fea in range(len(feature)):
        if stat[fea][1] == stat[fea][0]:
            df3[feature[fea]] = 0
        else:
            # normalization
            df3[feature[fea]] -= stat[fea][0]
            df3[feature[fea]] /= (stat[fea][1] - stat[fea][0])


    print('Normalized.')

    # feature statistic after normalization
    if signal == 1:
        df3.to_csv('ROcleaned_Yale_Acct_Info.csv')
    elif signal == 0:
        df3.to_csv('cleaned_Yale_Acct_Info.csv')
    print('df3:', df3.shape)

    df3_stat = df3.describe()
    with pd.ExcelWriter('cleaned_feastatistic_Yale_Acct.xlsx', mode = 'a') as writer:
        if signal == 1:
            df3_stat.to_excel(writer, sheet_name='after_normalization (RO)')
            print('df3.stat (RO, after):', df3_stat.shape)
        elif signal == 0:
            df3_stat.to_excel(writer, sheet_name = 'after_normalization')
            print('df3.stat (after):', df3_stat.shape)


    # generate a panel.npy dataset
    N = int(len(df3) / T)
    sample = np.random.choice(range(N), size=sample_N, replace = False)

    print(N, np.min(sample), np.max(sample))

    panel = np.array([[[0.0 for fea in range(len(feature))] for t in range(T)] for i in range(sample_N)])
    for i in range(sample_N):
        for t in range(T):
            for fea in range(len(feature)):
                panel[i][t][fea] = df3[feature[fea]][T * sample[i] + t]
    if signal == 1:
        np.save("ROpanel_Yale_Acct", panel)
    elif signal == 0:
        np.save("panel_Yale_Acct", panel)
    return


#####################################
if __name__ == "__main__":
    df = pd.read_stata('yale_data/Yale_Acct_Info.dta')
    df_stat = df.describe()
    df_stat.to_excel('cleaned_feastatistic_Yale_Acct.xlsx', sheet_name='all_features')
    #
    temp_df = clean(df,50)
    clean2(temp_df,50,5000, 0)
    # translate(df, 5000, 55)
    # panel.npy = np.load('panel_Yale_Acct.npy')
    # print(panel.npy.shape)