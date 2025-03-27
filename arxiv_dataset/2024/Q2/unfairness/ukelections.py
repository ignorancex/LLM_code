import pandas as pd
import numpy as np
import scipy.io as sio

if __name__ == "__main__":
    CR = 'country/region'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    xl = pd.read_excel('UK1918_2019election_results.xlsx')
    df = pd.DataFrame(xl, columns=['constituency', CR, 'con_votes', 'lib_votes', 'lab_votes', 'oth_votes',
                                   'turnout', 'election', 'total_votes'])
    #total votes and share columns turn out to be unreliable, I ignore them except for removing rows

    df['origindex'] = df.index

    all_votes = ['con_votes', 'lib_votes', 'lab_votes', 'oth_votes']

    df['constituency']=df['constituency'].str.upper()
    df[CR]=df[CR].str.upper()
    df = df[~df['turnout'].isnull()]

    df = df[~df['total_votes'].isnull()]
    #print(df[df[all_vote_votess].sum(1) == 0])


    df['locstr']=df['constituency']+'+'+df[CR]
    #translate each county to a unique integer
    locdict = {k: v for v, k in enumerate(set(df['locstr']))}
    df['locid']=[locdict[x] for x in df['locstr']]


    df.loc[df['election'] == '1974F', 'election'] = 1974
    df.loc[df['election'] == '1974O', 'election'] = 1975

    df['con_votes'] = df['con_votes'].fillna(0)
    df['lib_votes'] = df['lib_votes'].fillna(0)
    df['lab_votes'] = df['lab_votes'].fillna(0)
    df['oth_votes'] = df['oth_votes'].fillna(0)
    votesum = df[all_votes].sum(1)

    df['con_share'] = df['con_votes']/votesum
    df['lib_share'] = df['lib_votes']/votesum
    df['lab_share'] = df['lab_votes']/votesum
    df['oth_share'] = df['oth_votes']/votesum
    all_vote_shares = ['con_share', 'lib_share', 'lab_share', 'oth_share']

    #slice by election, then for each pair of elections take only the shared counties and calculate the relative weights
    #based on the total votes of the later election

    elections = np.unique(df['election'])
    data_by_elections = [df.loc[df['election']==el] for el in elections]

    inputs_dtype = np.dtype([('numg', 'O'), ('wg', 'O'), ('pigy', 'O'), ('pg', 'O')])
    uk_inputs = np.empty(len(elections)-1, dtype=inputs_dtype)
    for i in range(len(uk_inputs)):
        data1 = data_by_elections[i]
        data2 = data_by_elections[i+1]
        #find shared counties
        joined_data = data1.merge(data2, left_on='locid', right_on='locid', suffixes=('_before', '_after'))
        print('i='+str(i)+'. From ', elections[i], ' to ', elections[i+1], ' shared counties: ', joined_data.shape[0], ' out of (',
              data1.shape[0], ', ', data2.shape[0], ')')
        # calculate numg
        uk_inputs[i]['numg'] = joined_data.shape[0]
        # calculate wg
        uk_inputs[i]['wg'] = (joined_data['total_votes_after'] / sum(joined_data['total_votes_after'])).to_numpy()
        # calculate pigy
        uk_inputs[i]['pigy'] = joined_data[[s + '_after' for s in all_vote_shares]].to_numpy()
        uk_inputs[i]['pigy'] = uk_inputs[i]['pigy']/sum(uk_inputs[i]['pigy'])
        # calculate hatpg
        uk_inputs[i]['pg'] = (joined_data[[s + '_before' for s in all_vote_shares]]).to_numpy()
        uk_inputs[i]['pg'] = uk_inputs[i]['pg'] / sum(uk_inputs[i]['pg'])

    sio.savemat('uk_inputs_bycounty.mat', {'output': uk_inputs})

    #now do another one by country/region
    inputs_dtype = np.dtype([('numg', 'O'), ('wg', 'O'), ('pigy', 'O'), ('pg', 'O')])
    uk_inputs = np.empty(len(elections) - 1, dtype=inputs_dtype)
    for i in range(len(uk_inputs)):
        data1 = data_by_elections[i]
        data2 = data_by_elections[i+1]
        joined_data_by_county = data1.merge(data2, left_on='locid', right_on='locid', suffixes=('_before', '_after'))
        #region_names = joined_data[CR].unique()
        joined_data = joined_data_by_county.groupby(CR+'_after').sum()
        print('i=' + str(i) + '. From ', elections[i], ' to ', elections[i + 1], ' shared regions: ',
              joined_data.shape[0])
        # calculate numg
        uk_inputs[i]['numg'] = joined_data.shape[0]
        # calculate wg
        uk_inputs[i]['wg'] = (joined_data['total_votes_after'] / sum(joined_data['total_votes_after'])).to_numpy()
        # calculate pigy
        uk_inputs[i]['pigy'] = joined_data[[s + '_after' for s in all_vote_shares]].to_numpy()
        uk_inputs[i]['pigy'] = uk_inputs[i]['pigy'] / np.sum(uk_inputs[i]['pigy'], axis=1)[:,np.newaxis]
        # calculate hatpg
        uk_inputs[i]['pg'] = joined_data[[s + '_before' for s in all_vote_shares]].to_numpy()
        uk_inputs[i]['pg'] = uk_inputs[i]['pg'] / np.sum(uk_inputs[i]['pg'], axis=1)[:,np.newaxis]

    sio.savemat('uk_inputs.mat', {'output': uk_inputs})


