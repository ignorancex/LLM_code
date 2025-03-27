import pandas as pd
import numpy as np
import scipy.io as sio

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    xl = pd.read_excel('Educational_attainment.xlsx',
                       sheet_name=1, header=1)
    df = pd.DataFrame(xl)
    df = df.drop(52, 'index')  # removing Puerto Rico since the data is incomplete
    df = df.drop(0, 'index')   # removing the total united states row
    numg = df.shape[0]
    NHS = '' #no highschool
    HS = '.1' #only highschool
    SC = '.2' #some college
    CC = '.3' #completed college

    NUM_YEARS_IN_DATA = 5
    year_names = df.columns[1:(NUM_YEARS_IN_DATA+1)]
    df_by_year = [df[[YEAR+NHS, YEAR+HS, YEAR+SC, YEAR+CC]] for YEAR in year_names]


    year_arrays = [df_by_year[year].to_numpy() for year in range(NUM_YEARS_IN_DATA)]
    assert([all([np.isclose(val,1) for val in year_arrays[i].sum(axis=1)]) for i in range(NUM_YEARS_IN_DATA)])

    #read state population data (source: https://www.icip.iastate.edu/sites/default/files/uploads/tables/population/popest-annual-historical.xls)
    # for calculating the weight of each state
    xlpop = pd.read_excel('popest-annual-historical.xls',sheet_name=1, header=4)
    dfpop = pd.DataFrame(xlpop)



    # calculate numg
    inputs_dtype = np.dtype([('numg', 'O'), ('wg', 'O'), ('pigy', 'O'), ('pg', 'O')])
    data = np.empty(NUM_YEARS_IN_DATA- 1, dtype=inputs_dtype)
    for i in range(NUM_YEARS_IN_DATA-1):
        data[i]['numg'] = numg
        # calculate wg from population data at baseline year
        wg = np.ndarray((numg,))
        for state in range(numg):
            wg[state] = dfpop.loc[dfpop['Area Name']==df['Name'].iloc[state],int(year_names[i])]
        wg = wg/sum(wg)
        print(wg)
        data[i]['wg'] = wg
        # calculate pigy
        data[i]['pigy'] = year_arrays[i]
        # calculate hatpg
        data[i]['pg'] = year_arrays[i+1]

    sio.savemat('education_inputs.mat', {'output': data})


