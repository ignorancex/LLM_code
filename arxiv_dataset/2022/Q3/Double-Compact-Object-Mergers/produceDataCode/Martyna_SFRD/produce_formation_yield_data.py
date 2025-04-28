def createEmptyCSVplaceholder(metallicityGrid, nBPSmodels=17):


   

    BPSnameslist = list(string.ascii_uppercase)[0:nBPSmodels]
    
    for ind_bps, bps_name in enumerate(BPSnameslist):
    #   channel_names = ['total', 'I_classic', 'II_only_stable_MT', 'III_single_core_CE', 'IV_double_core_CE', 'V_other']
        column_names = [ 'Zi', 'X(Zi)_DNS', 'X(Zi)_BHBH', 'X(Zi)_BHNS']





        datas=[]


        # first column is the metallicities that we used
        datas.append(metallicityGrid)
        
        for ind_DCO, DCOname  in enumerate(['NSNS', 'BHBH', 'BHNS']):
         
            df2 = pd.read_csv(pathCSVfile + 'formationRatesTotalAndPerChannel_'+DCOname+ '_' +  '.csv', index_col=0)
            c_ = 'total'
            key_ =  bps_name + ' ' + c_ + '  [Msun^{-1}]'
            #  total yield for this DCO: 
            upperY = np.asarray(df2[key_])        
        
            datas.append(upperY)


        df = pd.DataFrame(data=datas, index=column_names).T#, columns=column_names)
#         df.columns =   df.columns.map(str)
        #   df.index.names = ['Z_i']
        #   df.columns.names = ['model']

        df.to_csv('/Users/floorbroekgaarden/Projects/GitHub/Double-Compact-Object-Mergers/dataFiles/Martyna_SFRD/1_formation_efficiency/formation_efficiency_' +bps_name+ '' +  '.csv')

    return 




Zlist = [0.0001, 0.00011, 0.00012, 0.00014, 0.00016, 0.00017,\
       0.00019, 0.00022, 0.00024, 0.00027, 0.0003, 0.00034, \
       0.00037, 0.00042, 0.00047, 0.00052, 0.00058, 0.00065,\
       0.00073, 0.00081, 0.0009, 0.00101, 0.00113, 0.00126,\
       0.0014, 0.00157, 0.00175, 0.00195, 0.00218, 0.00243, \
       0.00272, 0.00303, 0.00339, 0.00378, 0.00422, 0.00471, \
       0.00526, 0.00587, 0.00655, 0.00732, 0.00817, 0.00912, \
       0.01018, 0.01137, 0.01269, 0.01416, 0.01581, 0.01765, 0.01971, 0.022, 0.0244, 0.02705, 0.03]

createEmptyCSVplaceholder(metallicityGrid=Zlist, nBPSmodels=17)


