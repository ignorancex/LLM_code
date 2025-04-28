def createEmptyCSVplaceholder(metallicityGrid, nBPSmodels=17):









    BPSnameslist = list(string.ascii_uppercase)[0:nBPSmodels]
    
    for ind_bps, bps_name in enumerate(BPSnameslist):
        # add path to where the COMPASOutput.h5 file is stored. 
        # For you the part '/Volumes/Andromeda/DATA/AllDCO_bugfix/fiducial/' is probably different
        path = '/Volumes/Andromeda/DATA/AllDCO_bugfix/fiducial/COMPASOutput.h5' # change this line! 
        fdata = h5.File(path)
        
        
        
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

# createEmptyCSVplaceholder(metallicityGrid=Zlist, nBPSmodels=17)
print(NOT FINISHED)




nModels=17
BPSnameslist = list(string.ascii_uppercase)[0:nModels]
modelDirList = ['fiducial', 'massTransferEfficiencyFixed_0_25', 'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75', \
               'unstableCaseBB', 'alpha0_5', 'alpha2_0', 'fiducial', 'rapid', 'maxNSmass2_0', 'maxNSmass3_0', 'noPISN',  'ccSNkick_100km_s', 'ccSNkick_30km_s', 'noBHkick', 'wolf_rayet_multiplier_0_1', 'wolf_rayet_multiplier_5']

alphabetDirDict =  {BPSnameslist[i]: modelDirList[i] for i in range(len(BPSnameslist))}


bps_name = 'A'
path = '/Volumes/Andromeda/DATA/AllDCO_bugfix/'+ alphabetDirDict[bps_name]+'/COMPASOutput.h5' # change this line! 
fdata = h5.File(path)
# to obtain properties of ALL binaries simulated, do this:

DCOtype = 'BHNS'   # You can change this line to 'BBH', 'BHNS' 'NSNS', or 'ALL' (All DCOs)  # change this line! 

print('this might take a little while, particularly if you are using the BBH')

# This code below gets the COMPAS data and only the systems that are DCOs 
Data            = COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
Data.setCOMPASDCOmask(types=DCOtype,  withinHubbleTime=True, optimistic=False)
Data.setCOMPASData()
# SeedsHubble    = Data.seeds[Data.Hubble==True]











