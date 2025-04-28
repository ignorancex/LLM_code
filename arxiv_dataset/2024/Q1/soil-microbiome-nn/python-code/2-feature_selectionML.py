#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest,mutual_info_regression,RFE, f_classif,mutual_info_classif
from sklearn.linear_model import  LassoCV,LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,RepeatedKFold
import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[2]:


cv=RepeatedKFold(n_splits=10,n_repeats=3, random_state=100)
q=0.7


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:

path_response='response/response_original/'
path_x = 'OTU/OTUData-1-1/'



# In[4]:


def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features


# In[5]:


def process_data(data,data_train,data_val,cv,q):  
    x_column_list = data_train.drop(columns=['y_c']).columns 
    feature_list1 = data[x_column_list].max().sort_values(ascending=False)[data[x_column_list].max()>np.quantile(data[x_column_list].max(),q)].index
    #apply SelectKBest class to extract top 20 best features
    bestfeatures = SelectKBest(score_func=mutual_info_regression, k=round(data.shape[1]*(1-q)))
    fit = bestfeatures.fit(data[x_column_list],data['y_c'])
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_column_list)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Variable','Score']  #naming the dataframe columns
    feature_list2 = featureScores.nlargest(round(data.shape[1]*(1-q)),'Score')
    
    
    feature_list3=stepwise_selection(data[x_column_list],data['y_c'])
    reg = LassoCV(cv=cv).fit(data_train[x_column_list], data_train['y_c'])
    lassoCoefs0 = pd.DataFrame(
    data=reg.coef_[np.where(reg.coef_ != 0)[0]], 
    index=data[x_column_list].columns[np.where(reg.coef_ != 0)[0]],columns=['LASSO Coefs1'])
    feature_list4=lassoCoefs0
    
    parameters = {'n_estimators':(100, 500),
                  'min_samples_split':(3,4,5),
                  'min_samples_leaf':(3,4,5)}

    gb_model = GradientBoostingRegressor(random_state=7, warm_start=False)
    grid_obj = GridSearchCV(gb_model, param_grid=parameters, verbose=1, n_jobs=4, cv=cv)
    grid_obj = grid_obj.fit(data_train[x_column_list],data_train['y_c'])
    gb_model_best = grid_obj.best_estimator_
    y_hat = gb_model_best.predict(data_val[x_column_list])
    FeatImportance = gb_model_best.feature_importances_
    GBCoefs = pd.DataFrame(index=data_train[x_column_list].columns, data=FeatImportance,columns=['Coefs'])
    imp_coef = GBCoefs.sort_values(by='Coefs')
    feature_list6=imp_coef.loc[imp_coef['Coefs']>np.quantile(imp_coef,q)]
    feature_list6 = [item for item in feature_list6.index]
    parameters = {'n_estimators':(10,20,100, 500),
              'min_samples_split':(2,3,4),
              'min_samples_leaf':(1,2,3)}
    rf_model = RandomForestRegressor(warm_start=False)
    grid_obj = GridSearchCV(rf_model, param_grid=parameters, verbose=1, n_jobs=4, cv=cv)
    grid_obj = grid_obj.fit(data_train[x_column_list],data_train['y_c'])
    rf_model_best = grid_obj.best_estimator_
    y_hat_rf = rf_model_best.predict(data_val[x_column_list])  
    rf_model_best.score(data_val[x_column_list],data_val['y_c'])
    FeatImportance = rf_model_best.feature_importances_
    RFCoefs = pd.DataFrame(index=data_train[x_column_list].columns, data=FeatImportance,columns=['Coefs'])
    imp_coef_RF = RFCoefs.sort_values(by='Coefs')
    feature_list7=imp_coef_RF.loc[imp_coef_RF['Coefs']>np.quantile(imp_coef_RF,q)]
    feature_list7 = [item for item in feature_list7.index]
    
    mi = mutual_info_regression(data_train[x_column_list],data_train['y_c'], discrete_features=False, n_neighbors=3, copy=True, random_state=None)
    MI=pd.DataFrame(mi)
    MI['OTU']=data[x_column_list].columns
    MI=MI.set_index('OTU')
    MI=MI.sort_values(0,ascending=False)
    feature_list8=MI[MI[0]>np.quantile(mi,q)].index
    
    methodList = ['Maximum', 
              'KBest', 
              'Stepwise Regression', 
              'Lasso CV', 
              'GBM', 
              'Random Forest', 
              'Mutual Info']
    featureList = [[item for item in feature_list1], 
               [item for item in feature_list2['Variable'].values], 
               feature_list3, 
               [item for item in feature_list4.index], 
               feature_list6, 
               feature_list7,
               [item for item in feature_list8]]
    featureUniqueList = np.unique([item for sublist in featureList for item in sublist])
    featureDictionary = dict.fromkeys(featureUniqueList)
    for key in featureDictionary.keys():
        featureDictionary[key] = []
    

    for feature in featureUniqueList:
    
        for i, method in enumerate(methodList):
        
            if feature in featureList[i]:
                featureDictionary[feature].append(method)   

    featureDf = pd.DataFrame(index=featureUniqueList, columns=methodList)
    featureDf['Count'] = 0

    for feature in featureDictionary.keys(): 
        for method in methodList:
        
            if method in featureDictionary[feature]:
            
                featureDf['Count'][feature] += 1
            
                featureDf[method][feature] = 'X'
            else:
                featureDf[method][feature] = '-' 
    featureDf.sort_values(by='Count', ascending=False, inplace=True)
    return featureDf


# In[6]:


def process_data_binary(data,data_train,data_val,cv,q):  
    x_column_list = data.drop(columns=['y_b']).columns 
    feature_list1 = data[x_column_list].max().sort_values(ascending=False)[data[x_column_list].max()>np.quantile(data[x_column_list].max(),q)].index
    #apply SelectKBest class to extract top  best features
    bestfeatures = SelectKBest(score_func=f_classif, k=round(data.shape[1]*(1-q)))
    fit = bestfeatures.fit(data[x_column_list],data['y_b'])
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_column_list)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Variable','Score']  #naming the dataframe columns
    feature_list2 = featureScores.nlargest(round(data.shape[1]*(1-q)),'Score')
    
    #Recursive Feature Elimination LogisticRegression
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, n_features_to_select = round(data.shape[1]*(1-q)))
    fit = rfe.fit(data[x_column_list], data['y_b'])
    feature_list3 = data[x_column_list].columns[fit.ranking_==1]

    #Recursive Feature DecisionTree
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select = round(data.shape[1]*(1-q)))
    fit = rfe.fit(data[x_column_list], data['y_b'])
    feature_list4 = data[x_column_list].columns[fit.ranking_==1]
    
    #GradientBoostingClassifier
    parameters = {'n_estimators':(100, 500),
              'min_samples_split':(3,4),
              'min_samples_leaf':(3,4,5)}

    gb_model = GradientBoostingClassifier(random_state=7, warm_start=False)
    grid_obj = GridSearchCV(gb_model, param_grid=parameters, verbose=1, n_jobs=4, cv=cv)
    grid_obj = grid_obj.fit(data_train[x_column_list],data_train['y_b'])
    gb_model_best = grid_obj.best_estimator_
    FeatImportance = gb_model_best.feature_importances_
    GBCoefs = pd.DataFrame(index=data_train[x_column_list].columns, data=FeatImportance,columns=['Coefs'])
    imp_coef = GBCoefs.sort_values(by='Coefs')
    feature_list6=imp_coef.loc[imp_coef['Coefs']>np.quantile(imp_coef,q)]
    feature_list6 = [item for item in feature_list6.index]

    # RandomForestClassifier
    parameters = {'n_estimators':(10,20,100, 500),
          'min_samples_split':(2,3,4),
          'min_samples_leaf':(1,2,3)}
    rf_model = RandomForestClassifier(warm_start=False)
    grid_obj = GridSearchCV(rf_model, param_grid=parameters, verbose=1, n_jobs=-1, cv=cv)
    grid_obj = grid_obj.fit(data_train[x_column_list],data_train['y_b'])
    rf_model_best = grid_obj.best_estimator_
    y_hat_rf = rf_model_best.predict(data_val[x_column_list])  
    FeatImportance = rf_model_best.feature_importances_
    RFCoefs = pd.DataFrame(index=data_train[x_column_list].columns, data=FeatImportance,columns=['Coefs'])
    imp_coef_RF = RFCoefs.sort_values(by='Coefs')
    feature_list7=imp_coef_RF.loc[imp_coef_RF['Coefs']>np.quantile(imp_coef_RF,q)]
    feature_list7 = [item for item in feature_list7.index]
    #mutual_info_classif
    mi = mutual_info_classif(data_train[x_column_list],data_train['y_b'], discrete_features=False, n_neighbors=3, copy=True, random_state=None)
    MI=pd.DataFrame(mi)
    MI['OTU']=data[x_column_list].columns
    MI=MI.set_index('OTU')
    MI=MI.sort_values(0,ascending=False)
    feature_list8=MI[MI[0]>np.quantile(mi,q)].index
    
    methodList = ['Maximum', 
              'KBest', 
              'RFE_logistic', 
              'RFE_RF', 
              'GBM', 
              'Random Forest', 
              'Mutual Info']
    featureList = [[item for item in feature_list1], 
               [item for item in feature_list2['Variable'].values], 
               feature_list3, 
               [item for item in feature_list4], 
               feature_list6, 
               feature_list7,
               [item for item in feature_list8]]
    featureUniqueList = np.unique([item for sublist in featureList for item in sublist])
    featureDictionary = dict.fromkeys(featureUniqueList)
    for key in featureDictionary.keys():
        featureDictionary[key] = []
    

    for feature in featureUniqueList:
    
        for i, method in enumerate(methodList):
        
            if feature in featureList[i]:
                featureDictionary[feature].append(method)   

    featureDf = pd.DataFrame(index=featureUniqueList, columns=methodList)
    featureDf['Count'] = 0

    for feature in featureDictionary.keys(): 
        for method in methodList:
        
            if method in featureDictionary[feature]:
            
                featureDf['Count'][feature] += 1
            
                featureDf[method][feature] = 'X'
            else:
                featureDf[method][feature] = '-' 
    featureDf.sort_values(by='Count', ascending=False, inplace=True)
    return featureDf


# In[7]:


from datetime import datetime
start_time = datetime.now()


# In[11]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        if (file_response == 'Yield_Meter') or  (file_response == 'Yield_Plant'):
            print(file_response)  
            path_r= path_response+file_response
            for re in os.listdir(path_r):  
                if re[0:8] == 'response':  
                    response = pd.read_csv(path_response+file_response+'/'+re)  
                    response.rename(columns={'Column1':'Link_ID'}, inplace=True)  
                    response.rename(columns={response.columns[1]:'y_c'}, inplace=True)  

                    writer= pd.ExcelWriter(path_r+'/'+'feature_selection'+'.xlsx', engine='xlsxwriter')   
                    for file_folder in os.listdir(path_x):  
                        if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                            path = path_x+file_folder
                            file_list = []  

                            for file in os.listdir(path):  
                                if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store'):  
                                    print(file)  
                                    file_list.append(file)  
                                    data_temp = pd.read_csv(path+'/'+file)
                                    data_temp.rename(columns={'Unnamed: 0':'Link_ID'}, inplace=True)  
                                    data=pd.merge(response,data_temp,on='Link_ID')  
                                    data.drop(columns = 'Link_ID',inplace=True)  
                                    data_train,data_val = train_test_split(data,train_size=0.8, random_state=42) 
                                    output = process_data(data,data_train,data_val,cv,q) 
                            output.to_excel(writer, sheet_name=file_folder, index=True)    
                    writer.save()                     


# In[10]:


for file_response in os.listdir(path_response):  
  if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
      if (file_response != 'Yield_Meter') &  (file_response != 'Yield_Plant'):
          print(file_response)  
          path_r= path_response+file_response
          for re in os.listdir(path_r):  
              if re[0:8] == 'response':  
                  response = pd.read_csv(path_r+'/'+re)
                  response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)    
                  writer= pd.ExcelWriter(path_r+'/'+'feature_selection'+'.xlsx', engine='xlsxwriter')   
                  for file_folder in os.listdir(path_x):  
                      if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                          path = path_x+file_folder
                          file_list = []  
                          tRF=pd.DataFrame()  
                          tcluster=pd.DataFrame()  
                          k=0  
                          for file in os.listdir(path):  
                              if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store'):  
                                  print(file)  
                                  file_list.append(file)  
                                  data_temp = pd.read_csv(path+'/'+file)
                                  data_temp.rename(columns={'Unnamed: 0':'Link_ID'}, inplace=True)  
                                  data=pd.merge(response,data_temp,on='Link_ID')  
                                  data.drop(columns = 'Link_ID',inplace=True)  
                                  data_train,data_val = train_test_split(data,train_size=0.8, random_state=42) 
                                  output = process_data_binary(data,data_train,data_val,cv,q) 
                          output.to_excel(writer, sheet_name=file_folder, index=True)    
                  writer.save()                     


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

