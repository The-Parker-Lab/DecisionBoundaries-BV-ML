""" 
Script for exploration and testing of combined
srinivasan and ravel datasets

"""

import pandas as pd
import numpy as np

#%%
"""
Importing of the srinivasan and ravel datasets. Combination and
feature matching performed as well as addition of appropriate metadata
"""
#%%%
PATH_TO_DATA = 'data/srinivasan/data_processed.csv'
df = pd.read_csv(PATH_TO_DATA,index_col=0)

symptomatic_labels = pd.DataFrame(df[df.columns[-1]])
symptomatic_features = df[df.columns[:-2]]
symptomatic_ethnic = df.loc[:,'race']
#%%%
from utils import asymptomatic_data

asymptomatic_features,asymptomatic_labels = asymptomatic_data()
asymptomatic_ethnic = asymptomatic_features.loc[:,'Ethnic Groupa']
asymptomatic_features = asymptomatic_features.iloc[:,1:]
#%%%
#Find the common features between datasets
shared_features = symptomatic_features.columns.intersection(asymptomatic_features.columns)
print('number of shared features: ',len(shared_features))
#Selecting the features the two datasets have in common
symptomatic_features = symptomatic_features[shared_features]
asymptomatic_features = asymptomatic_features[shared_features]


print('Average total count of features per patient')
print('symp',np.mean(np.sum(symptomatic_features,axis=1)))
print('asymp',np.mean(np.sum(asymptomatic_features,axis=1)))

#%%%

df_asymptomatic = pd.concat([asymptomatic_features, asymptomatic_labels], axis = 1)
df_asymptomatic['Source'] = 'Ravel'
df_asymptomatic['Ethnic'] = asymptomatic_ethnic

df_symptomatic = pd.concat([symptomatic_features, symptomatic_labels], axis = 1)
df_symptomatic['Source'] = 'Srinivasan'
df_symptomatic['Ethnic'] = symptomatic_ethnic

df = pd.concat([df_asymptomatic, df_symptomatic])
df = df.reset_index(drop = True)

#Simplify Ethnic Groups
ethnic_list = ['Asian', 'White', 'Black']
for i,j in enumerate(df.Ethnic):
    if j in ethnic_list:
        pass
    else:
        df.Ethnic[i] = 'Other'
#%%

"""
Below is decomposisition of the combined dataset to observe differences
and similarity in the data
"""
#%%
import numpy as np
df_data = df.iloc[:,0:-3]
df_log = np.log(df_data + np.abs(np.min(df_data)) + 1)
df_log = (df_log - np.mean(df_log, axis = 0)) / np.std(df_log, axis = 0)

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components= 2)

principal_comps = pca.fit_transform(df_log.iloc[:,0:-2])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

principal_comps = pd.DataFrame(principal_comps)
principal_comps['Nugent score'] = df['Nugent score']
principal_comps['Ethnic'] = df['Ethnic']
principal_comps['Source'] = df['Source']

import seaborn as sns

sns.scatterplot(data = principal_comps, x = 0, y = 1, hue = 'Source')

#%%
"""
ML training for best models for each ethnic group
"""
#%%% Separation of ethnic groups
from utils import mass_train
from datetime import datetime
import warnings
import os

save_dir = 'results/'
    
path_exist = os.path.exists(save_dir)
if not path_exist:

   # Create a new directory because it does not exist
   os.makedirs(save_dir)
   
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    time_start = datetime.now()
    print(time_start)
    
    df_fit = df.copy()
    
    THRESHOLD = 7
    df_fit['Nugent score'][df_fit['Nugent score'] < THRESHOLD] = 0
    df_fit['Nugent score'][df_fit['Nugent score'] >= THRESHOLD] = 1
    
    df_White = df_fit[df_fit.Ethnic == 'White']
    df_Black = df_fit[df_fit.Ethnic == 'Black']
    df_Asian = df_fit[df_fit.Ethnic == 'Asian']
    df_Other = df_fit[df_fit.Ethnic == 'Other']
    
    X = df_fit.iloc[:,0:-3]
    X_White = df_White.iloc[:,0:-3]
    X_Black = df_Black.iloc[:,0:-3]
    X_Asian = df_Asian.iloc[:,0:-3]
    X_Other = df_Other.iloc[:,0:-3]
    
    y = df_fit.iloc[:,-3]
    y_White = df_White.iloc[:,-3]
    y_Black = df_Black.iloc[:,-3]
    y_Asian = df_Asian.iloc[:,-3]
    y_Other = df_Other.iloc[:,-3]
    
    results = pd.DataFrame()
    
    for random_state in range(10):
        
        for ethnic_group in ['Total','White','Black','Asian','Other']:
            
            for feature_test in ['No Test','Gini','Ftest','Ttest','Pbsig','Pbcorr']:
                    
                for set_clf in ['LR','RF','SVM','MLP','XGB']:
                    
                    print(f'Ethnic group: {ethnic_group}')
                    print(f'Feature test: {feature_test}')
                    print(f'Model: {set_clf}')
                    print(f'Random State: {random_state}...')
                    
                    temp_result, _ = mass_train(
                        ethnic_group = ethnic_group,
                        feature_test = feature_test, 
                        set_clf = set_clf, 
                        random_state = random_state
                        )
                    
                    results = pd.concat([results, temp_result])
                    

                    save_to = 'results/model_results.csv'
                       
                    results.to_csv(save_to)
    
    time_end = datetime.now()
    print(time_end)
    print(f'Training time is {time_end - time_start}')


