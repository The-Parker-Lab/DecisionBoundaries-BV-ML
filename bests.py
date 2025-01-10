# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:34 2024

@author: celestec
"""
import pandas as pd
import numpy as np

#%% Model Results

df_results = pd.read_csv('results/model_results.csv').iloc[:,1:]

col_feature = df_results.columns[4]
col_ethnic = df_results.columns[5]
col_model = df_results.columns[6]
col_index = df_results.columns[7]

#%% Best Models w/o Feature Selection

df_means = pd.DataFrame()
feature = 'No Test'
for model in ['LR','RF','SVM','MLP', 'XGB']:
        for ethnic_group in ['Total','White','Black','Asian','Other']:
            
            mask = (df_results[col_feature] == feature) & (df_results[col_model] == model) & (df_results[col_ethnic] == ethnic_group)
            group = df_results[mask].set_index(col_index)
            
            ba = group.iloc[:,0]
            mean = ba.mean(numeric_only = True)
            
            data = {col_ethnic:ethnic_group,
                    col_feature:feature,
                    col_model:model,
                    'BA': [mean]}
            df_temp = pd.DataFrame(data)
            df_means = pd.concat([df_means, df_temp], ignore_index= True)
       
bests = pd.DataFrame()
for ethnic_group in ['Total','White','Black','Asian','Other']:
    mask = df_means[col_ethnic] == ethnic_group
    group = df_means[mask]
    
    best = group[group['BA'] == group['BA'].max()]
    bests = pd.concat([bests, best])
    
#bests.to_csv('results/best_models_notest.csv')

metric = 'BA'
best_means = bests.reset_index(drop = True)

df_best = pd.DataFrame()

for i in best_means.index:
    temp = bests.iloc[i,:]
    mask = (df_results[col_feature] == temp[col_feature]) & (df_results[col_model] == temp[col_model]) & (df_results[col_ethnic] == temp[col_ethnic])
    best_set = df_results[mask].set_index(col_index)
    best_set = pd.concat([best_set[metric], best_set['Ethnic_Group']], axis = 1)
    best_pivot = best_set.pivot(columns = 'Ethnic_Group', values = 'BA')
    df_best = pd.concat([df_best, best_pivot], axis = 1)
    
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.boxplot(x = df_best, showmeans= True)
ax.set_xticks(ticks = [1,2,3,4,5], labels = df_best.columns, fontsize = 8.5)

ax.set_ylabel('Balanced Accuracy')
ax.set_xlabel('Dataset')
ax.set_title('Performance of Best Model\nw/o Feature Selection for Each Population')

savename = 'figures/best models no test'
#plt.savefig(savename)
df_best.to_csv('results/best_model_results_notest.csv')
#%%% Best overall

df_means = pd.DataFrame()
for feature in ['No Test','Gini','Ftest','Ttest','Pbsig','Pbcorr']:
    for model in ['LR','RF','SVM','MLP', 'XGB']:
        for ethnic_group in ['Total','White','Black','Asian','Other']:
            
            mask = (df_results[col_feature] == feature) & (df_results[col_model] == model) & (df_results[col_ethnic] == ethnic_group)
            group = df_results[mask].set_index(col_index)
            
            ba = group.iloc[:,0]
            mean = ba.mean(numeric_only = True)
            
            data = {col_ethnic:ethnic_group,
                    col_feature:feature,
                    col_model:model,
                    'BA': [mean]}
            df_temp = pd.DataFrame(data)
            df_means = pd.concat([df_means, df_temp], ignore_index= True)
    
bests = pd.DataFrame()
for ethnic_group in ['Total','White','Black','Asian','Other']:
    mask = df_means[col_ethnic] == ethnic_group
    group = df_means[mask]
    
    best = group[group['BA'] == group['BA'].max()]
    bests = pd.concat([bests, best])
    
#bests.to_csv('results/best_models.csv')

metric = 'BA'
best_means = bests.reset_index(drop = True)

df_best = pd.DataFrame()

for i in best_means.index:
    temp = bests.iloc[i,:]
    mask = (df_results[col_feature] == temp[col_feature]) & (df_results[col_model] == temp[col_model]) & (df_results[col_ethnic] == temp[col_ethnic])
    best_set = df_results[mask].set_index(col_index)
    best_set = pd.concat([best_set[metric], best_set['Ethnic_Group']], axis = 1)
    best_pivot = best_set.pivot(columns = 'Ethnic_Group', values = 'BA')
    df_best = pd.concat([df_best, best_pivot], axis = 1)
    
#Dropping White F-Test
df_best = df_best.iloc[:,[0,1,3,4,5]]
    
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
ax.boxplot(x = df_best, showmeans= True)
ax.set_xticks(ticks = [1,2,3,4,5], labels = df_best.columns, fontsize = 8.5)

ax.set_ylabel('Balanced Accuracy')
ax.set_xlabel('Dataset')
ax.set_title('Performance of Best Model\nw/ Feature Selection for Each Population')

savename = 'figures/best models'
#plt.savefig(savename)

df_best.to_csv('best_models_results.csv')
#%%Differences 
import seaborn as sns
import matplotlib.pyplot as plt
import os
     
df = df_results

df['BA'] = df['BA'].fillna(0)
df['AP'] = df['AP'].fillna(0)
df['FPR'] = df['FPR'].fillna(1)
df['FNR'] = df['FNR'].fillna(1)

#Metric list ['BA','AP','FPR','FNR']
for metric in ['BA'] :

    for model in ['LR','RF','SVM','MLP', 'XGB']:

        df_model = df[df['Model'] == model]
        df_baseline = df_model[(df_model['Ethnic_Group'] == 'Total') & (df_model['Feature_Test'] == 'No Test')][metric].to_numpy()
        df_baseline = np.average(df_baseline)
        
        df_disparity = pd.DataFrame()
        for ethnic_group in ['Total','White','Black','Asian','Other']:
            
            baseline = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'No Test')][metric].to_numpy() - df_baseline
            gini = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'Gini')][metric].to_numpy() - df_baseline
            ftest = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'Ftest')][metric].to_numpy() - df_baseline
            ttest = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'Ttest')][metric].to_numpy() - df_baseline
            pbsig = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'Pbsig')][metric].to_numpy() - df_baseline
            pbcorr = df_model[(df_model['Ethnic_Group'] == ethnic_group) & (df_model['Feature_Test'] == 'Pbcorr')][metric].to_numpy() - df_baseline
            
            data = [baseline, gini, ftest, ttest, pbsig, pbcorr]
            
            df_set = pd.DataFrame(data, index = ['Baseline','Gini', 'Ftest','Ttest','Pbsig','Pbcorr']).transpose()
            df_set = pd.melt(df_set, value_vars = ['Baseline','Gini', 'Ftest','Ttest','Pbsig','Pbcorr'])
            df_set['ethnicity'] = ethnic_group
            
            df_disparity = pd.concat([df_disparity, df_set], axis = 0)
        
        df_disparity.rename(columns={'variable':'Feature Test'},inplace=True)
        
        
        fig,ax = plt.subplots(figsize = (7,5))
        plt.title(f'{model} Model Performances')
        
        sns.set(style="darkgrid",font_scale=1)
        ax = sns.barplot(df_disparity,x= 'ethnicity' ,y='value',hue='Feature Test',
                    palette="Set1", errorbar = None)
        
        ax.set(xlabel='Ethnicity', 
               ylabel= metric + ' Model - ' + metric + ' Baseline')
        ax.legend(ncols = 2)
        
        plt.tight_layout(pad=1.)
   
        # save_dir = 'figures/difference/' + metric + '/'
        
        # path_exist = os.path.exists(save_dir)
        # if not path_exist:
        
        #    # Create a new directory because it does not exist
        #    os.makedirs(save_dir)
           
        # savename = save_dir + model
        # plt.savefig(savename)
        
        save_dir = 'figures/difference_csv/' + metric + '/'
        
        path_exist = os.path.exists(save_dir)
        if not path_exist:
        
            # Create a new directory because it does not exist
            os.makedirs(save_dir)
           
        savename = save_dir + model + '.csv'
        df_disparity.to_csv(savename)
        
        plt.show()
#%%% Opening Data
PATH_TO_DATA = 'data/srinivasan/data_processed.csv'
df = pd.read_csv(PATH_TO_DATA,index_col=0)

symptomatic_labels = pd.DataFrame(df[df.columns[-1]])
symptomatic_features = df[df.columns[:-2]]
symptomatic_ethnic = df.loc[:,'race']

from utils import asymptomatic_data

asymptomatic_features,asymptomatic_labels = asymptomatic_data()
asymptomatic_ethnic = asymptomatic_features.loc[:,'Ethnic Groupa']
asymptomatic_features = asymptomatic_features.iloc[:,1:]

#Find the common features between datasets
shared_features = symptomatic_features.columns.intersection(asymptomatic_features.columns)

#Selecting the features the two datasets have in common
symptomatic_features = symptomatic_features[shared_features]
asymptomatic_features = asymptomatic_features[shared_features]

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

#%% Training Best Models and Saving
from utils import get_train_test
from hyperparams import optimizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.inspection import permutation_importance

bests = pd.read_csv('results/best_models.csv', index_col = 0)
bests = bests.iloc[[0,1,3,4,5],:]

random_state = 1
importances = {}
for ethnic_group in ['Total', 'White', 'Black', 'Asian', 'Other']:
    
    feature_test = bests[bests['Ethnic_Group'] == ethnic_group]['Feature_Test'].iloc[0]
    set_clf = bests[bests['Ethnic_Group'] == ethnic_group]['Model'].iloc[0]

    X_train_selected, X_test_selected, y_train, y_test = get_train_test(ethnic_group = ethnic_group, feature_test = feature_test, random_state = random_state)      
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_train_selected, y_train, test_size = 0.3)
    
    clf = optimizer(X_train_sub, y_train_sub, X_test_sub, y_test_sub, set_clf = set_clf, random_state = random_state)
    clf.fit(X_train_selected, y_train)
    
    filename = 'models/' + ethnic_group + '_model.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)
        
    important = permutation_importance(
        clf, X_test_selected, y_test, scoring = 'balanced_accuracy', n_repeats = 10, random_state = random_state)
    importances[ethnic_group] = important.importances_mean
    importances[ethnic_group + '_features'] = X_test_selected.columns
#%%

with open('results/feature_importance.pickle', 'wb') as file:
    pickle.dump(importances, file)
    
#%%
