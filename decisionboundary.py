# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:39:39 2024

@author: camer
"""
import pandas as pd

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

#%% Important Features
import pickle

with open('results/feature_importance.pickle','rb') as openfile:
     dict_important = pickle.load(openfile)
     
features_Total = dict_important['Total_features'][dict_important['Total'] != 0]
features_White = dict_important['White_features'][dict_important['White'] != 0]
features_Black = dict_important['Black_features'][dict_important['Black'] != 0]
features_Asian = dict_important['Asian_features'][dict_important['Asian'] != 0]
features_Other = dict_important['Other_features'][dict_important['Other'] != 0]

features_Total = pd.DataFrame(data = dict_important['Total'][dict_important['Total'] != 0], index = features_Total, columns = ['Total'])
features_White = pd.DataFrame(data = dict_important['White'][dict_important['White'] != 0], index = features_White, columns = ['White'])
features_Black = pd.DataFrame(data = dict_important['Black'][dict_important['Black'] != 0], index = features_Black, columns = ['Black'])
features_Asian = pd.DataFrame(data = dict_important['Asian'][dict_important['Asian'] != 0], index = features_Asian, columns = ['Asian'])
features_Other = pd.DataFrame(data = dict_important['Other'][dict_important['Other'] != 0], index = features_Other, columns = ['Other'])


feature_importance = pd.concat([features_Total, features_White, features_Black, features_Asian, features_Other], axis = 1)
feature_importance.to_csv('results/feature_importance.csv')
#%% Asian Model Boundary
#Gardnerella vs Crispatus
from utils import plot_decision_boundary

#Changing Variables
ethnic_group = 'Other'
feature_test = 'No Test'
bacteria = ['Lactobacillus jensenii','Lactobacillus iners']
random_state =  1

min_x1, max_x1, min_x2, max_x2, grid_points, background_predictions = plot_decision_boundary(
    ethnic_group = ethnic_group, feature_test = feature_test, bacteria = bacteria,
    random_state = random_state)

#Plotting
import matplotlib.pyplot as plt

CMAP = "bwr"

fig,ax = plt.subplots(figsize = (7,5))
plt.title(f'{ethnic_group} Model Decision Boundary\n{bacteria[1]} vs {bacteria[0]}')

ax.scatter(grid_points[:, 0], grid_points[:, 1], 
           c=background_predictions, cmap=CMAP, alpha=0.4, s=20)
# scatter = ax.scatter(X_train.to_numpy()[:, 0], X_train.to_numpy()[:, 1], 
#                      c=y_train_pred, s=40, cmap=CMAP)
# ax.scatter(X_test.to_numpy()[:, 0], X_test.to_numpy()[:, 1], c=y_test_pred, 
#            marker="x", s=60, cmap=CMAP)


ax.set_xlabel(f'{bacteria[0]} Relative Abundance')
ax.set_ylabel(f'{bacteria[1]} Relative Abundance')
ax.set_xlim([min_x1, max_x1])
ax.set_ylim([min_x2, max_x2])
plt.tight_layout(pad=2.)


import os
save_dir = 'figures/decisions/' + ethnic_group + '/'

path_exist = os.path.exists(save_dir)
if not path_exist:

   # Create a new directory because it does not exist
   os.makedirs(save_dir)
   
   
savename = save_dir + bacteria[1] + ' vs ' + bacteria[0]
plt.savefig(savename)
plt.show()
