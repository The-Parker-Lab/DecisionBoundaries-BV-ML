# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:36:22 2024

@author: camer
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_boundary(ethnic_group, feature_test, bacteria, random_state):
    X_train_selected, X_test_selected, y_train, y_test = get_train_test(ethnic_group = ethnic_group, feature_test = feature_test, random_state = random_state)      
    
    X_voronoi = pd.concat([X_train_selected, X_test_selected], axis = 0)
    
    filename = 'models/' + ethnic_group + '_model.pickle'
    with open(filename,'rb') as openfile:
         clf = pickle.load(openfile)
    
    y_voronoi = clf.predict(X_voronoi)
    
    X_voronoi = X_voronoi.loc[:, bacteria]
    
    #Voronoi

    voronoi = KNeighborsClassifier(n_neighbors=1).fit(X_voronoi, y_voronoi)
    
    #build grid
      
    min_x1, max_x1, min_x2, max_x2 = extract_plot_ranges(X_voronoi.to_numpy(), pad = 0.1)
    max_x2 = max_x1
    
    grid_points = generate_grid_points(min_x1, max_x1, min_x2, max_x2)
    print("Grid points: {}".format(grid_points.shape))
    
    #get grid predictions
    
    background_predictions = voronoi.predict(grid_points)
    print("Background predictions: {}".format(background_predictions.shape))
         
    return min_x1, max_x1, min_x2, max_x2, grid_points, background_predictions

def extract_plot_ranges(X, pad=0.5):
    """Extract plot ranges given 1D arrays of X and Y axis values."""
    min_x1, max_x1 = np.min(X[:, 0]) - pad, np.max(X[:, 0]) + pad
    min_x2, max_x2 = np.min(X[:, 1]) - pad, np.max(X[:, 1]) + pad
    return min_x1, max_x1, min_x2, max_x2


def generate_grid_points(min_x, max_x, min_y, max_y, resolution=100):
    """Generate resolution * resolution points within a given range."""
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, resolution), 
                         np.linspace(min_y, max_y, resolution))
    return np.c_[xx.ravel(), yy.ravel()]

def asymptomatic_data():
  df= pd.read_csv("data/ravel/BV Dataset copy.csv")
  df = df.drop([394,395,396], axis = 0)
  X = df.iloc[:,:-1]
  y = df["Nugent score"].copy()  # make a copy of the "Nugent score" column
  X=X.drop(labels= [ 'Community groupc ',], axis=1)
  #Normalize 16s RNA data
  X.iloc[:,1::]=X.iloc[:,1::]/100
  #Remove pH
  X = X.drop(labels= ['pH'], axis=1)
  #Rename Columns
  X.columns = ["Lactobacillus" + name[2:] if name.startswith("L.") else name for name in X.columns]
  X = X.rename(columns={'Gardnerella': 'Gardnerella vaginalis'})
  # #Binary y
  # y[y<7]=0
  # y[y>=7]=1
  return X,y.astype('int64')

def clean_meta_data(df):
    df['clue'] = df['clue'].replace(np.nan,'Unknown')
    df['whiff'] = df['whiff'].replace(np.nan,'Unknown')
    
    df['vag_fluid'] = df['vag_fluid'].replace(np.nan,'Unknown')
    
    #df = simplify_race(df)
    
    return df

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.model_selection import train_test_split
from hyperparams import optimizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

def mass_train(ethnic_group = 'Total', feature_test = 'No Test', set_clf = 'LR', random_state = 0):
    
    X_train_selected, X_test_selected, y_train, y_test = get_train_test(ethnic_group = ethnic_group, feature_test = feature_test, random_state = random_state)      
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_train_selected, y_train, test_size = 0.3)
    
    if X_train_selected.shape[1] == 0:
        data = {'BA':None, 'FPR':None, 'FNR':None,
                'Feature_Test':feature_test,
                'Ethnic_Group':ethnic_group,
                'Model':set_clf,
                'Random_State':[random_state]}
        
        return pd.DataFrame(data), None
        
    else:
        clf = optimizer(X_train_sub, y_train_sub, X_test_sub, y_test_sub, set_clf = set_clf, random_state = random_state)
        clf.fit(X_train_selected, y_train)
        
        y_pred = clf.predict(X_test_selected)
        
        ba = balanced_accuracy_score(y_test, y_pred)
        
        tpr = recall_score(y_test, y_pred)
        tnr = recall_score(y_test, y_pred, pos_label = 0)
        
        fpr = 1-tnr
        fnr = 1-tpr
        
        ap = average_precision_score(y_test, y_pred)
        
        data = {'BA':[ba], 'AP': [ap], 'FPR':[fpr], 'FNR':[fnr],
                'Feature_Test':feature_test,
                'Ethnic_Group':ethnic_group,
                'Model':set_clf,
                'Random_State':[random_state]}
        
        return pd.DataFrame(data), clf

def get_train_test(ethnic_group = 'Total', feature_test = 'No Test' , random_state = 0):
    
    from __main__ import X, y, X_White, y_White, X_Black, y_Black, X_Asian, y_Asian, X_Other, y_Other
    
    if ethnic_group == 'Total':
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state)
    if ethnic_group == 'White':
        X_train,X_test,y_train,y_test = train_test_split(X_White, y_White, test_size=0.2, shuffle=True, stratify=y_White, random_state=random_state)
    if ethnic_group == 'Black':
        X_train,X_test,y_train,y_test = train_test_split(X_Black, y_Black, test_size=0.2, shuffle=True, stratify=y_Black, random_state=random_state)
    if ethnic_group == 'Asian':
        X_train,X_test,y_train,y_test = train_test_split(X_Asian, y_Asian, test_size=0.2, shuffle=True, stratify=y_Asian, random_state=random_state)
    if ethnic_group == 'Other':
        X_train,X_test,y_train,y_test = train_test_split(X_Other, y_Other, test_size=0.2, shuffle=True, stratify=y_Other, random_state=random_state)
    
    
    if feature_test == 'No Test':
        pass
    elif feature_test == 'Gini':
        feature_list, df_features = GiniTest(X_train,y_train)
    elif feature_test == 'Ftest':
        feature_list, df_features = Ftest(X_train,y_train)
    elif feature_test == 'Ttest':
        feature_list, df_features = Ttest(X_train,y_train)
    elif feature_test == 'Pbsig':
        feature_list, correlationlist, df_features,dfimpfeatcorr = PBtest(X_train,y_train)
    elif feature_test == 'Pbcorr':
        significantlist, feature_list, dfimpfeatsig,df_features = PBtest(X_train,y_train)
    
    
    if feature_test == 'No Test':
        return X_train, X_test, y_train, y_test
        
    else:
        return X_train[feature_list], X_test[feature_list], y_train, y_test
        
#Feature Tests
def GiniTest(xtrain,ytrain):
    
    #calculates gini gain (higher gain more important feature)
    clf = DecisionTreeClassifier(criterion='gini')

    px = xtrain
    #getting y values for amsel
    py = ytrain
    
    # Fit the decision tree classifier
    clf = clf.fit(px, py)

    # Feature importances based on reduction in Gini impurity each feature gives when splitting nodes!!
    feature_importances = clf.feature_importances_
    #print(feature_importances)
 
    # Sort the feature importances from greatest to least using the sorted indices
    sorted_indices = feature_importances.argsort()[::-1]

    #array of names sorted accoridng to index of feature importance
    sorted_feature_names = px.columns[sorted_indices]
    
    sorted_importances = feature_importances[sorted_indices]

    new_si = np.delete(sorted_importances, np.where(sorted_importances == 0))
    new_si_length = len(new_si)
    
    sfn_list = sorted_feature_names.tolist()
    sfn_list = sfn_list[0:new_si_length]

    #get list of only important features from gini
    giniimplist = []
    dfimpfeat = pd.DataFrame(columns = ['Feature Name', 'Gini score'])
    
    for i in range(len(sorted_importances)): 
        if sorted_importances[i] > 0:
            giniimplist.append(sorted_feature_names[i])
            new_row = pd.DataFrame({'Feature Name':[sorted_feature_names[i]],'Gini score':[sorted_importances[i]]})
            dfimpfeat = pd.concat([dfimpfeat, new_row], ignore_index=True)


   # for i in range(len(sorted_feature_names)):
        #print(sorted_feature_names[i].ljust(30)+":"+str(sorted_importances[i]))
    return giniimplist, dfimpfeat

def Ttest(xtrain, ytrain):

    dataset = pd.concat([xtrain,ytrain],axis = 1)
    Set0 = dataset[dataset['Nugent score'] == 0]
    Set1 = dataset[dataset['Nugent score'] == 1]

    impfeat =[]
    dfimpfeat = pd.DataFrame(columns = ['Feature Name', 'P-value'])
    for column in Set0:
        Set0data = Set0[column]
        Set1data = Set1[column]
        tstat, pval = stats.ttest_ind(a = Set0data, b = Set1data, alternative="two-sided")
    
        alpha = 0.05
        if pval < alpha:
            if column != 'Nugent score':
                impfeat.append(column)
            new_row = pd.DataFrame({'Feature Name':[column],'P-value':[pval]})
            dfimpfeat = pd.concat([dfimpfeat, new_row], ignore_index=True)
    dfimpfeat = dfimpfeat.sort_values(by = ['P-value'], ascending = True)
    dfimpfeat = dfimpfeat.loc[dfimpfeat["Feature Name"] != 'Nugent score']
    
    return impfeat, dfimpfeat

def Ftest(xtrain,ytrain):

    px = xtrain
    #getting y values for amsel
    py = ytrain

    fvalue_Best = SelectKBest(f_classif, k=15)

    fvalue_Best.fit(px, py)
    f_values = fvalue_Best.scores_
    #p_values = fvalue_Best.pvalues_
    
    cols = fvalue_Best.get_support(indices=True)

    #f_values, p_values = f_classif(px, py)
    #cols = [i for i, p in enumerate(p_values) if p < 0.05]

    features_df_newx = px.iloc[:,cols]
    ftest_list = []
    #table of features with fvalue
    dfimpfeatfval = pd.DataFrame(columns = ['Feature Name', 'F-score'])
    for (index, colname) in enumerate(features_df_newx):
        ftest_list.append(colname)
        new_row = pd.DataFrame({'Feature Name':[colname],'F-score':[f_values[index]]})
        dfimpfeatfval = pd.concat([dfimpfeatfval, new_row], ignore_index=True)
    dfimpfeatfval = dfimpfeatfval.sort_values(by = ['F-score'], ascending = False)
    
    return ftest_list, dfimpfeatfval

def PBtest(xtrain,ytrain,pbtype = 'correlation'):
    px = xtrain
    py = ytrain

    #pointbiserial
    #used when one variable is interval and other variable has only 2 possible variables
    significantlist = []
    correlationlist = []
    dfimpfeatsig = pd.DataFrame(columns = ['Feature Name', 'P-value'])
    dfimpfeatcorr = pd.DataFrame(columns = ['Feature Name', 'Correlation Coefficient'])

    
    for (columnName, X_trainfeature) in px.items():
        #point biserial 
        correlation_value,p_value= stats.pointbiserialr(X_trainfeature, py)

        alpha = 0.2
        #restricted p value by alpha
        if p_value < alpha:
            significantlist.append(columnName)
            new_row = pd.DataFrame({'Feature Name':[columnName],'P-value':[p_value]})
            dfimpfeatsig = pd.concat([dfimpfeatsig, new_row], ignore_index=True)
            #if between .5 and 1 is strongly correlated
            #further restricted by correlation value
            if correlation_value > 0.4: 
                correlationlist.append(columnName)
                new_row = pd.DataFrame({'Feature Name':[columnName],'Correlation Coefficient':[correlation_value]})
                dfimpfeatcorr = pd.concat([dfimpfeatcorr, new_row], ignore_index=True)
    dfimpfeatsig = dfimpfeatsig.sort_values(by = ['P-value'], ascending = True)
    dfimpfeatcorr = dfimpfeatcorr.sort_values(by = ['Correlation Coefficient'], ascending = False)
    
    return significantlist, correlationlist, dfimpfeatsig,dfimpfeatcorr