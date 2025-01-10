from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from hyperopt import hp
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import recall_score
from datetime import datetime
from hyperopt import space_eval

def optimizer(X_train, y_train, X_test, y_test, set_clf = 'LR', random_state = 0):
    print(datetime.now())  
    
    def unpack(dct):
        ret = {}
        for k,v in dct.items():
            if isinstance(v, dict):
                ret = {**ret, **unpack(v)}
            else:
                ret[k] = v
                
        return ret

    def classifier(params):
        params['random_state'] = random_state
        if set_clf == 'LR':
            return LogisticRegression(**unpack(params))
        if set_clf == 'RF':
            return RandomForestClassifier(**unpack(params))
        if set_clf == 'SVM':
            return SVC(**unpack(params))
        if set_clf == 'MLP':
            return MLPClassifier(**unpack(params))
        if set_clf == 'XGB':
            return xgb.XGBClassifier(**unpack(params))
        
    #Definition of objective function and feature space
    if set_clf == 'LR':
        def objective(params):
            clf = classifier(params)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = recall_score(y_test, y_pred)
            #print(score)
            return {
                'loss': -score,
                'status': STATUS_OK
                }
        penalty_0 = ['none', 'l2']
        penalty_1 = ['l1','l2']

        space = {
           'solver': hp.choice(
               'solver', [
                   {'solver':'lbfgs', 'penalty': hp.choice('penalty_01', penalty_0)},
                   {'solver':'newton-cg', 'penalty': hp.choice('penalty_02', penalty_0)},
                   {'solver':'newton-cholesky', 'penalty': hp.choice('penalty_03', penalty_0)},
                   {'solver':'liblinear', 'penalty': hp.choice('penalty_11', penalty_1)},
                   {'solver':'saga', 'penalty': hp.choice('penalty_21', [
                       {'penalty':'elasticnet', 'l1_ratio': hp.uniform('l1_ratio',0,1)},
                       {'penalty': 'l1'},
                       {'penalty':'l2'},
                       {'penalty':'none'}
                       
                       ])}
                   ]
               ),
           'C': hp.loguniform('C', np.log(0.001), np.log(100) )
            }
        num_eval = 4820 * 5
        
    elif set_clf == 'RF':
        def objective(params):
                
            clf = classifier(params)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = recall_score(y_test, y_pred)
            
            return {
                'loss': -score,
                'status': STATUS_OK
                }
        max_features = ['sqrt','log2']
        space = {
            'max_features': hp.choice('max_features',max_features),
            'n_estimators': hp.choice('n_estimators', np.arange(1+1,1000+1,dtype = int))
            }
        num_eval = 50 * 5
        
    elif set_clf == 'SVM':
        def objective(params):
            clf = classifier(params)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = recall_score(y_test, y_pred)
            #print(score)
            return {
                'loss': -score,
                'status': STATUS_OK
                }
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        space = {
            'kernel': hp.choice('kernel',kernel),
            'C': hp.loguniform('C', np.log(0.001), np.log(100) )
            }
        num_eval = 80 * 5
        
    elif set_clf == 'MLP':
        def objective(params):
            clf = classifier(params)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = recall_score(y_test, y_pred)
            #print(score)
            return {
                'loss': -score,
                'status': STATUS_OK
                }
        solver = ['lbfgs','sgd','adam']
        learning_rate = ['constant','invscaling','adaptive']
        hidden_layer_sizes = [[18],[109,15],[126,88,13]]
        
        space = {
            'solver': hp.choice('solver',solver),
            'learning_rate': hp.choice('learning_rate',learning_rate),
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', hidden_layer_sizes)
            }
        num_eval = 405*5
    elif set_clf == 'XGB':
        def objective(params):
            clf = classifier(params)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = recall_score(y_test, y_pred)
            #print(score)
            return {
                'loss': -score,
                'status': STATUS_OK
                }
            
        space={
            'eta': hp.uniform('eta', 0,1),
            'gamma': hp.uniform ('gamma', 0.01,100),
            'max_depth': hp.choice("max_depth", np.arange(2, 21, dtype = int)),
            'min_child_weight' : hp.choice('min_child_weight', np.arange(0, 11, dtype = int)),
            'subsample' : hp.uniform('subsample', 0.5,1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            'reg_alpha' : hp.choice('reg_alpha', np.arange(0,181,dtype = int)),
            'reg_lambda' : hp.uniform('reg_lambda', 0,1)
            }
        num_eval = 5461 * 10
    
    #Optimization Start
    
    trials = Trials()
    best_param = fmin(objective, space, algo=tpe.suggest, max_evals=num_eval, trials=trials,
                      rstate = np.random.default_rng(random_state),
                      early_stop_fn= no_progress_loss(num_eval*0.1)
                      )
    
    
   
    
    
      
    return classifier(space_eval(space, best_param))
    