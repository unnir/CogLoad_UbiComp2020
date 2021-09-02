import numpy as np 
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


def get_scores(model, X, y, lg=False):
    X = np.array(X)
    y = np.array(y)
    
    cv = StratifiedKFold(5, random_state=45, shuffle=True)
    spl = cv.split(X, y)
    
    results_acc = []
    results_auc = []
    for train_index, test_index in spl:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        
        if not lg: 
            y_hat = model.predict_proba(X_test)[:,1]
        else:
            y_hat = model.predict_proba(X_test)
            
        results_auc.append(roc_auc_score(y_test, y_hat))
        results_acc.append(accuracy_score(y_test, np.round(y_hat)))
    #########################################################
    cv = StratifiedKFold(4, random_state=1990, shuffle=True)
    spl = cv.split(X, y)
    for train_index, test_index in spl:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        if not lg: 
            y_hat = model.predict_proba(X_test)[:,1]
        else:
            y_hat = model.predict_proba(X_test)

        results_auc.append(roc_auc_score(y_test, y_hat))
        results_acc.append(accuracy_score(y_test, np.round(y_hat)))
    print('ACC:', np.mean(results_acc), np.std(results_acc))
    print('AUC:', np.mean(results_auc), np.std(results_auc))
    print('\n')
    return np.mean(results_acc), np.std(results_acc), np.mean(results_auc), np.std(results_auc)


def tr_(x):
    x[x>0.5] = 1
    x[x<0.51] = 0
    return x