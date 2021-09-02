import numpy as np 
import lightgbm as lg

class EnsembleModel():
    def __init__(self, n_models=6, num_boost_round=20):
        self.lg_models = {}
        self.n_models = n_models
        self.num_boost_round = num_boost_round
    def fit(self, X, y):
        params = {}
        
        params[0] = {'num_leaves': 216,
                     'max_bin': 398,
                     'colsample_bytree': '0.730',
                     'learning_rate': '0.270',
                     'lambda_l2': 55,
                     'objective': 'binary'}
        params[1] = {'num_leaves': 176,
                     'max_bin': 413,
                     'colsample_bytree': '0.730',
                     'learning_rate': '0.280',
                     'lambda_l2': 56,
                     'objective': 'binary'}
        params[2] = {'num_leaves': 362,
                     'max_bin': 438,
                     'colsample_bytree': '0.550',
                     'learning_rate': '0.220',
                     'lambda_l2': 69,
                     'objective': 'binary'}
        params[3] = {'num_leaves': 544,
                     'max_bin': 22,
                     'colsample_bytree': '0.320',
                     'learning_rate': '0.220', 'lambda_l2': 76, 'objective': 'binary'}
        params[4] = {'num_leaves': 366,
                     'max_bin': 218,
                     'colsample_bytree': '0.400',
                     'learning_rate': '0.180', 'lambda_l2': 86, 'objective': 'binary'}
        
        params[5] = {'num_leaves': 123, 'max_bin': 442,
                     'colsample_bytree': '0.410', 'learning_rate': '0.500',
                     'lambda_l2': 87, 'objective': 'binary'}
        
        params[6] = {'num_leaves': 246, 'max_bin': 316,
                     'colsample_bytree': '0.670', 'learning_rate': '0.450',
                     'lambda_l2': 138, 'objective': 'binary'}



        #y_hat_lg = np.zeros((X_test.shape[0],self.n_models))
        train_data = lg.Dataset(X, y)
        for i in range(self.n_models):
            self.lg_models[i] = lg.train(params[i], train_data, num_boost_round=self.num_boost_round)

    def predict_proba(self, X):
        #print(X.shape)
        y_hat_xgb_test = np.zeros((X.shape[0],self.n_models+1))
        for num in range(self.n_models+1):
            if num < self.n_models:
                y_hat_xgb_test[:,num] = self.lg_models[num].predict(X).reshape(-1)

        return np.mean(y_hat_xgb_test,1)
        
        
        
        