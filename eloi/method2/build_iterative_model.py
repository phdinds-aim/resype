import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(1, '../utils')
from utils import preprocess_utility_matrix as um


def initialize_models_itemwise(model, U, suffix='model'):
    """Initialize classifier/regressor per item to be predicted"""
    models = {f'{item}{suffix}':model for item in U.columns}
    return models

def initialize_models_userwise(model, U, suffix='_model'):
    """Initialize classifier/regressor per user to be predicted"""
    models = {f'{user}{suffix}':model for user in U.index}
    return models

def eval_convergence_criterion(
    pred_curr, pred_prev, stopping_criterion='mse',
    mse_threshold=0.1, stdev_threshold=None,
    scaled=False, scaling_method='max',
    rating_min=None, rating_max=None):

    if stopping_criterion == 'mse':
        if mse_threshold is None:
            print('Threshold for calculating MSE is not defined. '
                  'Input threshold value.')
        metric = mean_squared_error(pred_curr, pred_prev)

        if scaled:
            if scaling_method == 'max':
                if rating_max is None:
                    print('Scaled metric needs maximum possible value '
                          'of rating.')
                else:
                    scaling_factor = rating_max
            elif scaling_metho == 'minmax':
                if (rating_max is None) or (rating_min is None):
                    print(
                        'Scaled metric needs maximum and minimum '
                        'possible values of rating.')
                else:
                    scaling_factor = (rating_max - rating_min)
            metric /= scaling_factor

        stop_train = (metric <= mse_threshold)

    elif stopping_criterion == 'stdev_abs':
        if stdev_threshold is None:
            print('Threshold for calculating standard deviation of absolute '
                  'error is not defined. Input threshold value.')

        metric = np.std(np.abs(pred_curr-pred_prev))

        if scaled:
            if scaling_method == 'max':
                if rating_max is None:
                    print('Scaled metric needs maximum possible value '
                          'of rating.')
                else:
                    scaling_factor = rating_max
            elif scaling_metho == 'minmax':
                if (rating_max is None) or (rating_min is None):
                    print(
                        'Scaled metric needs maximum and minimum possible'
                        ' values of rating.')
                else:
                    scaling_factor = (rating_max - rating_min)
            metric /= scaling_factor

        stop_train = (metric <= stdev_threshold)
    
    else:
        if mse_threshold is None:
            print('Stopping criterion set to MSE. Input threshold value.')
        metric = mean_squared_error(pred_curr, pred_prev)

        stop_train = (metric <= mse_threshold)    

    return metric, stop_train

def train_model_itemwise(U_df, model_object, return_models=True, max_iter=100, 
                         stopping_criterion='mse', mse_threshold=0.1, stdev_threshold=None,
                         scaled=False, scaling_method='max', rating_min=None, rating_max=None):
    U = U_df.copy()
    U_update = U.copy()
    
    models_item = initialize_models_itemwise(model_object, U, suffix='')

    known_index, missing_index = um.known_missing_split_U(
        U, split_axis=1, missing_val_filled=True)

    len_missing_vals = len(sum([i.tolist()
                                for i in missing_index.values()], []))

    preds_per_iter = [np.zeros(len_missing_vals)]
    metric_iter = []

    for i in range(max_iter):
        #     print(i)
        preds = []
    #     print(np.hstack(preds))
        for item in U.columns:
            models_item[str(item)].fit(U_update.drop(item, axis=1).loc[known_index[item]],
                                       U_update.loc[known_index[item], item])
            if len(missing_index[item]) > 0:
                pred = models_item[str(item)].predict(
                    U_update.drop(item, axis=1).loc[missing_index[item]])
            else:
                pred = np.array([])
            preds.append(pred)
            U_update.loc[missing_index[item], item] = pred

        metric, stopping_criterion = eval_convergence_criterion(
            np.hstack(preds), preds_per_iter[-1],
            stopping_criterion=stopping_criterion, mse_threshold=mse_threshold, 
            stdev_threshold=stdev_threshold, scaled=scaled,
            scaling_method=scaling_method, rating_min=rating_min, rating_max=rating_min)
        metric_iter.append(metric)
        if stopping_criterion:
            break
        preds_per_iter.append(np.hstack(preds))
    
    if return_models: 
        return U_update, metric_iter, models_item
    else: 
        return U_update, metric_iter




