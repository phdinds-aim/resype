from utils import preprocess_utility_matrix as um
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(1, '../utils')


def initialize_models_itemwise(model, U, suffix='model'):
    """Initializes classifier/regressor per item to be predicted

    Parameters:
        model : model object to use to fit the data
        U (DataFrame) : utilily matrix (rows are users, columns are items) 
        suffix (str) : suffix for keys in output dictionary

    Returns:
        models (dict): dictionary of models, keys correspond to columns/items 
        in the utility matrix and values are the model objects
    """
    models = {f'{item}{suffix}': model for item in U.columns}
    return models


def initialize_models_userwise(model, U, suffix='_model'):
    """Initializes classifier/regressor per user to be predicted

    Parameters:
        model : model object to use to fit the data
        U (DataFrame) : utilily matrix (rows are users, columns are items) 
        suffix (str) : suffix for keys in output dictionary

    Returns:
        models (dict): dictionary of models, keys correspond to the rows/users 
            in the utility matrix and values are the model objects
    """
    models = {f'{user}{suffix}': model for user in U.index}
    return models


def eval_convergence_criterion(
        pred_curr, pred_prev, stopping_criterion='mse',
        mse_threshold=0.1, stdev_threshold=None,
        scaled=False, scaling_method='max',
        rating_min=None, rating_max=None):
    """
    Evaluates whether the model training has converged

    Parameters:
        pred_curr (array) : array of predicted ratings from current iteration
        pred_prev (array) : array of predicted ratings from previous iteration
        stopping_criterion (str) : metric for evaluating convergence, 
            {mse/'mean squared error', stdev_abs/'standard deviation of 
            absolute difference'}, default 'mse'
        mse_threshold (float) : threshold for stopping criterion when 
            'mse'is selected, default 0.1            
        stdev_threshold (float) : threshold for stopping criterion when 
            'stdev_abs'is selected, default None
        scaled (bool) : Indicates whether metric for stopping criterion is 
            to be scaled/normalized
        scaling_method (str) : indicates method for scaling when scaled==True, 
            {max/'maximum rating', minmax/'maximum rating - minimum rating'},
            default 'max'
        rating_min (numeric) : minimum value of rating, default None
        rating_max (numeric) : maximum value of rating, default None

    Returns:
        metric (float) : value of metric
        stop_train (bool) : Indicates convergence (stop training when True)

    """

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


def train_model_itemwise(
        U_df, model_object, return_models=True, max_iter=100,
        stopping_criterion='mse', mse_threshold=0.1, stdev_threshold=None,
        scaled=False, scaling_method='max', rating_min=None, rating_max=None):
    """
    Trains model iteratively for the item-wise recommender system: 
    (1) Estimates the missing entries of each column/item by setting it as 
    the target variable and the remaining columns as the feature variables. 
    (2) For the remaining columns, the current set of filled in values are 
    used to create a complete matrix of feature variables. 
    (3) The observed ratings in the target column are used for training. 
    (4) The missing entries are updated based on the prediction of the model 
    on each target column. 

    Parameters:
        U_df (DataFrame) : utilily matrix (rows are users, columns are items) 
        model_object : model object to use to fit the data
        return_models (bool) : Indicates whether trained models are returned 
            as output, default True
        max_iter (int) : maximum number of iterations for model training and 
            updatingof missing values, default 100
        stopping_criterion (str) : metric for evaluating convergence, 
            {mse/'mean squared error', stdev_abs/'standard deviation of 
            absolute difference'}, default 'mse'
        mse_threshold (float) : threshold for stopping criterion when 
            'mse'is selected, default 0.1            
        stdev_threshold (float) : threshold for stopping criterion when 
            'stdev_abs'is selected, default None
        scaled (bool) : Indicates whether metric for stopping criterion is 
            to be scaled/normalized
        scaling_method (str) : indicates method for scaling when scaled==True, 
            {max/'maximum rating', minmax/'maximum rating - minimum rating'},
            default 'max'
        rating_min (numeric) : minimum value of rating, default None
        rating_max (numeric) : maximum value of rating, default None

    Returns:
        U_update (DataFrame) : complete utility matrix
        metric_iter (array-like) : value of convergence metric per iteration
        models_item (dict) : dictionary of trained models, returned only if
            return_models=True
    """
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
        preds = []
        for item in U.columns:
            models_item[str(item)].fit(
                U_update.drop(item, axis=1).loc[known_index[item]],
                U_update.loc[known_index[item], item])
            if len(missing_index[item]) > 0:
                pred = models_item[str(item)].predict(
                    U_update.drop(item, axis=1).loc[missing_index[item]])
            else:
                pred = np.array([])
            preds.append(pred)
            U_update.loc[missing_index[item], item] = pred

        metric, stopping_criterion = eval_convergence_criterion(
            np.hstack(preds),
            preds_per_iter[-1],
            stopping_criterion=stopping_criterion,
            mse_threshold=mse_threshold,
            stdev_threshold=stdev_threshold,
            scaled=scaled,
            scaling_method=scaling_method,
            rating_min=rating_min,
            rating_max=rating_min)
        metric_iter.append(metric)
        if stopping_criterion:
            break
        preds_per_iter.append(np.hstack(preds))

    if return_models:
        return U_update, metric_iter, models_item
    else:
        return U_update, metric_iter