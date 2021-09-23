import pandas as pd
import numpy as np
from collections import Counter
import pickle
import joblib
import sys

from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

# import warnings
# warnings.filterwarnings("ignore")

class Resype:
    """
    Resype implements a machine learning framework for recommender systems.
            Parameters:
                    transaction_list (pandas.DataFrame): Dataframe with columns user_id, item_id, rating in the form
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 1     | 4    |
                        
            Final outputs:
                    recommendations (pandas.DataFrame): Dataframe with columns user_id, item_id, score
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 3     | 2    |                    
    """
    
    def __init__(self, transaction_list):
        """
            Parameters:
                    transaction_list (pandas.DataFrame): Dataframe with columns user_id, item_id, rating in the form
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 1     | 4    |        
        """
        self.transaction_list = transaction_list
        self.users_clustered = False # whether the users were clustered
        self.items_clustered = False # whether the items were clustered
        
    def construct_utility_matrix(self):
        self.utility_matrix = self.transaction_list.pivot(index='user_id', columns='item_id', values='rating') # utility matrix 
        return self.utility_matrix
    

    def mean_center_utilmat(self, U_df, axis=1, fillna=True, fill_val=None):
        """Gets the mean-centered utility matrix

            Parameters:
                    U_df (DataFrame): utility matrix (rows are users, columns are items) 
                    axis (int): The axis along mean is evaluated, 
                        {0/'index', 1/'columns'}, default 1
                    fillna (bool): Indicates whether missing/null values are to 
                        be filled
                    fill_val (None/float) : Value to be used to fill null values 
                        when fillna==True, default None

            Returns:
                mean_centered (DataFrame): mean-centered utility matrix
        """
        mean_centered = U_df.sub(U_df.mean(axis=axis), axis=1-axis)
        if fillna:
            if fill_val is not None:
                return mean_centered.fillna(fill_val)
            else:
                return mean_centered.fillna(0)
        else:
            return mean_centered



    def split_utilmat_label_features(self, label_index, axis=1):
        """Splits utility matrix into label (column/row where ratings are predicted) 
        and features (columns/rows to be used as input in the model)

            Parameters:
                    U_df (DataFrame): utility matrix (rows are users, columns are items) 
                    label_index (int/str): column name or index corresponding to  item 
                        ratings (column) or user ratings (row) to be predicted
                    axis (int): The axis along the utility matrix is split, 
                        {0/'index', 1/'columns'}, default 1

            Returns:
                    label_df (DataFrame): contains the column/row to be predicted
                    feature_df (DataFrame): contains the features   
        """

        # VARIABLES
        U = self.utility_matrix

        if axis == 1:
            label_col = U.columns[U.columns == label_index]
            feature_col = U.columns[~(U.columns == label_index)]
            label_df = U.loc[:, label_col]
            feature_df = U.loc[:, feature_col]
        elif axis == 0:
            label_row = U.index[U.index == label_index]
            feature_row = U.index[~(U.index == label_index)]
            label_df = U.loc[label_row, :]
            feature_df = U.loc[feature_row, :]

        return label_df, feature_df


    def known_missing_split_1d(label_data, feature_data, split_axis=1,
                               missing_val_filled=False, fill_val=None):
        """Returns index of the dataset corresponding to known and missing ratings
        in the label data (row or column to be predicted)

        Parameters:
            label_df (DataFrame) : contains the column/row to be predicted
            feature_df (DataFrame) : contains the features  
            split_axis (int) : The axis along the utility matrix is split, 
                {0/'index', 1/'columns'}, default 1
            missing_val_filled (bool) : Indicates whether missing/null values 
                in the label/feature data were filled
            fill_val (None/float) : Value used to fill the null values when 
                missing_val_filled==True, default None            

        Returns:
            X_known.index : index corresponding to known ratings
            X_missing.index : index corresponding to missing/unknown ratings
        """    
        if missing_val_filled:
            if fill_val is None:
                missing_vals = (label_data == 0).values.flatten()
            else:
                missing_vals = (label_data == fill_val).values.flatten()
        else:
            missing_vals = label_data.isnull().values.flatten()
        if split_axis == 1:
            X_missing = feature_data.loc[missing_vals, :]
            X_known = feature_data.loc[~missing_vals, :]
        elif split_axis == 0:
            X_missing = feature_data.loc[:, missing_vals]
            X_known = feature_data.loc[:, ~missing_vals]
        else:
            X_missing = feature_data.loc[missing_vals, :]
            X_known = feature_data.loc[~missing_vals, :]

        return X_known.index, X_missing.index


    
    def known_missing_split_U(self, U, split_axis=1, missing_val_filled=False,
                              fill_val=None):
        """Returns index of the dataset corresponding to known and missing ratings
        in for the whole utility matrix

            Parameters:
                    U_df (DataFrame) : utility matrix (rows are users, columns are items) 
                    split_axis (int) : The axis along the utility matrix is split, 
                        {0/'index', 1/'columns'}, default 1
                    missing_val_filled (bool) : Indicates whether missing/null 
                        values in the label/feature data were filled
                    fill_val (None/float) : Value used to fill the null values when 
                        missing_val_filled==True, default None            

            Returns:
                    known_idx (dict): keys are the column name/index to be predicted, 
                        values are index of utility matrix that contains known values
                    missing_idx (dict): keys are the column name/index to be predicted, 
                        values are index of utility matrix that contains missing values
            """    

        if missing_val_filled:
            if fill_val is None:
                missing_val = 0
            else:
                missing_val = fill_val
            if split_axis == 1:
                known_idx = dict((U == missing_val).T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
                missing_idx = dict((U == missing_val).T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(x).flatten()]))
            elif split_axis == 0:
                known_idx = dict((U == missing_val).apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.T.index[np.argwhere(~x).flatten()]))
                missing_idx = dict((U == missing_val).apply(lambda x: np.array(x), axis=1).apply(
                    lambda x: U.T.index[np.argwhere(x).flatten()]))
            else:
                print('Invalid axis. Result for axis=1 is returned.')
                known_idx = dict((U == missing_val).T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
                missing_idx = dict((U == missing_val).T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(x).flatten()]))
        else:
            if split_axis == 1:
                known_idx = dict(U.isnull().T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
                missing_idx = dict(U.isnull().T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(x).flatten()]))
            elif split_axis == 0:
                train_idx = dict(U.isnull().apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.T.index[np.argwhere(~x).flatten()]))
                test_idx = dict(U.isnull().apply(lambda x: np.array(x), axis=1).apply(
                    lambda x: U.T.index[np.argwhere(x).flatten()]))
            else:
                print('Invalid axis. Result for axis=1 is returned.')
                known_idx = dict(U.isnull().T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
                missing_idx = dict(U.isnull().T.apply(lambda x: np.array(
                    x), axis=1).apply(lambda x: U.index[np.argwhere(x).flatten()]))

        return known_idx, missing_idx
    
    
    def nan_mask(p=0.2):
        """Randomly sets values of the utility matrix to NaN

        Parameters:
                U (numpy.array): utility matrix (rows are users, columns are items) 
                p (float): percentage of matrix which will be set to NaN, 
                    value ranges from 0 to 1, default 0.2

        Returns:
                U*mask (numpy.array): utility matrix masked with NaNs
        """    
        # VARS
        U = self.utility_matrix

        mask = np.ones(np.shape(U))
        random_index = np.random.choice(U.size, size=int(U.size*p), replace=False)
        np.ravel(mask)[random_index] = np.nan
        return U*mask    
    
    
    def gen_missing_ratings(U_df, p=0.2, n_masks=10):
        """Generates multiple sets of masked utility matrix 

        Parameters:
                U_df (DataFrame): utility matrix (rows are users, columns are items) 
                p (float): percentage of matrix which will be set to NaN, 
                    value ranges from 0 to 1, default 0.2
                n_masks (int): number of masks to be generated; indicates number 
                    of synthetic datasets to be generated, default 10

        Returns:
                masked_um (list): list of masked utility matrices
        """    
        cols = U_df.columns
        idx = U_df.index
        U_arr = U_df.values
        masked_um = []
        for n in range(n_masks):
            masked_um.append(pd.DataFrame(nan_mask(U_arr, p=p),
                                          columns=cols,
                                          index=idx))
        return masked_um    
    


    def initialize_models_itemwise(self, U, model, suffix='model'):
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


    def initialize_models_userwise(U, model, suffix='_model'):
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


    def eval_convergence_criterion(self, 
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


    def train_model_itemwise(self,
            U_df, model_object, return_models=True, max_iter=100,
            stopping_criterion='mse', mse_threshold=0.1, stdev_threshold=None,
            scaled=False, scaling_method='max', rating_min=None, rating_max=None):
        """Trains model iteratively for the item-wise recommender system: 
        (1) Estimates the missing entries of each column/item by setting it as 
        the target variable and the remaining columns as the feature variables. 
        (2) For the remaining columns, the current set of filled in values are 
        used to create a complete matrix of feature variables. 
        (3) The observed ratings in the target column are used for training. 
        (4) The missing entries are updated based on the prediction of the model 
        on each target column. 

            Parameters:
                    U_df (DataFrame): raw utility matrix (rows are users, 
                        columns are items) 
                    model_object : model object to use to fit the data
                    return_models (bool): Indicates whether trained models are 
                        returned as output, default True
                    max_iter (int): maximum number of iterations for model 
                        training and updating of missing values, default 100
                    stopping_criterion (str): metric for evaluating convergence, 
                        {mse/'mean squared error', stdev_abs/'standard deviation 
                        of absolute difference'}, default 'mse'
                    mse_threshold (float): threshold for stopping criterion when 
                        'mse'is selected, default 0.1            
                    stdev_threshold (float): threshold for stopping criterion 
                        when 'stdev_abs'is selected, default None
                    scaled (bool): Indicates whether metric for stopping criterion 
                        is to be scaled/normalized
                    scaling_method (str): indicates method for scaling when 
                        scaled==True, {max/'maximum rating',
                        minmax/'maximum rating - minimum rating'}, default 'max'
                    rating_min (numeric): minimum value of rating, default None
                    rating_max (numeric): maximum value of rating, default None

            Returns:
                    U_update (DataFrame): complete utility matrix
                    metric_iter (array-like): value of convergence metric per iteration
                    models_item (dict): dictionary of trained models, returned only if
                        return_models=True
        """
        # VARS
        U = U_df.copy()

        models_item = self.initialize_models_itemwise(
            model=model_object, U=U, suffix='')

        known_index, missing_index = self.known_missing_split_U(
            U=U, split_axis=1, missing_val_filled=True)

        len_missing_vals = len(sum([i.tolist()
                                    for i in missing_index.values()], []))

        U = self.mean_center_utilmat(U, axis=1, fillna=True, fill_val=0)
        U_update = U.copy()    

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

            metric, stopping_criterion = self.eval_convergence_criterion(
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
        
        
    def train_model_svd(self,
            U_df, model_object, d, return_models=True):
        """
        Trains model with dimensionality reduction (SVD): 
        (1) Estimates the missing entries of the utility matrix.
        (2) Each column/item is set as the target variable one at a time, and
        the remaining columns are set as the feature matrix.
        (3) SVD is performed on the feature matrix before model training.
        (4) Rows with missing items are in the test set, while the rest are in 
        the training set.
        (5) Process is repeated for all columns/items, yielding a completed 
        utility matrix.

        Parameters:
            U_df (DataFrame) : raw utilily matrix (rows are users, columns are items) 
            model_object : model object to use to fit the data
            d : number of desired dimensions after dimensionality reduction
            return_models (bool) : Indicates whether trained models are returned 
                as output, default True

        Returns:
            U_update (DataFrame) : complete utility matrix
            models_item (dict) : dictionary of trained models, returned only if
                return_models=True
        """
        U = U_df.copy()

        known_index, missing_index = self.known_missing_split_U(
            U, split_axis=1)

        U = self.mean_filled_utilmat(U)
        U_update = U.copy()

        models_item = self.initialize_models_itemwise(model=model_object, U=U, suffix='')

        for item in U.columns:
            U_temp = U.drop(item, axis=1)
            S = np.matmul(U_temp.T.values, U_temp.values)
            _, _, PT = svds(S, k=d)
            Pd = PT.T
            U_svd = pd.DataFrame(np.matmul(U_temp.values, Pd),
                                 index=U_temp.index)

            models_item[str(item)].fit(
                U_svd.loc[known_index[item]],
                U_update.loc[known_index[item], item])
            if len(missing_index[item]) > 0:
                pred = models_item[str(item)].predict(
                    U_svd.loc[missing_index[item]])
            else:
                pred = np.array([])
            U_update.loc[missing_index[item], item] = pred


        if return_models:
            return U_update, models_item
        else:
            return U_update        
        
        
    def mean_filled_utilmat(self, U, axis=1):
        if axis:
            return U.T.fillna(U.mean(axis=axis)).T
        else:
            return U.fillna(U.mean(axis=axis))


    def fit(self, model_object, method="iterative"):
        U_df_mc = self.mean_center_utilmat(self.utility_matrix, axis=1, fillna=False)

        if method == 'iterative':
            self.models_item = self.initialize_models_itemwise(U_df_mc, model_object, suffix='')    
            U_imputed, metrics, models = self.train_model_itemwise(U_df_mc, model_object, return_models=True) 
            self.utility_matrix_preds = U_imputed.add(U_df_mc.mean(axis=1), axis=0)

        if method == 'svd':
            self.models_item = self.initialize_models_itemwise(U_df_mc, model_object, suffix='')
            U_imputed, models = self.train_model_svd(U_df_mc, model_object, d=2, return_models=True)        
            self.utility_matrix_preds = U_imputed

        return None      
    
    def get_rec(self, user_list, top_n, uc_assignment=None):

        """Returns the top N item cluster recommendations for each user in the user list

                Parameters:
                        utility_matrix (numpy.ndarray): Matrix of utilities for each user-item pairing
                        utility_matrix_o (numpy.ndarray): Original utility matrix, before imputation
                        user_list (array-like): List of users
                        uc_assignment (array-like): List containing the cluster assignment of each user
                        top_n (int): Number of item clusters to recommend

                Returns:
                        df_rec (pandas.DataFrame): Table containing the top N item cluster recommendations for each user in the user list

        """

        utility_matrix_o = self.utility_matrix.fillna(0).values
        utility_matrix = self.utility_matrix_preds.values

        # Don't recommend items that are already rated
        utility_matrix[np.where(utility_matrix_o != 0)] = -np.inf

        # Get top N per user cluster
        cluster_rec = utility_matrix.argsort()[:, -top_n:]

        # Create recommendation table
        df_rec = pd.DataFrame()
        df_rec['user_id'] = user_list

        for i in range(top_n):
            df_rec['rank_'+str(i+1)] = np.zeros(df_rec.shape[0])
            for j in range(df_rec.shape[0]):
                if uc_assignment is None:
                    df_rec.iloc[j, i+1] = cluster_rec[user_list[j], top_n-i-1]
                else:
                    df_rec.iloc[j, i+1] = cluster_rec[uc_assignment[user_list[j]], top_n-i-1]

        self.df_rec = df_rec
        return df_rec    