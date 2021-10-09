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

class CollabFilteringModel:
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
    
    
    def nan_mask(self, U, p=0.2):
        """Randomly sets values of the utility matrix to NaN

        Parameters:
                U (numpy.array): utility matrix (rows are users, columns are items) 
                p (float): percentage of matrix which will be set to NaN, 
                    value ranges from 0 to 1, default 0.2

        Returns:
                U*mask (numpy.array): utility matrix masked with NaNs
        """    
        mask = np.ones(np.shape(U))
        random_index = np.random.choice(U.size, size=int(U.size*p), replace=False)
        np.ravel(mask)[random_index] = np.nan
        return U*mask     
    
    
    def gen_missing_ratings(self, U_df, p=0.2, n_masks=10):
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
            masked_um.append(pd.DataFrame(self.nan_mask(U_arr, p=p),
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


    def initialize_models_userwise(self, U, model, suffix='model'):
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

            metric = np.std(np.abs(np.array(pred_curr)-np.array(pred_prev)))

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


    def train_model_iterative(self,
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
            U=U, split_axis=1, missing_val_filled=False)

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
            U_df, model_object, d=2, return_models=True, verbose=True):
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

        models_item = self.initialize_models_itemwise(
            model=model_object, U=U, suffix='')

        known_index, missing_index = self.known_missing_split_U(
            U=U, split_axis=1, missing_val_filled=False)

        U_update = U.copy()

        models_item = self.initialize_models_itemwise(model=model_object, U=U, suffix='')

        training_count = 0
        item_total = U.shape[1]
        
        for item in U.columns:
            training_count+=1
#             print(item, len(known_index[item]))
            if len(known_index[item])>0:
                U_temp = U.drop(item, axis=1)
                U_temp = self.mean_filled_utilmat(U_temp).fillna(0)
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
            if verbose:
                if (training_count%100==0)|(training_count==item_total):
                    print(f'Done training {training_count} out of {item_total}')


        if return_models:
            return U_update, models_item
        else:
            return U_update   
        
        
        
    def mean_filled_utilmat(self, U, axis=1):
        if axis:
            return U.T.fillna(U.mean(axis=axis)).T
        else:
            return U.fillna(U.mean(axis=axis))
       
        
    def train_model_iterative_cluster(self,
            Uc_df, model_object, n_synth_data=100, p=0.2, return_models=True):
        """Trains model iteratively for the cluster-based recommender system: 
        (1) Given cluster-based utility matrix, create multiple synthetic data of 
        missing ratings. Randomly drop matrix elements by setting them to NaN to 
        create "missing" ratings. 
        (2) For each set of synthetic data:
            (2a) Estimate the missing entries of each column/item by setting it as 
            the target variable and the remaining columns as the feature variables. 
            (2b) For the remaining columns, the current set of filled in values are 
            used to create a complete matrix of feature variables. 
            (2c) The observed ratings in the target column are used for training. 
            (2d) The missing entries are updated based on the prediction of the 
            model on each target column. 
        (3) Get mean of the completed utility matrix from all imputed synthetic data. 

        Parameters:
                Uc_df (DataFrame): output utility matrix from clustering
                    (rows are users, columns are items) 
                model_object: model object to use to fit the data
                n_synth_data (int): number of synthetic datasets to be generated,
                    default 100
                p (float): percentage of matrix which will be set to NaN, 
                    value ranges from 0 to 1, default 0.2         

        Returns:
                (DataFrame): updated cluster-based utility matrix 

        """
        # VARS
        
        synth_data = self.gen_missing_ratings(Uc_df, p=p, n_masks=n_synth_data)
        um_output = []
        for n in range(n_synth_data):
            U_df = synth_data[n]
            U_imputed, metrics, models = self.train_model_iterative(
                U_df, model_object, return_models=return_models)
            um_output.append(U_imputed)
        um_output = pd.concat(um_output)

        # final preds
        self.utility_matrix_preds = um_output.groupby(um_output.index).mean()

        return self.utility_matrix_preds

    def train_model_svd_cluster(self,
            Uc_df, model_object, n_synth_data=100, d=10, p=0.2, return_models=True):
        """Trains model iteratively for the cluster-based recommender system: 
        (1) Given cluster-based utility matrix, create multiple synthetic data of 
        missing ratings. Randomly drop matrix elements by setting them to NaN to 
        create "missing" ratings. 
        (2) For each set of synthetic data:
            (2a) Estimates the missing entries of the utility matrix.
            (2b) Each column/item is set as the target variable one at a time, and
            the remaining columns are set as the feature matrix.
            (2c) SVD is performed on the feature matrix before model training.
            (2d) Rows with missing items are in the test set, while the rest are in 
            the training set.
            (2e) Process is repeated for all columns/items, yielding a completed 
            utility matrix.
        (3) Get mean of the completed utility matrix from all imputed synthetic data. 

        Parameters:
                Uc_df (DataFrame): output utility matrix from clustering
                    (rows are users, columns are items) 
                model_object: model object to use to fit the data
                n_synth_data (int): number of synthetic datasets to be generated,
                    default 100
                p (float): percentage of matrix which will be set to NaN, 
                    value ranges from 0 to 1, default 0.2         

        Returns:
                (DataFrame): updated cluster-based utility matrix 

        """
        # VARS
        
        synth_data = self.gen_missing_ratings(Uc_df, p=p, n_masks=n_synth_data)
        um_output = []
        for n in range(n_synth_data):
            U_df = synth_data[n]
            U_imputed, models = self.train_model_svd(
                U_df, model_object, d=d, return_models=return_models)
            um_output.append(U_imputed)
        um_output = pd.concat(um_output)

        # final preds
        self.utility_matrix_preds = um_output.groupby(um_output.index).mean()

        return self.utility_matrix_preds    



    def fit(self, model_object, method="iterative", n_synth_data=5,
            p=0.1, d=2, return_models=False):
        U_df_mc = self.mean_center_utilmat(self.utility_matrix, axis=1, fillna=False)

        if method == 'iterative':        
            if self.users_clustered or self.items_clustered: # if clustered
                
                self.utility_matrix_preds = self.train_model_iterative_cluster(
                    self.utility_matrix, model_object=model_object, 
                    n_synth_data=n_synth_data, p=p)            

            else: # if not clustered    
                self.models_item = self.initialize_models_itemwise(
                    self.utility_matrix, model_object, suffix='')
                if return_models:
                    U_imputed, metrics, trained = self.train_model_iterative(
                        self.utility_matrix, model_object,
                        return_models=return_models) 
                    self.utility_matrix_preds = U_imputed.add(U_df_mc.mean(axis=1), axis=0)
                    self.trained_models = trained
                else:
                    U_imputed, metrics = self.train_model_iterative(
                        self.utility_matrix, model_object,
                        return_models=return_models) 
                    self.utility_matrix_preds = U_imputed.add(U_df_mc.mean(axis=1), axis=0)
                    self.trained_models = {}


        # works for both clustered or unclustered?
        if method == 'svd':
            if self.users_clustered or self.items_clustered: # if clustered

                self.utility_matrix_preds = self.train_model_svd_cluster(
                    self.utility_matrix, model_object=model_object, 
                    n_synth_data=n_synth_data, p=p, d=d)     
                
            else: 
                self.models_item = self.initialize_models_itemwise(
                    self.utility_matrix, model_object, suffix='')
                
                if return_models:
                    U_imputed, trained = self.train_model_svd(
                        self.utility_matrix, model_object, d=d,
                        return_models=return_models) 
                    self.utility_matrix_preds = U_imputed
                    self.trained_models = trained                   
                else:
                    U_imputed  = self.train_model_svd(
                        self.utility_matrix, model_object, d=d, 
                        return_models=return_models)        
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
                        df_rec (pandas.DataFrame): Table containing the top N item cluster 
                        recommendations for each user in the user list

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
                    df_rec.iloc[j, i+1] = cluster_rec[uc_assignment.iloc[user_list[j], 0], top_n-i-1]

        # look-up tables
        if uc_assignment is None:
            user_id_lookup = self.utility_matrix_preds.index
            item_id_lookup = self.utility_matrix_preds.columns
            for j in range(df_rec.shape[0]):
                df_rec.iloc[j, 0] = user_id_lookup[df_rec.iloc[j, 0].astype('int32')]
                for i in range(top_n):
                    df_rec.iloc[j, i+1] = item_id_lookup[df_rec.iloc[j, i+1].astype('int32')]

        self.df_rec = df_rec
        return df_rec
    
    def get_rec_item(self, top_k):

        """Returns the top K item recommendations for each user in the user list. 
        Items are selected randomly from the top recommended item cluster, exhaustively. Left overs are taken from the next highest ranked item clusters in a cascading fashion.

                Parameters:
                        df_rec (pandas.DataFrame): Table containing the top N item cluster recommendations for each user in the user list
                        ic_assignment (array-like): List containing the cluster assignment of each item
                        top_n (int): Number of items to recommend

                Returns:
                        df_rec_item (pandas.DataFrame): Table containing the top K item recommendations for each user in the user list

        """
        df_rec = self.df_rec # recommendations after running get_rec()
        ic_assignment = self.item_assignment # item-cluster assignment

        # Create recommendation table
        df_rec_item = pd.DataFrame()
        df_rec_item['user_id'] = df_rec['user_id']  

        for i in range(top_k):
            df_rec_item['rank_'+str(i+1)] = np.zeros(df_rec_item.shape[0])

        # Get items
        for j in range(df_rec_item.shape[0]):
            item_rec = []
            rank = 0
            while len(item_rec) < top_k:
                if rank+1 >= df_rec.shape[1]:
                    item_list = list(set(self.transaction_list['item_id'])-set(item_rec))
                    item_rec = item_rec + list(np.random.choice(item_list, size=top_k-len(item_rec), replace=False))
                    break
                item_list = ic_assignment.index[np.where(ic_assignment == df_rec.iloc[j, rank+1])[0]]
                if top_k-len(item_rec) > len(item_list):
                    item_rec = item_rec + list(item_list)
                    rank += 1
                else:
                    item_rec = item_rec + list(np.random.choice(item_list, size=top_k-len(item_rec), replace=False))
            df_rec_item.iloc[j, 1:] = item_rec

        # look-up tables
        user_id_lookup = self.user_assignment.index
        for j in range(df_rec_item.shape[0]):
            df_rec_item.iloc[j, 0] = user_id_lookup[df_rec_item.iloc[j, 0].astype('int32')]

        return df_rec_item    
    
    
### CLUSTERED VERSION

    def cluster_users(self, model):
        """
        Perform user-wise clustering and assign each user to a cluster.

        Paramters
        ---------                  
        model        : an sklearn model object
                       An object with a fit_predict method. Used to cluster the
                       users into groups with similar ratings of items.

        Returns
        -------
        model         : an sklearn model object
                        The fitted version of the model input used to predict the
                        clusters of users from fname

        result        : dict
                        A mapping of each user's cluster with the keys being the
                        user_id and the values their cluster membership

        df            : pandas DataFrame
                        Utility matrix derived from fname with the final column
                        corresponding to the cluster membership of that user
        """

        # SOME VARIABLES
        df = self.utility_matrix # utility matrix    
        df = df.fillna(0) # fillna with 0

        # Aggregation through tables
        u_clusterer = model
        u_predict = u_clusterer.fit_predict(df)
        df['u_cluster'] = u_predict

        model = u_clusterer
        result = dict(df['u_cluster'])

        # Output variables
        self.user_cluster_model = model # attach the user_cluster_model to the class
        self.utility_matrix_w_user_clusters = df # utility matrix with user clusters
        self.user_cluster_mapping_dict = result # mapping of users and cluster labels
        self.users_clustered = True # tag that we clustered the users

        return model, result, df
    
    
    
    def cluster_items(self, model):

        # WE MIGHT WANT TO FIX TO DROP COLS AS HARD CODED INSTEAD OF AN ARGUMENT
        # SO LONG AS WE STANDARDIZE THE INPUT

        """
        Perform item-wise clustering and assign each item to a cluster of similar
        items based on the users that 

        Paramters
        ---------

        model        : an sklearn model object
                       An object with a fit_predict method. Used to cluster the
                       users into groups with similar ratings of items.

        Returns
        -------
        model         : an sklearn model object
                        The fitted version of the model input used to predict the
                        clusters of items from fname

        result        : dict
                        A mapping of each item's cluster with the keys being the
                        item_id and the values their cluster membership

        df_items      : pandas DataFrame
                        Utility matrix derived from fname with the final column
                        corresponding to the cluster membership of that item
        """

        # SOME VARIABLES
        df = self.utility_matrix # utility matrix      
        df = self.utility_matrix # utility matrix    
        df = df.fillna(0) # fillna with 0

        df_items = df.T
        i_clusterer = model

        i_predict = i_clusterer.fit_predict(df_items)
        df_items['i_cluster'] = i_predict

        model = i_clusterer
        result = dict(df_items['i_cluster'])

        # Output variables
        self.item_cluster_model = model # attach the item_cluster_model to the class
        self.utility_matrix_w_item_clusters = df_items # utility matrix with item clusters
        self.item_cluster_mapping_dict = result # mapping of users and cluster labels    
        self.items_clustered = True # tag that we clustered the items

        return model, result, df_items    
    
    
    def cluster_assignment(self):

        """
        Converts the dictionary containing user_id and user_cluster assignment  
        to a pandas data frame 

        Returns
        -------
        result        : dataframe of cluster assignments

        """

        if self.users_clustered: # if we ran the cluster_users method: 
            data_name='user_id'        
            cluster_name='u_cluster'        
            self.user_assignment = pd.DataFrame(list(self.user_cluster_mapping_dict.items()), columns=[data_name, cluster_name])
            self.user_assignment.set_index(data_name, inplace=True)

        if self.items_clustered: # if we ran the cluster_users method: 
            data_name='item_id'        
            cluster_name='i_cluster'        
            self.item_assignment = pd.DataFrame(list(self.item_cluster_mapping_dict.items()), columns=[data_name, cluster_name])
            self.item_assignment.set_index(data_name, inplace=True)

        return None

    def utility_matrix_agg(self, u_agg='mean', i_agg='mean'):
        """
        Aggregates the results of the clustering with respect to item clusters and user clusters. 
        ------
        Methods : two possible ways to aggregate the results of cluster assignments in df_u and df_i are 'sum' and 'mean'
        u_agg   : aggregration method to be used for users

        i_agg   : aggregation method to be used for items

        -----
        Returns : utility matrix consisting of the aggregrated user clusters as rows and aggregated item clusters as columns

        """

        # GET utility matrices with cluster labels
        df_u = self.utility_matrix_w_user_clusters
        df_i = self.utility_matrix_w_item_clusters

        u_series = df_u['u_cluster']
        i_series = df_i['i_cluster']

        u_ids = np.unique(u_series.values)
        i_ids = np.unique(i_series.values) 

        u_feats = {}
        for u_id in u_ids: #u_ids are clusters of u_id
            sub_df = df_u.groupby('u_cluster').get_group(
                u_id).drop(columns=['u_cluster']).T
            sub_df = sub_df.merge(i_series, left_index=True, right_index=True)

            if u_agg == 'sum':
                df_grp = sub_df.groupby('i_cluster').sum()
            if u_agg == 'mean':
                df_grp = sub_df.groupby('i_cluster').mean()
            if not isinstance(u_agg,str):
                df_grp = sub_df.groupby('i_cluster').apply(u_agg)

            if i_agg == 'sum':
                df_grp = df_grp.sum(axis=1)
            if i_agg == 'mean':
                df_grp = df_grp.mean(axis=1)
            if not isinstance(i_agg,str):
                df_grp = df_grp.apply(i_agg, axis=1)

            u_feats[u_id] = df_grp


        u_matrix = pd.DataFrame()
        for k, v in u_feats.items():
            u_matrix = u_matrix.merge(v.rename(k), how='outer',
                                      left_index=True, right_index=True)

        # UPDATE THE UTILITY MATRIX
        self.utility_matrix = u_matrix.fillna(0).T 
        self.utility_matrix.index.rename('u_cluster', inplace=True)
        return self.utility_matrix   
    
    
    
