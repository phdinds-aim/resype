from utils import preprocess_utility_matrix as um
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
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


def train_model_svd(
        U_df, model_object, d, return_models=True):
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
        d : number of desired dimensions after dimensionality reduction
        return_models (bool) : Indicates whether trained models are returned 
            as output, default True

    Returns:
        U_update (DataFrame) : complete utility matrix
        models_item (dict) : dictionary of trained models, returned only if
            return_models=True
    """
    U = U_df.copy()
    
    known_index, missing_index = um.known_missing_split_U(
        U, split_axis=1)

    len_missing_vals = len(sum([i.tolist()
                                for i in missing_index.values()], []))
    
    U = um.mean_filled_utilmat(U)
    U_update = U.copy()
    
    models_item = initialize_models_itemwise(model_object, U, suffix='')

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