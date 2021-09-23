import pandas as pd
import numpy as np


def mean_filled_utilmat(U, axis=1):
    if axis:
        return U.T.fillna(U.mean(axis=axis)).T
    else:
        return U.fillna(U.mean(axis=axis))

def mean_center_utilmat(U, axis=1, fillna=True, fill_val=None):
    """Gets the mean-centered utility matrix

    Parameters:
        U (DataFrame) : utilily matrix (rows are users, columns are items) 
        axis (int) : The axis along mean is evaluated, 
            {0/'index', 1/'columns'}, default 1
        fillna (bool) : Indicates whether missing/null values are to be filled
        fill_val (None/float) : Value to be used to fill null values when 
            fillna==True, default None

    Returns:
        U (DataFrame): mean-centered utility matrix
    """
    mean_centered = U.sub(U.mean(axis=axis), axis=1-axis)
    if fillna:
        if fill_val is not None:
            return mean_centered.fillna(fill_val)
        else:
            return mean_centered.fillna(0)
    else:
        return mean_centered


def split_utilmat_label_features(U, label_index, axis=1):
    """Splits utility matrix into label (column/row where ratings are predicted) 
    and features (columns/rows to be used as input in the model)

    Parameters:
        U (DataFrame) : utilily matrix (rows are users, columns are items) 
        label_index : column name or index corresponding to  item ratings (column)
            or user ratings (row) to be predicted
        axis (int) : The axis along the utility matrix is split, 
            {0/'index', 1/'columns'}, default 1

    Returns:
        label_df (DataFrame) : contains the column/row to be predicted
        feature_df (DataFrame) : contains the features   
    """
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


def known_missing_split_U(U, split_axis=1, missing_val_filled=False,
                          fill_val=None):
    """Returns index of the dataset corresponding to known and missing ratings
    in for the whole utility matrix

    Parameters:
        U (DataFrame) : utilily matrix (rows are users, columns are items) 
        split_axis (int) : The axis along the utility matrix is split, 
            {0/'index', 1/'columns'}, default 1
        missing_val_filled (bool) : Indicates whether missing/null values 
            in the label/feature data were filled
        fill_val (None/float) : Value used to fill the null values when 
            missing_val_filled==True, default None            

    Returns:
        known_idx (dict): keys are the column name/index to be predicted, 
            values are index of the utility matrix that contains known values
        missing_idx (dict): keys are the column name/index to be predicted, 
            values are index of the utility matrix that contains missing values
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