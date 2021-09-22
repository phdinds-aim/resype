import pandas as pd
import numpy as np

def mean_center_utilmat(U, axis=1, fillna=True, fill_val=None): 
    mean_centered = U.sub(U.mean(axis=axis), axis=1-axis)
    if fillna:
        if fill_val is not None: 
            return mean_centered.fillna(fill_val)
        else: 
            return mean_centered.fillna(0)
    else: 
        return mean_centered
    
def split_utilmat_label_features(U, label_index, axis=1):
    if axis==1: 
        label_col = U.columns[U.columns==label_index]
        feature_col = U.columns[~(U.columns==label_index)]
        return U.loc[:, label_col], U.loc[:, feature_col]
    elif axis==0:
        label_row = U.index[U.index==label_index]
        feature_row = U.index[~(U.index==label_index)]
        return U.loc[label_row, :], U.loc[feature_row, :]    
    
def known_missing_split_1d(label_data, feature_data, split_axis=1, missing_val_filled=False, fill_val=None):
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

def known_missing_split_U(U, split_axis=1, missing_val_filled=False, fill_val=None):
    if missing_val_filled:
        if fill_val is None:
            missing_val = 0
        else:
            missing_val = fill_val
        if split_axis == 1:
            known_idx = dict((U==missing_val).T.apply(lambda x: np.array(
                x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
            missing_idx = dict((U==missing_val).T.apply(lambda x: np.array(
                x), axis=1).apply(lambda x: U.index[np.argwhere(x).flatten()]))
        elif split_axis == 0:
            known_idx = dict((U==missing_val).apply(lambda x: np.array(
                x), axis=1).apply(lambda x: U.T.index[np.argwhere(~x).flatten()]))
            missing_idx = dict((U==missing_val).apply(lambda x: np.array(x), axis=1).apply(
                lambda x: U.T.index[np.argwhere(x).flatten()]))
        else:
            print('Invalid axis. Result for axis=1 is returned.')
            known_idx = dict((U==missing_val).T.apply(lambda x: np.array(
                x), axis=1).apply(lambda x: U.index[np.argwhere(~x).flatten()]))
            missing_idx = dict((U==missing_val).T.apply(lambda x: np.array(
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