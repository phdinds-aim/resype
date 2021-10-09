import numpy as np
import pandas as pd
from sklearn.base import clone


def fit_ml_cb(train_df, model, target_col='rating', drop_cols=['userId', 'movieId','timestamp']):
    """
    Perform item-wise clustering and assign each item to a cluster of similar
    items based on the users that 

    Paramters
    ---------
    train_df     : pandas DataFrame
                   The training set as a transaction table. Each row
                   corresponds to a user's features and that item's features
                   along with the user's rating for that item.

    model        : an sklearn regressor object
                   An object with a fit and predict method that outputs a
                   float.

    target_col   : str
                   The column corresponding to the rating.

    drop_cols    : list
                   Columns to be dropped in train_df.

    Returns
    -------
    rs_model      : an sklearn model object
                    The fitted version of the model input used to predict the
                    rating of a user for an object given the user's features
                    and the item's features.
    """
    rs_model = clone(model)
    target = train_df[target_col].dropna().values.ravel()
    train_df = train_df.drop(columns=[target_col]+drop_cols)
    rs_model = model.fit(train_df, target)
    return rs_model


def reco_ml_cb(user_df, item_df, model_fitted):
    """
    Completes the entire utility matrix based on the model passed

    Paramters
    ---------
    train_df     : pandas DataFrame
                   The training set as a transaction table. Each row
                   corresponds to a user's features and that item's features
                   along with the user's rating for that item.

    model        : an sklearn regressor object
                   An object with a fit and predict method that outputs a
                   float.

    target_col   : str
                   The column corresponding to the rating.
                   
    Returns
    -------
    full_matrix  : a pandas DataFrame
                   The completed utility matrix.
    """
    recos = {}
    c = 1
    for u, u_feats in user_df.iterrows():
        print(c, 'out of', len(user_df), end='\r')
        u_feats = pd.concat([pd.DataFrame(u_feats).T] *
                            len(item_ids)).reset_index(drop=True)
        a_feats = u_feats.join(item_df)
        reco = pd.Series(model_fitted.predict(a_feats), index=item_ids)
        recos[u] = reco
        c += 1
    full_matrix = pd.DataFrame.from_dict(recos, orient='index')
    return full_matrix
