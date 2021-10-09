# +
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# -


def reco_ml_cb_tt(df_test, model_fitted, target='rating', drop_cols=['userId', 'movieId', 'timestamp']):
    """
    Make predictions on the test set and outputs an array of the predicted
    values for them.

    Paramters
    ---------
    df_test      : pandas DataFrame
                   The test set as a transaction table. Each row
                   corresponds to a user's features and that item's features
                   along with the user's rating for that item.

    model_fitted : an sklearn regressor object
                   An object with a fit and predict method that outputs a
                   float. Must be fitted already

    target_col   : str
                   The column corresponding to the rating.
                   
    drop_cols    : list
                   Columns to be dropped in df_test.

    Returns
    -------
    result        : numpy array
                   The results of the model using df_test's features
    """
    df_test = df_test.drop(columns=[target]+drop_cols)
    result = model_fitted.predict(df_test)
    return result





def fit_ml_cb(train_df, model, target_col='rating', drop_cols=['userId', 'movieId', 'timestamp']):
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



def reco_ml_cb(user_df, item_df, item_ids, model_fitted):
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


def split_train_test(data, train_ratio=0.7,uid='userId', iid='movieId', rid='rating'):
    """
    Splits the transaction data into train and test sets.
    
    Parameters
    ----------
    data         : pandas DataFrame for transaction table containing user, item, and ratings
    
    train_ratio  : the desired ratio of training set, while 1-train ratio is automatically set for the test set 
    
    
    Returns
    ---------
    df_train_fin : dataframe for the training set
    
    df_test_fin  : dataframe for the test set
    
    df_test_fin* : possible option is a pivoted df ready as the util matrix input of the recsys. In our case, the
                   index='userId', columns='movieId', values='rating'. To generalize a transaction table, 
                   index=column[0], columns=itemId, values=rating.
    """
    
    list_df_train = []
    list_df_test = []
    
    #group by user id
    d = dict(tuple(data.groupby(data.columns[0]))) #assuming column[0] is the userId
    
    #splitting randomly per user
    for i in (d):
        if len(d[i])<2:
            list_df_test.append(d[i])
            
        else:            
            df_train = d[i].sample(frac=train_ratio)  
            ind = df_train.index
            df_test = d[i].drop(ind)
            list_df_train.append(df_train) 
            list_df_test.append(df_test)

    # 2. merge selected train set per user to a single dataframe
    df_train_fin = pd.concat(list_df_train)
    df_test_fin = pd.concat(list_df_test)
    
    # 3. Option to pivot it to create the utility matrix ready as input for recsys
    df_test_um = df_test_fin.pivot(index=uid, columns=iid, values=rid)
    
    # 4. get indices of train and test sets
    indx_train = df_train_fin.index
    indx_test = df_test_fin.index
    
    return df_train_fin, df_test_fin, df_test_um, indx_train, indx_test #return indices


def split_train_test_chronological(data, train_ratio=0.7):
    """
    Splits the transaction data into train and test sets based on chronological order.
    
    Parameters
    ----------
    data         : pandas DataFrame for transaction table containing user, item, and ratings
    
    train_ratio  : the desired ratio of training set, while 1-train ratio is automatically set for the test set 
    
    
    Returns
    ---------
    df_train_fin : dataframe for the training set
    
    df_test_fin  : dataframe for the test set
    
    df_test_fin* : possible option is a pivoted df ready as the util matrix input of the recsys. In our case, the
                   index='userId', columns='movieId', values='rating'. To generalize a transaction table, 
                   index=column[0], columns=itemId, values=rating.
    """
    
    # 1. check if the data has timestamp
    col = 'timestamp'
    if col not in data.columns:
    #     print('column does not exist')
        raise ValueError('could not find %s in %s' % (col,list(data.columns)))
    
    # 2. split data into train and test. test set is automatically the last 30% of the data set
    list_df_train = []
    list_df_test = []
    
    #group by user id
    d = dict(tuple(data.groupby(data.columns[0]))) #assuming column[0] is the userId
    
    #splitting randomly per user
    for i in (d):
        if len(d[i])<2:
            print(len(d[i]))
            list_df_test.append(d[i])
            
        else:
            d[i].sort_values('timestamp', inplace=True)
            df_train = d[i].iloc[0:int(train_ratio*len(d[i])),:]
            
            ind = df_train.index
            df_test = d[i].drop(ind)
            
            list_df_train.append(df_train) 
            list_df_test.append(df_test)

    # 3. merge selected train set per user to a single dataframe
    df_train_fin = pd.concat(list_df_train)
    df_test_fin = pd.concat(list_df_test)
    
    # 4. Option to pivot it to create the utility matrix ready as input for recsys
    df_test_um = df_test_fin.pivot_table(index=df_test_fin.columns[0], columns=df_test_fin.columns[1], values=df_test_fin.columns[2])
    
    # 5. get indices of train and test sets
    indx_train = df_train_fin.index
    indx_test = df_test_fin.index

    return df_train_fin, df_test_fin, df_test_um, indx_train, indx_test 


def evaluate_arrays(model_result_arr, df_data, indx_test):
    """
    Calculates the mse and mae of the recommender system for a given result and test set.
    
    Parameters
    ----------
    
    model_result_arr   : ratings from the results of the recommender sys using test set
    
    df_test_truth      : the original dataframe for before splitting.
                         the original ratings or ground truth from the test set will be extracted from here using indices
                         
    indx_test          : result indices of test set from splitting
    
    Returns
    ---------
    
    mse                : mse value using sklearn 
    
    mae                : mse value using sklearn 
    
    """
    
    df_test_truth = df_data.loc[pd.Index(indx_test), df_data.columns[2]]
    test_arr = df_test_truth.values
         
#     test indices first, all user ids should be represented in the test matrix 

    result_len = len(model_result_arr) 
    test_len = len(test_arr)
      
    if result_len!=test_len:
        raise ValueError('the arrays are of different lengths %s in %s' % (result_len,test_len))
        
    else:
        print('proceed')
            
        mse = mean_squared_error(test_arr, model_result_arr)
        mae = mean_absolute_error(test_arr, model_result_arr)

            
    return mse, mae


# +
def evaluate(df_test_result, df_test_data):
    """
    Calculates the mse and mae per user of the results of the recommender system for a given test set.
    
    Parameters
    ----------
    
    df_test_result   : utility matrix containing the result of the recommender systems
    
    df_test_data     : pivoted test data generated from splitting the transaction table and tested on the recommender systems
    
    Returns
    ---------
    
    mse_list         : list of mean squared error for each user
    
    mae_list         : list of mean absolute error for each user
    
    """
    
    
    mse_list = []
    mae_list = []
    
#     test indices first, all user ids should be represented in the test matrix 
    idx_orig_data = df_test_data.index
    idx_result = df_test_result.index
    a=idx_orig_data.difference(idx_result)
    
    if len(a)==0:
        print('proceed')
        for i in (df_test_result.index):
            y_pred = df_test_result[df_test_result.index==i].fillna(0)
            y = df_test_data[df_test_data.index==i].fillna(0)
            y_pred = y_pred[y.columns]

            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            mse_list.append(mse)
            mae_list.append(mae)
    else:
        print('error')
    
    return mse_list, mae_list


# -

def append_error_to_df(test_result, mse, mae):
    """
    Inserts the error values into the first two rows of the dataframe of the predictions of system for easy visualization
    and for further computations.
    
    Parameters
    ----------
    
    test_result   : utility matrix for the result of the recommender systems on the test set
    
    mse           : mse computed from function evaluate
    
    mae           : mae computed from function evaluate
    
    Returns
    -------
    
    test_result   : modified utility matrix with errors
    """
    
    test_result.insert(0, 'mse_u', mse)
    test_result.insert(0, 'mae_u', mae)
    
    return test_result


def cross_val(df, k, model, split_method='random'):
    """
    Performs cross-validation for different train and test sets.

    Parameters
    -----------
    df                    : the data to be split in the form of vanilla/transaction++ table (uid, iid, rating, timestamp)

    k                     : the number of times splitting and learning with the model is desired
    
    model                 : an unfitted sklearn model

    split_method          : 'random' splitting or 'chronological' splitting of the data


    Returns
    --------
    mse and mae           : error metrics using sklearn


    """
    mse = []
    mae = []

    if split_method == 'random':

        for i in range(k):
            print(i)
            # 1. split
            print('Starting splitting')
            df_train, df_test, df_test_um, indx_train, indx_test = split_train_test(
                df, 0.7)
            print('Finished splitting')
            # 2. train with model
            model_clone = clone(model)
            print('Starting training')
            model_clone_fit = fit_ml_cb(df_train, model_clone)
            print('Finished training')
            print('Starting completing matrix')
            result = reco_ml_cb_tt(df_test, model_clone_fit)
            print('Finished completing matrix')
            print('Starting computing MAE and MSE')
            # 3. evaluate results (result is in the form of utility matrix)
            mse_i, mae_i = evaluate_arrays(result, df, indx_test)
            print('Finished computing MAE and MSE')

            mse.append(mse_i)
            mae.append(mae_i)

    elif split_method == 'chronological':

        # 1. split
        df_train, df_test, df_test_um, indx_train, indx_test = split_train_test_chronological(
            df, 0.7)

        print('Starting splitting')
        print('Finished splitting')
        # 2. train with model
        model_clone = clone(model)
        print('Starting training')
        model_clone_fit = fit_ml_cb(df_train, model_clone)
        print('Finished training')
        print('Starting completing matrix')
        result = reco_ml_cb_tt(df_test, model_clone_fit)
        print('Finished completing matrix')
        print('Starting computing MAE and MSE')
        # 3. evaluate results (result is in the form of utility matrix)
        mse_i, mae_i = evaluate_arrays(result, df, indx_test)
        print('Finished computing MAE and MSE')

        mse.append(mse_i)
        mae.append(mae_i)

    return mse, mae