import numpy as np
import pandas as pd

def get_rec(utility_matrix, utility_matrix_o, user_list, top_n, uc_assignment=None):
    
    """Returns the top N cluster recommendations for each user in the user list.
    
            Parameters:
                    utility_matrix (numpy.ndarray): Matrix of utilities for each user-item pairing (assumes that indices correspond to user_cluster_id and item_cluster_id)
                    utility_matrix_o (numpy.ndarray): Original utility matrix, before imputation (i need this so i dont recommend items that have already been "consumed"/"rated")
                    user_list (array-like): List of users
                    uc_assignment (array-like): List containing the cluster assignment of each user (assumes that indices correspond to user_id)
                    top_n (int): Number of item clusters to recommend

            Returns:
                    df_rec (pandas.DataFrame): Table containing the top N recommendations for each user in the user list
                    
    """
    
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
                
    return df_rec