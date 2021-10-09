import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class ContentBasedModel:
    """
    Resype implements a machine learning framework for recommender systems.
    Content based machine learning model for recommender systems
            Parameters:
                    user_df (pandas.DataFrame): Dataframe with columns userId and user features
                        |userId|feat1|feat2|
                        |======|=====|=====|
                        | 1    | 1   | 4   |
                        
                    item_df (pandas.DataFrame): Dataframe with columns itemId and item features
                        |userId|feat1|feat2|
                        |======|=====|=====|
                        | 1    | 1   | 4   | 
                        
                    transaction_list (pandas.DataFrame): Dataframe with columns user_id, item_id, rating in the form
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 1     | 4    |                  
                        
                    user_id_name (str): default = 'userId' - user id column name
                    item_id_name (str): default = 'movieId' - item id column name
                    target_name (str): default = 'rating' - target column name
                    timestamp_name (str): default = 'timestamp' - timestamp column name                    
                        
            Final outputs:
                    recommendations (pandas.DataFrame): Dataframe with columns user_id, item_id, score
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 3     | 2    |                    
    """
    
    def __init__(self, user_df, item_df, transaction_list, 
                 item_id_name='movieId', user_id_name='userId', 
                 target_name='rating', timestamp_name = 'timestamp'):
        
        self.user_df = user_df
        self.item_df = item_df
        self.transaction_list = transaction_list 

        self.item_id_name = item_id_name
        self.user_id_name = user_id_name
        self.timestamp_name = timestamp_name
        self.target_name = target_name           
                
        self.df = self.get_augmented_table() # mixed user id, item id, user features, item features, transaction ratings  
        self.item_ids = self.item_df[item_id_name].unique()
        self.user_ids = self.df[user_id_name].unique()      
        
        self.construct_utility_matrix() # generate utility matrix
        
    # PREPROCESSING
    
    def construct_utility_matrix(self):
        self.utility_matrix = self.transaction_list.pivot(index=self.user_id_name, 
                                                          columns=self.item_id_name, 
                                                          values=self.target_name) # utility matrix 
        return None
    
    def get_augmented_table(self):
        '''
        Replace integrate user_features and item_features
        to the transaction_list

        Input
        ------
        user_df - user features
        item_df - item features
        transaction_list 

        Output
        -------
        augmented_transaction_table  : transaction_list concatenated with user_features
                                       from genres and item_features from movie synopsis


        '''
        augmented_tt = self.transaction_list.merge(self.user_df, on=self.user_id_name, how='left')
        augmented_tt_2 = augmented_tt.merge(self.item_df, on=self.item_id_name, how='left')
        augmented_tt_2 = augmented_tt_2.fillna(0)
        
        return augmented_tt_2   
    
    
     
    # TRAIN TEST SPLIT
    
    def split_train_test(self, train_ratio=0.7):
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

        data = self.df
                                                   
        uid = self.user_id_name
        iid = self.item_id_name
        rid = self.target_name
        
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
        
        
        
        self.df_train = df_train_fin
        self.df_test = df_test_fin 
        self.df_test_um = df_test_um 
        self.indx_train = indx_train 
        self.indx_test = indx_test
        
        return None

#        return df_train_fin, df_test_fin, df_test_um, indx_train, indx_test #return indices


    def split_train_test_chronological(self, train_ratio=0.7):
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

        
        data = self.df
        
        # 1. check if the data has timestamp
        col = self.timestamp_name
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
                d[i].sort_values(self.timestamp_name, inplace=True)
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
    
    
    # MODELS
    
    def fit_ml_cb(self, model):
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
        
        train_df = self.df_train
        target_col = self.target_name
        drop_cols = [self.user_id_name, self.item_id_name, self.timestamp_name]        
        
        rs_model = clone(model)
        target = train_df[target_col].dropna().values.ravel()
        train_df = train_df.drop(columns=[target_col]+drop_cols)
        rs_model = model.fit(train_df, target)
        
        self.model = rs_model
        
        return rs_model
    
    
    
    # PREDICTIONS
    
    def reco_ml_cb_tt(self):
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
        
        
        
        target = self.target_name
        drop_cols = [self.user_id_name, self.item_id_name, self.timestamp_name]
        
        df_test = self.df_test.drop(columns=[self.target_name]+drop_cols)
        result = self.model.predict(df_test)
        return result    
    
    
    # GET FILLED IN UTILITY MATRIX
    def reco_ml_cb(self):
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
        
        user_df = self.user_df
        item_df = self.item_df
        item_ids = self.item_ids 
        model_fitted = self.model # whether fit or not
        
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
        
        self.utility_matrix_preds = full_matrix
        
        return full_matrix    
    
    
    # GET RECOMMENDATIONS
    
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

        utility_matrix_preds = self.utility_matrix_preds
        utility_matrix = self.utility_matrix
        
        # Class stuff
        utility_matrix_o = utility_matrix.fillna(0).values
        utility_matrix = utility_matrix_preds.fillna(0).values

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
            user_id_lookup = utility_matrix_preds.index
            item_id_lookup = utility_matrix_preds.columns
            for j in range(df_rec.shape[0]):
                df_rec.iloc[j, 0] = user_id_lookup[df_rec.iloc[j, 0].astype('int32')]
                for i in range(top_n):
                    df_rec.iloc[j, i+1] = item_id_lookup[df_rec.iloc[j, i+1].astype('int32')]

        return df_rec    
    
    