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
                        
            Final outputs:
                    recommendations (pandas.DataFrame): Dataframe with columns user_id, item_id, score
                        |user_id|item_id|rating|
                        |=======|=======|======|
                        | 1     | 3     | 2    |                    
    """
    
    def __init__(self, user_df, item_df, transaction_list, 
                 item_id_name='movieId', user_id_name='userId'):
        self.user_df = user_df
        self.item_df = item_df
        self.transaction_list = transaction_list  
        self.df = self.get_augmented_table() # mixed user id, item id, user features, item features, transaction ratings

        self.item_id_name = item_id_name
        self.user_id_name = user_id_name
                
        self.item_ids = self.item_df[item_id_name].unique()
        self.user_ids = self.df[user_id_name].unique()
        
        
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
        augmented_tt = self.transaction_list.merge(self.user_df, on='userId', how='left')
        augmented_tt_2 = augmented_tt.merge(self.item_df, on='movieId', how='left')
        augmented_tt_2 = augmented_tt_2.fillna(0)
        
        return augmented_tt_2     