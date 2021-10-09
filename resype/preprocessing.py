import numpy as np
import pandas as pd

import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer # get from VP later
from nltk.tokenize import word_tokenize
from collections import Counter    
from sklearn.feature_extraction.text import TfidfVectorizer

def create_user_feature():
    '''
    Return a user_feature matrix
    
    Takes in the transaction list from the Movielens 100k dataset
    and replaces the userId with a feature vector representing
    the number of movies seen by the user per genre
    
    possible genres include the following:
    'IMAX', 'Adventure', 'Mystery', 'Animation', 'Documentary', 'Comedy',
       'Western', 'War', 'Film-Noir', 'Crime', 'Drama', 'Thriller', 'Fantasy',
       'Action', 'Sci-Fi', 'Children', 'Romance', 'Horror', 'Musical',
       '(no genres listed)'
       
    Input
    ---------
    none
    
    
    Output
    ---------
    user_feature (pd.DataFrame): feature_vector containing number of count of 
                                 genres seen based on ratings given by a user
                                 - each movie can have several genres
                                 - each row correspond to a transaction (user rating)
    
    
    
    '''
    
    raw_transaction_list = pd.read_csv('sample_data/ratings.csv')
    transaction_list =  raw_transaction_list[['userId','movieId', 'rating']].copy()
    
    # reduce size of DataFrame for transaction_list by downcasting
    for col in transaction_list:
        if transaction_list[col].dtype == 'int64':
            transaction_list[col] = pd.to_numeric(transaction_list[col], downcast='integer')
        if transaction_list[col].dtype == 'float64':
            transaction_list[col] = pd.to_numeric(transaction_list[col], downcast='float')

    
    # preprocess movie list and genres
    movie_description = pd.read_csv('sample_data/movies.csv')    
    movie_description = movie_description.set_index('movieId')
    movie_description['genre'] = movie_description['genres'].str.split('|')
    
    # extract the genres for the movie in each transaction/rating
    movie_IDs_list = transaction_list['movieId']
    transaction_list['genre'] = list(movie_description.loc[movie_IDs_list[:len(movie_IDs_list)]]['genre'])

    # count the number of genres seen by each userId
    genre_count = (transaction_list.groupby('userId')['genre']
                     .apply(list)
                     .apply(lambda x: [item for sublist in x for item in sublist])
                     .apply(Counter))
    
    # remove genre column in transaction list (just to conserve memspace)
    del transaction_list['genre']
        
    # create user_feature with count of genres per user
    user_feature = pd.DataFrame(list(genre_count)).fillna(0)
    for col in user_feature:
        user_feature[col] = pd.to_numeric(user_feature[col], downcast='integer')
        
    
    user_feature['userId'] = genre_count.index
    
    
    # re-arrange columns
    cols = user_feature.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    user_feature = user_feature[cols]
    
    # rename cols
    old_cols = user_feature.columns[1:]
    new_cols = []
    for idx, col in enumerate(cols[1:], 1):
        new_cols.append(f'u_{idx}')
    user_feature.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)
    
    return raw_transaction_list, user_feature

    


def preprocess_string(text):
    ''' Preprocess text for tf-idf
    
    Transforms the text into lowercase and removes symbols
    and punctuations
    Removes stopwords using NLTK library
    Lemmatizes words using SnowballStemmer (NLTK Library)
    
    Input
    --------
    text (string) :  string from the Movielens synopsis dataset 
    
    
    Output
    --------
    new_text (string)  : preprocessed text for further tf-idf processing
    
    '''    
    
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer(language='english')
    
    text = text.lower()
    
    text = ''.join([char for char in text if char not in string.punctuation])
    
    new_text = ""
    words = word_tokenize(text)
    for word in words:
        if word not in stop_words and len(word) > 2:
            new_text = new_text + " " + stemmer.stem(word)
    
    return new_text


# +
def create_item_feature(num_features = 300):
    '''
    Return item_feature matrix based on TF-IDF of Movie Synopsis
    
    Takes in the list of movies that has been rated in the MovieLens 100k
    dataset and fetches the respective synopsis for TF-IDF computation
    
       
    Input
    ---------
    num_features : number of features to be used for the TF-IDF extraction
                 : default value 300 (~sqrt[100k rows])
    
    
    Output
    ---------
    item_feature (pd.DataFrame): feature_vector from TF-IDF extracted
                            from movie synopses the TheMovieDB dataset
    
    
    
    '''
    
    transaction_list = pd.read_csv('sample_data/ratings.csv', usecols=['movieId'])
    
    # filter the unique movie IDs
    seen_movies = pd.DataFrame(transaction_list['movieId'].unique(), columns={'movieId'})
    
    # the synopsis is based on the "The Movie DB" Id system
    # links.csv has a mapping between MovieLens ID and The MovieDB Id
    movie_id_links = pd.read_csv('sample_data/links.csv', usecols =['movieId','tmdbId'])
    movie_id_links = movie_id_links.dropna()
    movie_id_links.head()
    
    # get mapping between MovieLens IDs and TMDB IDs
    seen_movies = seen_movies.merge(movie_id_links, on='movieId', how='inner')
    
    # Read MetaData CSV file with movie plots/synopsis
    metadata = pd.read_csv('sample_data/movies_metadata.csv', usecols=['id','overview'])
    metadata = metadata.rename(columns={'id':'tmdbId'})

    # drop movies with invalid tmbdId (e.g., date string instead of integer)
    ids1 = pd.to_numeric(metadata['tmdbId'], errors='coerce').isna()
    metadata = metadata.drop(metadata[ids1].index)

    # drop movies with NaN synopsis
    metadata = metadata.dropna()
    metadata['tmdbId'] = metadata['tmdbId'].astype(float)
    metadata = metadata.drop_duplicates(subset=['tmdbId'])

        
    # get only synopsis for movies in the transaction list
    synopsis_set = seen_movies.merge(metadata, on='tmdbId', how='inner')
    
    # preprocess synopsis strings
    synopsis_set['overview'] = synopsis_set['overview'].apply(preprocess_string)
    
    # TF-IDF processing
    tfidfvectorizer = TfidfVectorizer(analyzer='word', token_pattern = '[a-z]+\w*', stop_words='english', max_features=num_features)
    tfidf_vector = tfidfvectorizer.fit_transform(synopsis_set['overview'])
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=synopsis_set['movieId'], columns=tfidfvectorizer.get_feature_names_out())
    
    # normalization per column (word)
    tfidf_df = tfidf_df.apply(lambda x: (x - x.min())/(x.max() - x.min()))
    tfidf_df = tfidf_df.reset_index()
    
    # rename cols
    old_cols = tfidf_df.columns
    new_cols = []
    new_cols.append(old_cols[0])
    for idx, col in enumerate(old_cols[1:], 1):
        new_cols.append(f'i_{idx}')
    tfidf_df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)
    
    return tfidf_df


# -

def load_data(aug_tt, item_tt,user_tt):
    """
    Load the data from the transaction tables

    Paramters
    ---------
    aug_tt       : str
                   File name of the parquet file with each row corresponding
                   to a user's features, an item's features, and the user's
                   rating for that item

    item_tt      : str
                   File name of the parquet file with each row corresponding
                   to an item's features

    user_tt      : str
                   File name of the parquet file with each row corresponding
                   to a user's features

    Returns
    -------
    df            : pandas DataFrame
                    The augmented transaction table
                    
    item_df       : pandas DataFrame
                    The item features as a transaction table
                    
    user_df       : pandas DataFrame
                    The userfeatures as a transaction table
                    
    item_ids      : list
                    All unique item ids
                    
    user_ids      : list
                    All unique user ids
    """
    
    df = pd.read_parquet(aug_tt).dropna()
    item_df = pd.read_parquet(item_tt)
    item_ids = item_df['movieId'].unique()
    item_df = item_df.drop(columns=['movieId'])
    user_df = pd.read_parquet(user_tt).drop(columns=['userId'])
    user_ids = df['userId'].unique()
    return df, item_df, user_df, item_ids, user_ids



def get_augmented_table():
    '''
    Replace integrate user_features and item_features
    to the transaction_list
    
    Input
    ------
    none
    
    
    Output
    -------
    augmented_transaction_table  : transaction_list concatenated with user_features
                                   from genres and item_features from movie synopsis
    
    
    '''
    import pandas as pd
    transaction_list, user_feature = create_user_feature()
    item_feature = create_item_feature()
    augmented_tt = transaction_list.merge(user_feature, on='userId', how='left')
    augmented_tt_2 = augmented_tt.merge(item_feature, on='movieId', how='left')
    
    return augmented_tt_2
    