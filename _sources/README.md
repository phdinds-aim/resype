<div align="center">
<img alt="Logo" src="resype_logo.png" width="60%"/>
</div>

# ReSyPE: Recommender System in Python Environment

## Introduction
ReSyPE (pronounced *recipe*) is a Python library built for both practitioners and researchers that wish to quickly develop and deploy ML-based recommender systems.

The library provides an end-to-end pipeline that includes:

1. Loading transaction, user feature, and item feature datasets
2. Interchangable methods for user and item clustering
3. Modular framework for machine learning models
4. Iterative and decomposition-based techniques

## Installation

`pip install resype`

## Getting Started

    import pandas as pd
    import numpy as np
    from resype.resype import Resype
    
    # load transaction list
    transaction_list = pd.read_csv("sample_data/ratings.csv")[['userId', 'movieId', 'rating']]
    transaction_list = transaction_list.sample(20)
    transaction_list.columns = ["user_id", 'item_id', 'rating']
    
    re = Resype(transaction_list)
    
    # construct utlity matrix
    re.construct_utility_matrix()
    
    # import sklearn Model
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    mlp1 = MLPRegressor(hidden_layer_sizes=(50, 50))
    
    # fit and predict
    re.fit(mlp1, method='iterative')
    
    # recommend
    user_list = [0, 1, 2] # indices
    top_n = 10
    re.get_rec(user_list, top_n)
    
    