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

    import resype
    from sklearn.cluster import KMeans
    from sklearn.neural_network import MLPRegressor
    
    re = resype(transaction_list, user_features, item_features)
    km = KMeans()
    re.cluster_fit(user_model=km, item_model=None, user_n=20, item_n=None, agg_func='sum')
    ml = MLPRegressor()
    re.fit(model=ml, method='iterative') 

    df_rec = re.get_rec(top_k=10, user_list = [1, 2, 3])