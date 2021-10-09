![logo](resype_logo.png)




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

## Setup

The package uses Python 3 and `virtualenv` to manage the environment.  Once you've cloned this repo run this from the root of the repo to create the environment:

```
conda env create -f environment.yml
source activate resype
```

## Getting Started

```python
import pandas as pd
import numpy as np
from resype.collab_filtering import CollabFilteringModel

# load transaction list
transaction_list = pd.read_csv("sample_data/ratings.csv")
transaction_list = transaction_list[['userId', 'movieId', 'rating']]
transaction_list = transaction_list.sample(20)
transaction_list.columns = ["user_id", 'item_id', 'rating']

re = CollabFilteringModel(transaction_list)

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
```

## Comparison with Surprise

| Features                     | Surprise | ReSyPE |
|:-----------------------------|:--------:|:------:|
| Handles explicit rating data | ✔️        | ✔️      |
| Cross validation             | ✔️        | ✔️      |
| Recommendation evaluation    | ✔️        | ✔️      |
| Collaborative filtering      | ✔️        | ✔️      |
| Content-based filtering      |          | ✔️      |
| Customizable ML models       |          | ✔️      |

### Performance

| Prediction Algorithm   | MSE      | MAE      |
|:-----------------------|:--------:|:--------:|
| NormalPredictor        | 2.051080 | 1.135742 |
| BaselineOnly           | 0.858667 | 0.735921 |
| KNNBasic               | 1.362782 | 0.906558 |
| KNNWithMeans           | 1.173480 | 0.850230 |
| KNNWithZScore          | 1.185011 | 0.842193 |
| KNNBaseline            | 1.057957 | 0.796983 |
| SVD                    | 0.862225 | 0.730675 |
| NMF                    | 1.360950 | 0.921752 |
| SlopeOne               | 1.222082 | 0.869121 |
| CoClustering           | 1.299210 | 0.900984 |
| ReSyPE (SVD-based Collaborative Filtering using Random Forest) | 1.343327 | 0.907147 |
| ReSyPE (Content-based; Random Forest) | 0.955075 | 0.757576 |
