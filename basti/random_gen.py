import numpy as np
import pandas as pd

def random_user_list(n_user=100, sample_size=10, random_seed=1):
    
    """Generates random user-to-cluster assignment.
    
            Parameters:
                    n_user (int): Number of users
                    sample_size (int): Number of user to sample

            Returns:
                    user_list (numpy.ndarray): List of users to recommend to
    """
    
    np.random.seed(random_seed)
    user_list = np.random.choice(range(n_user), size=sample_size, replace=False)

    return user_list

def random_user_cluster(n_user=100, n_user_cluster=5, random_seed=1):
    
    """Generates random user-to-cluster assignment.
    
            Parameters:
                    n_user (int): Number of users
                    n_user_cluster (int): Number of user clusters

            Returns:
                    uc_assignment (numpy.ndarray): List of cluster assignments
    """
    
    np.random.seed(random_seed)
    uc_assignment = np.random.randint(low=0, high=n_user_cluster, size=n_user)
    
    return uc_assignment

def random_utility_matrix(n_user_cluster=5, n_item_cluster=5, random_seed=1):

    """Generates a random imputed utility matrix.
    
            Parameters:
                    n_user (int): Number of users
                    n_item (int): Number of users
                    n_user_cluster (int): Number of user clusters
                    n_item_cluster (int): Number of item clusters
                    random_seed (int): Random seed

            Returns:
                    utility_matrix_o (numpy.ndarray): A random utility matrix before imputation
                    utility_matrix (numpy.ndarray): A random utility matrix after imputation            
    """

    user_cluster_list = list(range(n_user_cluster))
    item_cluster_list = list(range(n_item_cluster))
    
    # Generate random utility matrix
    np.random.seed(random_seed)
    utility_matrix = np.eye(N=len(user_cluster_list), M=len(item_cluster_list))
    np.random.shuffle(utility_matrix)
    
    utility_matrix_o = utility_matrix.copy() # Assume that 1 indicates that it has been rated, everything else is imputed
    
    utility_matrix += np.random.beta(a=1, b=1, size=(len(user_cluster_list), len(item_cluster_list))).round(4)
    utility_matrix[utility_matrix > 1] = 1
    
    return utility_matrix_o, utility_matrix