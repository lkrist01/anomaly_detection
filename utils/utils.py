import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def normilize_dataset(train, test):

    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)

    return X_train, X_test

def train_test_split_dataset(X, y, type="timeseries", test_size=0.15):

    if type == "timeseries":
        train_len = int(len(X) * (1-test_size))

        X_train, y_train = X[:train_len] , y[:train_len]
        X_test, y_test = X[train_len:] , y[train_len:]

    elif type == "random":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def pca_transform(train_data, test_data, n_components = 2, solver= "full"):
    pca = PCA(n_components=n_components, svd_solver=solver)

    train_pca = pca.fit_transform(train_data)
    test_pca = pca.transform(test_data)

    return train_pca, test_pca

def is_pd(Cov):
    try:
        np.linalg.cholesky(Cov)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0
        else:
            raise

def calculate_cov_matrix(data):
    cov_matrix = np.cov(data, rowvar=False)

    #Check if cov is positive definite
    if is_pd(cov_matrix):
        inv_cov_m = np.linalg.inv(cov_matrix)
        if is_pd(inv_cov_m):
            return cov_matrix, inv_cov_m
        else:
            print("Inverse Cov matrix is not positive definite")
    else:
        print("Cov matrix is not positive definite")
    return cov_matrix, inv_cov_m

def cal_md_distance(X, inv_conv_m, mean_d):
    x_mu = X - mean_d

    left = np.dot(x_mu, inv_conv_m)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()
