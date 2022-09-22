from utils.utils import train_test_split_dataset ,calculate_cov_matrix, cal_md_distance
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics

class MahalanobisDistance():

    def __init__(self, std_dev=None, normalize=False, pca=False, extreme=False, thresh=None, inv_cov= None, distrib=None):
        '''
        Class Constructor Initializer
        :param std_dev: coefficient multiplying for threshold from std to detect anomaly (Example 3sd)
        :param normalize: Flag for doing Scaling on data
        :param pca: Flag for doing pca on data
        :param extreme: Flag for automatic coefficient of std
        :param thresh: Parameter for storing cluster MD Threshold
        :param inv_cov: Parameter for storing cluster inverse covariance
        :param distrib: Parameter for storing cluster mean distribution for each feature
        '''

        self.std_dev = std_dev
        self.extreme = extreme

        # Scaler and pca params
        if pca:
            self.pca = PCA(n_components=2, svd_solver='full')
        else:
            self.pca = None

        if normalize:
            self.scaler = preprocessing.MinMaxScaler()
        else:
            self.scaler = None

        # Cluster Snapshot Params
        self.threshold = thresh
        self.inv_cov = inv_cov
        self.mean_dist = distrib

    def fit(self, X):
        '''
        Function for getting Cluster MD threshold
        :param X: Pandas Dataframe
        :return: Mahalanobis distances for each data row
        '''

        #Checking for any preprocessing
        if self.pca:
            # only keep the first 2 principal components
            X = self.pca.fit_transform(X)

        if self.scaler:
            X = self.scaler.fit_transform(X)

        # Calculate Covariances and update params
        cov_m, inv_cov_m = calculate_cov_matrix(X)
        self.mean_dist = X.mean(axis=0)
        self.inv_cov = inv_cov_m

        # Calculate Mahalanobis distance
        mahal = cal_md_distance(X, self.inv_cov, self.mean_dist)

        #Specify type of threshold
        if self.std_dev:
            k = self.std_dev
        else:
            k = 3 if self.extreme else 2

        #update cluster params
        self.threshold = (np.std(mahal) * k) + np.mean(mahal)

        return mahal

    def predict(self, X):
        '''
        This function is used to return the prediction of a given dataset
        :param X: Pandas Dataframe
        :return: np.array class prediction (Anomaly=1 , Normal=0)
        '''

        if self.pca:
            X = self.pca.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)

        md_dist = cal_md_distance(X, self.inv_cov, self.mean_dist)
        outliers = []

        for i in range(len(md_dist)):
            if md_dist[i] >= self.threshold:  #check if outside threshold
                outliers.append(1)
            else:
                outliers.append(0)
        return np.array(outliers)

    def score(self, X, y, type="silhouette"):
        '''
        Function for cluster analysis for evaluation
        :param X: Pandas dataset used on testing
        :param y: 1D Array of predicted class
        :return: Cluster evaluation score
        '''
        if type == "davies":
            return metrics.davies_bouldin_score(X, y)
        elif type == "calinski":
            return metrics.calinski_harabasz_score(X, y)
        elif type == "silhouette":
            return metrics.silhouette_score(X, y, metric='euclidean')
        else:
            print("Please choose between ['davies', 'calinski','silhouette']")
            return
        # return contingency_matrix(X, y)


class AutoEncoder():
    def __init__(self, input_dim, num_layers):
        self.encoder = None

        self.decoder = None

    def fit(self, X):

        return self

    def predict(self, X):
        return None

if __name__ == '__main__':
    data_train = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
            'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
            'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
            'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
            }

    data_test = {'score': [91, 100, 84, 91],
            'hours': [2, 1, 4, 16],
            'prep': [0, 0, 3, 4],
            'grade': [88, 88, 90, 60]
            }

    df_train = pd.DataFrame(data_train, columns=['score', 'hours', 'prep', 'grade'])
    df_test = pd.DataFrame(data_test, columns=['score', 'hours', 'prep', 'grade'])


    model = MahalanobisDistance(pca=True, normalize=True)
    md = model.fit(df_train)

    df_train["anomaly"]= model.predict(df_train)
    df_train["md"] = md

    print(df_train.head(), "\nCalculated Threshold = ", model.threshold)

    y_hat = model.predict(df_test)
    print("\nCluster analysis score: ",model.score(df_test ,y_hat))
    df_test["anomaly"] = y_hat
    print(df_test)