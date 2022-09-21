from utils.utils import train_test_split_dataset ,calculate_cov_matrix, cal_md_distance
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing


class AnomalyDetection():

    def __init__(self, std_dev=None, normalize=False, pca=False, extreme=False, thresh=None, inv_cov= None, distrib=None):
        '''

        :param std_dev:
        :param normalize:
        :param pca:
        :param extreme:
        :param thresh:
        :param inv_cov:
        :param distrib:
        '''

        self.std_dev = std_dev
        self.extreme = extreme

        # Scaler and pca params
        self.scaler = normalize
        self.pca = pca

        # Cluster Snapshot Params
        self.threshold = thresh
        self.inv_cov = inv_cov
        self.mean_dist = distrib

    def fit(self, X):
        '''

        :param X: Pandas Dataframe
        :return:
        '''

        #Checking for any preprocessing
        if self.pca:
            # only keep the first 2 principal components
            self.pca = PCA(n_components=2, svd_solver='full')
            X = self.pca.fit_transform(X)

        if self.scaler:
            self.scaler = preprocessing.MinMaxScaler()
            X = self.scaler.fit_transform(X)


        # Calculate Covariances and update params
        cov_m, inv_cov_m = calculate_cov_matrix(X)
        self.mean_dist = X.mean(axis=0)
        self.inv_cov = inv_cov_m

        # Calculate Mahalanobis distance
        mahal = cal_md_distance(X, self.inv_cov, self.mean_dist)

        #Specify type of threshold
        if self.threshold:
            k = self.threshold
        else:
            k = 3 if self.extreme else 2

        #update cluster params
        self.threshold = (np.std(mahal) * k) + np.mean(mahal)

        return mahal

    def predict(self, X):
        '''

        :param X: Pandas Dataframe
        :return:
        '''

        if self.pca:
            X = self.pca.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)

        md_dist = cal_md_distance(X, self.inv_cov, self.mean_dist)
        outliers = []

        for i in range(len(md_dist)):
            if md_dist[i] >= self.threshold:  #check if outside threshhold
                outliers.append(1)
            else:
                outliers.append(0)
        return np.array(outliers)

    def prep_process_data(self):
        '''

        :return:
        '''

        return None

    def score(self, X, y):
        '''

        :param X:
        :param y:
        :return:
        '''

        return None

if __name__ == '__main__':
    data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
            'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
            'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
            'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
            }

    data_test = {'score': [91, 100, 84, 91],
            'hours': [2, 1, 4, 16],
            'prep': [0, 0, 3, 4],
            'grade': [88, 88, 90, 60]
            }

    df = pd.DataFrame(data, columns=['score', 'hours', 'prep', 'grade'])
    df_test = pd.DataFrame(data_test, columns=['score', 'hours', 'prep', 'grade'])


    model = AnomalyDetection(pca=True, normalize=True)
    md = model.fit(df)

    df["anomaly"]= model.predict(df)
    df["md"] = md

    print(df.head(), "\n Threshold = ", model.threshold)

    df_test["anomaly"] = model.predict(df_test)
    print(df_test)