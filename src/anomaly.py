from utils.utils import (
    normilize_dataset,
    train_test_split_dataset,
    pca_transform,
    calculate_cov_matrix,
    cal_md_distance,
)
import pandas as pd
import numpy as np


class AnomalyDetection:
    def __init__(
        self,
        std_dev=None,
        normalize=False,
        pca=False,
        extreme=False,
        thresh=None,
        inv_cov=None,
        distrib=None,
    ):
        self.std_dev = std_dev
        self.normalize = normalize
        self.pca = pca
        self.extreme = extreme

        # stats for the cluster
        self.threshold = thresh
        self.inv_cov = inv_cov
        self.mean_dist = distrib

    def fit(self, X):
        cov_m, inv_cov_m = calculate_cov_matrix(X)

        self.mean_dist = X.mean(axis=0)
        self.inv_cov = inv_cov_m

        mahal = cal_md_distance(X, self.inv_cov, self.mean_dist)

        if self.threshold:
            k = self.threshold
        else:
            k = 3 if self.extreme else 2

        # update cluster params
        self.threshold = (np.std(mahal) * k) + np.mean(mahal)

        return mahal

    def predict_outliers(self, test_data):

        md_dist = cal_md_distance(test_data, self.inv_cov, self.mean_dist)
        outliers = []

        for i in range(len(md_dist)):
            if md_dist[i] >= self.threshold:
                outliers.append(1)
            else:
                outliers.append(0)
        return np.array(outliers)

    def score(self, X, y):
        return None


if __name__ == "__main__":
    data = {
        "score": [
            91,
            93,
            72,
            87,
            86,
            73,
            68,
            87,
            78,
            99,
            95,
            76,
            84,
            96,
            76,
            80,
            83,
            84,
            73,
            74,
        ],
        "hours": [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
        "prep": [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        "grade": [
            70,
            88,
            80,
            83,
            88,
            84,
            78,
            94,
            90,
            93,
            89,
            82,
            95,
            94,
            81,
            93,
            93,
            90,
            89,
            89,
        ],
    }

    data_test = {
        "score": [91, 100],
        "hours": [
            2,
            1,
        ],
        "prep": [
            0,
            0,
        ],
        "grade": [88, 88],
    }
    df = pd.DataFrame(data, columns=["score", "hours", "prep", "grade"])
    df_test = pd.DataFrame(data_test, columns=["score", "hours", "prep", "grade"])

    model = AnomalyDetection()
    md = model.fit(df)

    df["anomaly"] = model.predict_outliers(df)
    df["md"] = md

    print(df.head(), "\n Threshold = ", model.threshold)

    df_test["anomaly"] = model.predict_outliers(df_test)
    print(df_test)
