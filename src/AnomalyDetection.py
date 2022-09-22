from utils.utils import train_test_split_dataset ,calculate_cov_matrix, cal_md_distance
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

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


# config file
config = {
    'learning_rate': 0.001,
    'dense_layers':[
          {'enc_neurons': 64, 'activation': 'relu'},
          {'code_size':4, 'activation': 'relu'},
          {'dec_neurons': 16, 'activation': 'relu'},
        ],
    "outputs": 1,
    'output_activation':'sigmoid',
    'dropout_rate': 0.08,
    'batch_size': 128,
}

class AutoEncoder(tf.keras.Model):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(config["dense_layers"][0]["enc_neurons"], activation=config["dense_layers"][0]["activation"]),
            tf.keras.layers.Dense(config["dense_layers"][0]["enc_neurons"]/2, activation=config["dense_layers"][0]["activation"]),
            tf.keras.layers.Dense(config["dense_layers"][0]["enc_neurons"]/4, activation=config["dense_layers"][0]["activation"]),
            tf.keras.layers.Dense(config["dense_layers"][1]["code_size"], activation=config["dense_layers"][1]["activation"])
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(config["dense_layers"][2]["dec_neurons"], activation=config["dense_layers"][2]["activation"]),
            tf.keras.layers.Dense(config["dense_layers"][2]["dec_neurons"]*2, activation=config["dense_layers"][2]["activation"]),
            tf.keras.layers.Dense(config["dense_layers"][2]["dec_neurons"]*2, activation=config["dense_layers"][2]["activation"]),
            tf.keras.layers.Dense(1, activation=config["output_activation"])
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

        # A convenient way to get model summary and plot in subclassed api

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def find_threshold(self, x_train_scaled):
        reconstructions = self.predict(x_train_scaled)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)

        # threshold for anomaly scores
        threshold = np.mean(reconstruction_errors.numpy())  + (3 * np.std(reconstruction_errors.numpy()))
        return threshold

    def get_predictions(self, x_test_scaled, threshold):
        predictions = self.predict(x_test_scaled)
        # provides losses of individual instances
        errors = tf.keras.losses.msle(predictions, x_test_scaled)
        # 0 = anomaly, 1 = normal
        preds = pd.Series(errors).apply(lambda x: 1 if x > threshold else 0)

        return preds


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

    y_hat = model.predict(df_train)
    print(y_hat)
    # df_train["md"] = md
    # print(df_train.head(), "\nCalculated Threshold = ", model.threshold)

    y_hat = model.predict(df_test)
    print("\nCluster analysis score: ",model.score(df_test ,y_hat))
    print(y_hat)

    # Autoencoder model
    # min max scale the input data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = min_max_scaler.fit_transform(df_train.copy())
    x_test_scaled = min_max_scaler.transform(df_test.copy())

    nn_model = AutoEncoder(config)
    nn_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

    history = nn_model.fit(
                            x_train_scaled,
                            x_train_scaled,
                            epochs=100,
                            batch_size=512,
                            validation_data=(x_test_scaled, x_test_scaled)
                          )

    # Geting model structure
    nn_model.build_graph(x_train_scaled.shape).summary()
    print(nn_model.summary())

    # Model Evaluation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSLE Loss')
    plt.legend(['loss', 'val_loss'])
    plt.show()


    print("Calculating threshold")
    threshold = nn_model.find_threshold(x_train_scaled)
    print(f"Threshold method one: {threshold}")

    print("Getting Prediction on test data")
    y_hat = nn_model.get_predictions(x_test_scaled, threshold)
    print(y_hat)