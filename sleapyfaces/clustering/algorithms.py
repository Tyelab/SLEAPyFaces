from math import sqrt
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class dataobjectprotocol(Protocol):
    all_data: pd.DataFrame
    cols: tuple[list[str], list[str]]


class Cluster:
    """BETA: Clustering algorithm class

    Args:
        data (dataobjectprotocol): data object
        prediction_column (str): the prediction class (from FeatureExtractor.classes)
        pipeline (bool, optional): whether to use a sklearn pipeline. Defaults to False.
    """
    def __init__(self, data: dataobjectprotocol, prediction_column: str):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        print("Clustering...")
        self._all_data = data.all_data
        self.qual_cols, self.quant_cols = data.cols

        self.pred = prediction_column
        self.predData = self._all_data.loc[:, ("Classes", self.pred)]

        self.clusterData = self._all_data.loc[:, self.quant_cols]
        self.clusterData = MinMaxScaler().fit_transform(self.clusterData)

        self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(
            self.clusterData, self._all_data.loc[:, ("Classes", self.pred)], test_size=0.3, random_state=12345
        )


    def kmeans(self, output: bool = False, gridSearch: bool = False, *args, **kwargs):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        print("\t KMeans Clustering...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 8

        kmeans = KMeans( *args, **kwargs)
        if gridSearch:
            from sklearn.model_selection import GridSearchCV
            params = dict(**self.params, **{
                'kmeans__n_clusters': range(2, 20),
                'kmeans__init': ['k-means++', 'random'],
                'kmeans__n_init': [10, 20, 30],
                'kmeans__max_iter': [300, 500, 1000],
                'kmeans__tol': [1e-4, 1e-3, 1e-2],
                'kmeans__random_state': [12345],
            })
            kmeans = GridSearchCV(kmeans, params, cv=5, n_jobs=-1, verbose=1)
        kmeans.fit(self.clusterData)
        labels = kmeans.predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["KMeans"]]))
        self.data = pd.concat([self.data, clusters.T], axis=1)
        if output:
            return kmeans

    def affinity_propagation(self, output: bool = False, gridSearch: bool = False, *args, **kwargs):
        from sklearn.cluster import AffinityPropagation
        from sklearn.metrics import silhouette_score

        print("\t Affinity Propagation Clustering...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 3

        ap = AffinityPropagation( *args, **kwargs)
        if gridSearch:
            from sklearn.model_selection import GridSearchCV
            params = dict(**self.params, **{
                'ap__damping': [0.5, 0.6, 0.7, 0.8, 0.9],
                'ap__max_iter': [200, 300, 400, 500],
                'ap__convergence_iter': [15, 20, 25, 30],
            })
            ap = GridSearchCV(ap, params, cv=5, n_jobs=-1, verbose=1)
        ap.fit(self.clusterData)
        labels = ap.predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["AffinityPropagation"]]))
        self.data = pd.concat([self.data, clusters.T], axis=1)
        if output:
            return ap

    def hierarchical(self, output: bool = False, *args, **kwargs):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        print("\t Hierarchical Clustering (Agglomerative)...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 2

        hc = AgglomerativeClustering( *args, **kwargs)
        hc.fit(self.clusterData)
        labels = None
        if hasattr(hc, "labels_"):
            labels = hc.labels_
        else:
            labels = hc.predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["HeirarchicalAgglomerative"]]))
        self.data = pd.concat([self.data, clusters.T], axis=1)
        if output:
            return hc

    def dbscan(self, output: bool = False, *args, **kwargs):
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score

        print("\t DBSCAN Clustering...")

        if "eps" in kwargs or len(args) != 0:
            if "eps" in kwargs:
                eps = kwargs["eps"]
            else:
                eps = args[0]
        else:
            eps = 3

        dbscan = DBSCAN( *args, **kwargs)
        if self.pipeline is not None:
            dbscan = Pipeline([*self.pipeline, ('dbscan', dbscan)])
        dbscan.fit(self.clusterData)
        labels = None
        if hasattr(dbscan, "labels_"):
            labels = dbscan.labels_
        else:
            labels = dbscan.predict(self.clusterData)
        print(f'Silhouette Score(eps={eps}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["Density-Based"]]))
        self.data = pd.concat([self.data, clusters.T], axis=1)
        if output:
            return dbscan

    def birch(self, output: bool = False, *args, **kwargs):
        from sklearn.cluster import Birch
        from sklearn.metrics import silhouette_score

        print("\t Birch Clustering...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 3

        bir = Birch(*args, **kwargs)
        if self.pipeline is not None:
            bir = Pipeline([*self.pipeline, ('bir', bir)])
        bir.fit(self.clusterData)
        labels: np.ndarray[any, np.intp] = bir.predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["Balanced"]]))
        self.data = pd.concat([self.data, clusters], axis=1)
        if output:
            return bir

    def OutlierSVM(self, output: bool = False,  *args, **kwargs):
        from sklearn.metrics import silhouette_score
        from sklearn.svm import OneClassSVM

        print("\t One Class SVM anomaly detection...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 3

        svm = OneClassSVM(*args, **kwargs)
        if self.pipeline is not None:
            svm = Pipeline([*self.pipeline, ('svm', svm)])
        svm = svm.fit(self.clusterData)
        labels = None
        if isinstance(svm, Pipeline):
            labels: np.ndarray[any, np.intp] = svm.predict(self.clusterData)
        elif isinstance(svm, OneClassSVM):
            labels: np.ndarray[any, np.intp] = svm.fit_predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["SVM"]]))
        self.data = pd.concat([self.data, clusters], axis=1)

        if output:
            return svm

    def knn(self, n_neighbors: int | list[int], output: bool = False):
        from sklearn.ensemble import BaggingRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsRegressor

        if isinstance(n_neighbors, int):
            knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn_model.fit(self.trainData, self.trainLabels)
            train_preds = knn_model.predict(self.trainData)
            mse = mean_squared_error(self.trainLabels, train_preds)
            rmse = sqrt(mse)
            print(f"Train RMSE: {rmse}")
            test_preds = knn_model.predict(self.testData)
            mse = mean_squared_error(self.testLabels, test_preds)
            rmse = sqrt(mse)
            print(f"Test RMSE: {rmse}")

        elif isinstance(n_neighbors, list):

            params = {"n_neighbors": n_neighbors,
                      "weights": ["uniform", "distance"],}
            knn_model = KNeighborsRegressor()
            knn_cv = GridSearchCV(knn_model, params, cv=5)
            knn_cv.fit(self.trainData, self.trainLabels)
            print(f"Best Parameters: {knn_cv.best_params_}")
            train_preds = knn_cv.predict(self.trainData)
            mse = mean_squared_error(self.trainLabels, train_preds)
            rmse = sqrt(mse)
            print(f"Train RMSE: {rmse}")
            test_preds = knn_model.predict(self.testData)
            mse = mean_squared_error(self.testLabels, test_preds)
            rmse = sqrt(mse)
            print(f"Test RMSE: {rmse}")
            knn_model = KNeighborsRegressor(n_neighbors=knn_cv.best_params_["n_neighbors"], weights=knn_cv.best_params_["weights"])

        bagged_knn = BaggingRegressor(knn_model, n_estimators=10, random_state=12345)
        bagged_knn.fit(self.trainData, self.trainLabels)
        train_preds = bagged_knn.predict(self.trainData)
        mse = mean_squared_error(self.trainLabels, train_preds)
        rmse = sqrt(mse)
        print(f"Train RMSE: {rmse}")
        test_preds = bagged_knn.predict(self.testData)
        mse = mean_squared_error(self.testLabels, test_preds)
        rmse = sqrt(mse)
        print(f"Test RMSE: {rmse}")

        if output:
            return knn_model

    @property
    def all_data(self):
        return self.clusterData.merge(self._all_data[self.qual_cols])

    @property
    def cols(self):
        return self.qual_cols, [col for col in self.clusterData.columns if col not in self.qual_cols]
