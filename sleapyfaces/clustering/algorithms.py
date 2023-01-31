from math import sqrt
from typing import Protocol
import logging

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
    def __init__(self, data: dataobjectprotocol, prediction_column: str, parallel_processing: bool = True, *args, **kwargs):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder

        self.clusters = pd.DataFrame()
        self.silhouette = pd.DataFrame()
        self._all_data = data.all_data
        self.qual_cols, self.quant_cols = data.cols

        self.pred = prediction_column
        self.predData: pd.Series = self._all_data.loc[:, ("Classes", self.pred)]
        if np.can_cast(self.predData.dtype, np.number):
            self.predData = self.predData.astype(np.number)
        else:
            self.predData = LabelEncoder().fit_transform(self.predData)
            self._all_data = self._all_data.join(pd.DataFrame(self.predData, columns=pd.MultiIndex.from_product([["Classes"], [f"{self.pred}_encoded"]])))
            self.qual_cols.append(("Classes", f"{self.pred}_encoded"))

        self.clusterData = self._all_data.loc[:, self.quant_cols]
        self.clusterData = MinMaxScaler().fit_transform(self.clusterData)

        self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(
            self.clusterData, self._all_data.loc[:, ("Classes", self.pred)], test_size=0.3, random_state=12345
        )

        self.parallel = None
        if parallel_processing:
            from joblib import Parallel
            self.parallel = Parallel(n_jobs=-1, verbose=1, prefer="processes", max_nbytes='5M' if "max_nbytes" not in kwargs else kwargs["max_nbytes"])

    def _score(self, model, X, y, z, cross_validate: bool = False):
        from sklearn.metrics import silhouette_score
        from sklearn.model_selection import cross_validate, cross_val_score
        if cross_validate:
            if self.parallel is not None:
                with self.parallel:
                    scores = cross_validate(model, X, y, cv=5, n_jobs=-1, verbose=4)
            else:
                scores = cross_validate(model, X, y, cv=5, n_jobs=-1, verbose=4)
        else:
            if self.parallel is not None:
                with self.parallel:
                    scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, verbose=4)
            else:
                scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, verbose=4)
        logging.info(f'Cross Validation Scores: {scores}')
        score = silhouette_score(X, z)
        logging.info(f'Silhouette Score: {score}')
        return scores, score

    def kmeans(self, output: bool = False, gridSearch: bool = False, cross_validate: bool = False, *args, **kwargs):
        from sklearn.cluster import KMeans

        print("\t KMeans Clustering...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 8

        if cross_validate or gridSearch:
            X_train = self.clusterData
            y_train = self.predData
            X_test = self.clusterData
            y_test = self.predData
            cross_validate = False if gridSearch else True
        else:
            X_train = self.trainData
            y_train = self.trainLabels
            X_test = self.testData
            y_test = self.testLabels

        kmeans = KMeans( *args, **kwargs)
        if gridSearch:
            from sklearn.model_selection import GridSearchCV

            params = {
                'n_clusters': range(2, 20),
                'init': ['k-means++', 'random'],
                'n_init': [10, 20, 30],
                'max_iter': [300, 500, 1000],
                'tol': [1e-4, 1e-3, 1e-2],
                'random_state': [12345],
            }
            if self.parallel is not None:
                with self.parallel:
                    kmeans = GridSearchCV(kmeans, params, cv=5, n_jobs=-1, verbose=4)
                    kmeans.fit(X_train, y_train)
            else:
                kmeans = GridSearchCV(kmeans, params, cv=5, n_jobs=-1, verbose=4)
                kmeans.fit(X_train, y_train)
        if cross_validate or gridSearch:
            X_train = self.clusterData
            y_train = self.predData
            X_test = self.clusterData
            y_test = self.predData
            cross_validate = False if gridSearch else True
        else:
            X_train = self.trainData
            y_train = self.trainLabels
            X_test = self.testData
            y_test = self.testLabels

        labels = kmeans.predict(X_test)
        scores, score = self._score(kmeans, X_test, y_test, labels, cross_validate)
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["KMeans"]]))
        self.clusters = pd.concat([self.clusters, clusters.T], axis=1)
        self.silhouette = pd.concat([self.silhouette, pd.DataFrame([score, scores.values()], columns=["KMeans"])], axis=1)
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
        self.clusters = pd.concat([self.clusters, clusters.T], axis=1)
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
        self.clusters = pd.concat([self.clusters, clusters.T], axis=1)
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
        self.clusters = pd.concat([self.clusters, clusters.T], axis=1)
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
        self.clusters = pd.concat([self.clusters, clusters], axis=1)
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
        self.clusters = pd.concat([self.clusters, clusters], axis=1)

        if output:
            return svm

    def SVM(self, output: bool = False, *args, **kwargs):
        from sklearn.metrics import silhouette_score
        from sklearn.svm import SVC

        print("\t SVM...")
        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 3

        svm = SVC(*args, **kwargs)
        if self.pipeline is not None:
            svm = Pipeline([*self.pipeline, ('svm', svm)])
        svm = svm.fit(self.clusterData, self.clusterLabels)
        labels = None
        if isinstance(svm, Pipeline):
            labels: np.ndarray[any, np.intp] = svm.predict(self.clusterData)
        elif isinstance(svm, SVC):
            labels: np.ndarray[any, np.intp] = svm.fit_predict(self.clusterData)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.clusterData, labels)}')
        clusters: pd.DataFrame[np.intp] = pd.DataFrame(labels.T, dtype=np.intp, columns=pd.MultiIndex.from_product([["Clustering"], ["SVM"]]))
        self.clusters = pd.concat([self.clusters, clusters], axis=1)

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
