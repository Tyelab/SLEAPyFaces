from typing import Protocol
import pandas as pd
from dataclasses import dataclass
import numpy as np
from math import sqrt
from sklearn.pipeline import Pipeline

class dataobjectprotocol(Protocol):
    data: pd.DataFrame
    scores: pd.DataFrame
    quant_cols: list[str]
    qual_cols: list[str]
    cols: tuple[list[str], list[str]]

def polar(x: float, y: float) -> tuple[float, float]:
    """Converts cartesian coordinates to polar coordinates.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.

    Returns:
        tuple[float, float]: The polar coordinates.
    """
    from math import atan2, sqrt

    r = sqrt(x ** 2 + y ** 2)
    theta = atan2(y, x)
    return r, theta

def cartesian_to_polar(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Converts cartesian pandas dataframe columns to polar coordinates.

    Args:
        df (pd.DataFrame): The dataframe to convert to polar coordinates.
        cols (list[str]): The columns to convert.

    Returns:
        df (pd.DataFrame): The dataframe with the polar columns appended as "{...}_r" and "{...}_theta".
    """
    df = data.copy()
    index = []
    new_index = False
    if type(cols[0]) is tuple:
        new_cols = []
        for col in cols:
            new_cols.append(f"{col[0]}_{col[1]}")
        cols = new_cols
        df.columns = cols
        new_index = True

    for i in range(0, len(cols), 2):
        df[cols[i].replace("_x", "_r")], df[cols[i+1].replace("_y", "_theta")] = zip(
            *df.apply(
                lambda column: polar(
                    column[cols[i]], column[cols[i + 1]]
                ),
                axis=1
            )
        )
        if new_index:
            index.append((cols[i].replace("_x", ""), "r"))
            index.append((cols[i+1].replace("_y", ""), "theta"))

    df.drop(columns=cols, inplace=True)
    if new_index:
        df.columns = pd.MultiIndex.from_tuples(index)
    return df

def extractName(cols: list[any]) -> str:
    if type(cols) is not str:
        cols = extractName(cols[0])
    return cols

def euclidean_distance(df1: pd.DataFrame, df2: pd.DataFrame, multiindex: bool = False) -> pd.DataFrame:
    """Calculates the euclidean distance between two dataframes.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        pd.DataFrame: The euclidean distance between the two dataframes.
    """
    point1 = extractName(df1.columns.to_list()).replace('_x', '').replace('_y', '').replace('_r', '').replace('_theta', '')
    point2 = extractName(df2.columns.to_list()).replace('_x', '').replace('_y', '').replace('_r', '').replace('_theta', '')
    x = np.subtract(df1.filter(regex="_x").to_numpy(), df2.filter(regex="_x").to_numpy())
    y = np.subtract(df1.filter(regex="_y").to_numpy(), df2.filter(regex="_y").to_numpy())
    x = np.real(np.square(x))
    y = np.real(np.square(y))
    df = pd.DataFrame(np.squeeze(np.sqrt(x + y)).T)
    df.columns = [f"distance({point1}->{point2})"] if not multiindex else pd.MultiIndex.from_tuples([(f"{point1}->{point2}", "euclidean_distance")])
    return df


@dataclass(slots=True)
class FeatureExtractor:

    data: pd.DataFrame
    cartesian: pd.DataFrame
    polar: pd.DataFrame
    scores: pd.DataFrame
    calcData: pd.DataFrame
    points: set[str]
    cols: list[str]
    cartesianCols: list[str]
    polarCols: list[str]
    classes: list[str]
    basefeats: list[str]
    calcsLog: list[str]

    def __init__(self, dataObject: dataobjectprotocol, base_features: list[str]):

        self.basefeats: list[str] = base_features

        self.cartesian: pd.DataFrame = dataObject.data
        self.cartesianCols, self.classes = dataObject.cols

        self.polar: pd.DataFrame = cartesian_to_polar(self.cartesian, self.cartesianCols)
        self.polarCols: list[str] = [col.replace("_x", "_r").replace("_y", "_theta") for col in self.cartesianCols]

        self.cols: list[str] = self.cartesianCols + self.polarCols
        self.points: set[str] = set([point.replace("_x", "").replace("_y", "") for point in self.cartesianCols])

        all_data: pd.DataFrame = pd.concat([self.cartesian, self.polar], axis=1)
        all_data.reset_index(inplace=True)
        num_data: list[pd.DataFrame] = [pd.DataFrame({'x': np.squeeze(all_data.loc[:, self.cols].filter(like=point).filter(like="_x").values), 'y': np.squeeze(all_data.loc[:, self.cols].filter(like=point).filter(like="_y").values), 'r': np.squeeze(all_data.loc[:, self.cols].filter(like=point).filter(like="_r").values), 'theta': np.squeeze(all_data.loc[:, self.cols].filter(like=point).filter(like="_theta").values).T}, columns=["x", "y", "r", "theta"]) for point in self.points]
        num_data: pd.DataFrame = pd.concat(num_data, axis=1, keys=self.points)
        qual_data: pd.DataFrame = all_data.loc[:, self.classes]
        qual_data: pd.DataFrame = qual_data.loc[:,~qual_data.columns.duplicated()].copy()
        qual_data.columns = pd.MultiIndex.from_product([["Classes"], self.classes])
        self.data = pd.concat([qual_data, num_data], axis=1)

        self.cartesian.reset_index(inplace=True)
        self.cartesian.drop(columns=self.classes, inplace=True)
        self.polar.reset_index(inplace=True)
        self.polar.drop(columns=self.classes, inplace=True)
        self.calcData = qual_data

        self.calcsLog = []
        self.scores = dataObject.scores

    def extractCentroids(self, inplace: bool = False, topolar: bool = False) -> pd.DataFrame | None:
        if self.basefeats is None:
            raise ValueError("No base features have been defined.")
        self.calcsLog.append(f"extractCentroids(inplace={inplace}, topolar={topolar})")
        df_coords = []
        for feature in self.basefeats:
            for coord in ["x", "y"]:
                df_coord: pd.Series = self.cartesian.loc[:, self.cartesianCols].filter(like=feature).filter(like=coord).apply(np.mean, axis=1)
                df_coords.append(df_coord.rename((feature, coord)))
        centroids: pd.DataFrame = pd.concat(df_coords, axis=1)
        centroid_cols: list[str] = centroids.columns.to_list()
        if topolar:
            centroids = cartesian_to_polar(centroids, centroid_cols)
        self.calcData = pd.concat([self.calcData, centroids], axis=1)
        return centroids if not inplace else None

    def extractCentroid(self, feature: str, inplace: bool = False, topolar: bool = False, multiindex: bool = True) -> pd.DataFrame | None:
        if feature not in self.basefeats:
            raise ValueError(f"Centroid {feature} not found")
        self.calcsLog.append(f"extractCentroid(feature={feature}, inplace={inplace}, topolar={topolar})")
        df_coords = []
        for coord in ["x", "y"]:
            df_coord: pd.Series = self.cartesian.loc[:, self.cartesianCols].filter(like=feature).filter(like=coord).apply(np.mean, axis=1)
            if multiindex:
                df_coords.append(df_coord.rename((feature, coord)))
            else:
                df_coords.append(df_coord.rename(f"{feature}_{coord}"))
        centroid: pd.DataFrame = pd.concat(df_coords, axis=1)
        centroid_col: list = centroid.columns.to_list()
        if topolar:
            centroid = cartesian_to_polar(centroid, centroid_col)
        return centroid if not inplace else None

    def twoPointsDist(self, pointa: str, pointb: str, inplace: bool = False) -> pd.DataFrame | None:
        if pointa not in self.points or pointb not in self.points:
            raise ValueError(f"Point {pointa} or {pointb} not found")
        self.calcsLog.append(f"twoPointsDist(pointa={pointa}, pointb={pointb}, inplace={inplace})")
        distance = euclidean_distance(self.cartesian.loc[:, [f"{pointa}_x", f"{pointa}_y"]], self.cartesian.loc[:, [f"{pointb}_x", f"{pointb}_y"]], multiindex=True)
        self.calcData = pd.concat([self.calcData, distance], axis=1)
        return distance if not inplace else None

    def distToCentroid(self, point: str, centroid: str, inplace: bool = False) -> pd.DataFrame | None:
        if centroid not in self.basefeats:
            raise ValueError(f"Centroid {centroid} not found")
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(f"distToCentroid(point={point}, centroid={centroid}, inplace={inplace})")
        centroid = self.extractCentroid(centroid, multiindex=False)
        distance = euclidean_distance(self.cartesian.loc[:, [f"{point}_x", f"{point}_y"]], centroid, multiindex=True)
        self.calcData = pd.concat([self.calcData, distance], axis=1)
        return distance if not inplace else None

    def twoCentroidsDist(self, centroida: str, centroidb: str, inplace: bool = False) -> pd.DataFrame | None:
        if centroida not in self.basefeats:
            raise ValueError(f"Centroid 1 {centroida} not found")
        if centroidb not in self.basefeats:
            raise ValueError(f"Centroid 2 {centroidb} not found")
        self.calcsLog.append(f"twoCentroidsDist(centroida={centroida}, centroidb={centroidb}, inplace={inplace})")
        centroida = self.extractCentroid(centroida, multiindex=False)
        centroidb = self.extractCentroid(centroidb, multiindex=False)
        distance = euclidean_distance(centroida, centroidb, multiindex=True)
        self.calcData = pd.concat([self.calcData, distance], axis=1)
        return distance if not inplace else None

    def pointAngle(self, point: str, inplace: bool = False) -> pd.DataFrame | None:
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(f"pointAngleDiff(point={point}, inplace={inplace})")
        angle = pd.DataFrame(self.polar[f"{point}_theta"], columns=pd.MultiIndex.from_product([[point], ["theta"]]))
        return angle if not inplace else None

    def centroidAngle(self, centroid: str, inplace: bool = False) -> pd.DataFrame | None:
        if centroid not in self.basefeats:
            raise ValueError(f"Centroid {centroid} not found")
        self.calcsLog.append(f"centroidAngleDiff(centroid={centroid}, inplace={inplace})")
        centroid_df = self.extractCentroid(centroid, topolar=True, multiindex=True, inplace=False)
        if centroid_df is None:
            raise ValueError(f"Centroid {centroid} not found")
        angle = centroid_df.loc[:, (centroid, "theta")]
        self.calcData = pd.concat([self.calcData, angle], axis=1)
        return angle if not inplace else None

    def velocities(self, inplace: bool = False) -> pd.DataFrame | None:
        self.calcsLog.append(f"velocities(inplace={inplace})")
        velocities = self.cartesian.loc[:, self.cartesianCols].diff()
        vel = [pd.Series] * len(self.points)
        for i, point in enumerate(self.points):
            vel[i] = pd.Series(np.sqrt(np.real(np.square(velocities[f"{point}_x"]) + np.square(velocities[f"{point}_y"])))).T
            vel[i].iloc[0] = vel[i].iloc[1]
        velocities: pd.DataFrame = pd.concat(vel, axis=1, keys=self.points)
        velocities.columns = pd.MultiIndex.from_product([list(self.points), ["velocity"]])
        self.calcData = pd.concat([self.calcData, velocities], axis=1)
        return velocities if not inplace else None

    def velocity(self, point: str, inplace: bool = False) -> pd.DataFrame | None:
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(f"velocity(inplace={inplace})")
        velocity = self.cartesian.loc[:, [f"{point}_x", f"{point}_y"]].diff()
        velocity = pd.Series(np.sqrt(np.real(np.square(velocity[f"{point}_x"]) + np.square(velocity[f"{point}_y"])))).T
        velocity: pd.DataFrame = pd.DataFrame(velocity, columns=pd.MultiIndex.from_product([list(point), ["velocity"]]))
        return velocity if not inplace else None

    def flattenScores(self) -> pd.Series:
        flatScores = self.scores.mean(axis=1)
        flatScores.name = "MeanScores"
        return flatScores

    @property
    def extract(self) -> pd.DataFrame:
        self.extractCentroids(inplace=True)
        self.extractCentroids(inplace=True, topolar=True)
        self.velocities(inplace=True)
        data = pd.concat([self.data, self.calcData], axis=1)
        return data.loc[:,~data.columns.duplicated()].copy()

    @property
    def extractManifold(self) -> pd.DataFrame:
        return pd.concat([self.cartesian, self.polar], axis=1)

class Cluster:
    def __init__(self, data: FeatureExtractor, prediction_column: str, pipeline: bool = False):
        from sklearn.model_selection import train_test_split

        self.data = data.extract
        self.cols = data.cols
        self.pred = prediction_column
        self.pipeline = None
        array, _ = self.data.columns.get_loc_level("Classes")
        self.numCols = np.invert(array)
        self.clusterData = self.data.loc[:, self.numCols]
        self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(
            self.clusterData, self.data.loc[:, ("Classes", self.pred)], test_size=0.3, random_state=12345
        )
        if pipeline:
            from sklearn.preprocessing import MaxAbsScaler
            from sklearn.decomposition import PCA

            self.pipeline = [
                ('scaler', MaxAbsScaler()),
                ('pca', PCA(n_components=len(data.points))),
            ]
            self.params = {
                'pca__n_components': [i for i in range((len(data.points) * 2))],
                'pca__whiten': [True, False],
                'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized']
            }
        else:
            self.pipeline = None


    def kmeans(self, output: bool = False, gridSearch: bool = False, *args, **kwargs):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 8

        kmeans = KMeans( *args, **kwargs)
        if self.pipeline is not None:
            kmeans = Pipeline([*self.pipeline, ('kmeans', kmeans)])
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

        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 3

        ap = AffinityPropagation( *args, **kwargs)
        if self.pipeline is not None:
            ap = Pipeline([*self.pipeline, ('ap', ap)])
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

    def heirarchical(self, output: bool = False,  *args, **kwargs):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        if "n_clusters" in kwargs or len(args) != 0:
            if "n_clusters" in kwargs:
                n_clusters = kwargs["n_clusters"]
            else:
                n_clusters = args[0]
        else:
            n_clusters = 2

        hc = AgglomerativeClustering( *args, **kwargs)
        if self.pipeline is not None:
            hc = Pipeline([*self.pipeline, ('hc', hc)])
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
        from sklearn.svm import OneClassSVM
        from sklearn.metrics import silhouette_score

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
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import BaggingRegressor

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
