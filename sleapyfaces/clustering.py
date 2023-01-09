from tkinter import N
import pandas as pd
from dataclasses import dataclass
import numpy as np
from math import sqrt

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

def cartesian_to_polar(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Converts cartesian pandas dataframe columns to polar coordinates.

    Args:
        df (pd.DataFrame): The dataframe to convert to polar coordinates.
        cols (list[str]): The columns to convert.

    Returns:
        df (pd.DataFrame): The dataframe with the polar columns appended as "{...}_r" and "{...}_theta".
    """
    df = df.copy()
    for i in range(0, len(cols), 2):
            df[cols[i].replace("_x", "_r")], df[cols[i+1].replace("_y", "_theta")] = zip(
                *df.apply(
                    lambda column: polar(
                        column[cols[i]], column[cols[i + 1]]
                    ),
                    axis=1
                )
            )
    df.drop(columns=cols, inplace=True)
    return df

def euclidean_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Calculates the euclidean distance between two dataframes.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        pd.DataFrame: The euclidean distance between the two dataframes.
    """
    x_sq = (df1.filter(regex="_x").to_numpy() - df2.filter(regex="_x").to_numpy()) ** 2
    y_sq = (df1.filter(regex="_y").to_numpy() - df2.filter(regex="_y").to_numpy()) ** 2
    return pd.DataFrame(np.sqrt(x_sq + y_sq).flatten(), index=df1.index)

@dataclass(slots=True)
class FeatureExtractor:

    data: pd.DataFrame
    cols: list[str]
    basefeats: list[str]
    polar: pd.DataFrame
    points: set[str]
    calcsLog: list[str]

    def __init__(self, data: pd.DataFrame, cluster_columns: list[str], base_features: list[str]):
        self.data = data
        self.cols = cluster_columns
        self.basefeats = base_features
        self.polar = cartesian_to_polar(self.data.copy(), self.cols)
        self.points = set([point.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", "") for point in [*self.data.columns.to_list(), *self.polar.columns.to_list()]])
        self.calcsLog = []

    def extractCentroids(self, inplace: bool = False, topolar: bool = False):
        self.calcsLog.append(f"extractCentroids(inplace={inplace}, topolar={topolar})")
        df_coords = []
        for feature in self.basefeats:
            for coord in ["_x", "_y"]:
                df_coord: pd.Series = self.data.loc[:, self.cols].filter(like=feature).filter(like=coord).groupby(level=[0, 1], group_keys=False).apply(np.mean, axis=1)
                df_coords.append(df_coord.rename(f"{feature}{coord}"))
        centroids: pd.DataFrame = pd.concat(df_coords, axis=1)
        centroid_cols: list[str] = centroids.columns.to_list()
        if topolar:
            centroids = cartesian_to_polar(centroids, centroid_cols)
        if inplace:
            self.points.add([point.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", "") for point in centroid_cols])
            self.data = pd.concat([self.data, centroids], axis=1)
        else:
            return centroids

    def extractCentroid(self, feature: str, inplace: bool = False, topolar: bool = False):
        self.calcsLog.append(f"extractCentroid(feature={feature}, inplace={inplace}, topolar={topolar})")
        df_coords = []
        for coord in ["_x", "_y"]:
            df_coord: pd.Series = self.data.loc[:, self.cols].filter(like=feature).filter(like=coord).groupby(level=[0, 1], group_keys=False).apply(np.mean, axis=1)
            df_coords.append(df_coord.rename(f"{feature}{coord}"))
        centroid: pd.DataFrame = pd.concat(df_coords, axis=1)
        centroid_col: list[str] = centroid.columns.to_list()
        if topolar:
            centroid = cartesian_to_polar(centroid, centroid_col)
        if inplace:
            self.points.add([point.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", "") for point in centroid_col])
            self.data = pd.concat([self.data, centroid], axis=1)
        else:
            return centroid

    def twoPointsDist(self, pointa: str, pointb: str, title: str, inplace: bool = False):
        self.calcsLog.append(f"twoPointsDist(pointa={pointa}, pointb={pointb}, title={title}, inplace={inplace})")
        distance = euclidean_distance(self.data.loc[:, [f"{pointa}_x", f"{pointa}_y"]], self.data.loc[:, [f"{pointb}_x", f"{pointb}_y"]])
        distance.columns = [title]
        if inplace:
            self.points.add(title.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", ""))
            self.data = pd.concat([self.data, distance], axis=1)
        else:
            return distance

    def distToCentroid(self, point: str, centroid: str, title: str, inplace: bool = False):
        self.calcsLog.append(f"distToCentroid(point={point}, centroid={centroid}, title={title}, inplace={inplace})")
        centroid = self.extractCentroid(centroid)
        distance = euclidean_distance(self.data.loc[:, [f"{point}_x", f"{point}_y"]], centroid)
        distance.columns = [title]
        if inplace:
            self.points.add(title.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", ""))
            self.data = pd.concat([self.data, distance], axis=1)
        else:
            return distance

    def twoCentroidsDist(self, centroida: str, centroidb: str, title: str, inplace: bool = False):
        self.calcsLog.append(f"twoCentroidsDist(centroida={centroida}, centroidb={centroidb}, title={title}, inplace={inplace})")
        centroida = self.extractCentroid(centroida)
        centroidb = self.extractCentroid(centroidb)
        distance = euclidean_distance(centroida, centroidb)
        distance.columns = [title]
        if inplace:
            self.points.add(title.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", ""))
            self.data = pd.concat([self.data, distance], axis=1)
        else:
            return distance

    def pointAngle(self, point: str, title: str, inplace: bool = False):
        self.calcsLog.append(f"pointAngleDiff(point={point}, title={title}, inplace={inplace})")
        angle = pd.DataFrame(self.polar[[f"{point}_theta"]], columns=[title])
        if inplace:
            self.points.add(title.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", ""))
            self.data = pd.concat([self.data, angle], axis=1)
        else:
            return angle

    def centroidAngle(self, centroid: str, title: str, inplace: bool = False):
        self.calcsLog.append(f"centroidAngleDiff(centroid={centroid}, title={title}, inplace={inplace})")
        centroid_df = self.extractCentroid(centroid, topolar=True)
        if centroid_df is None:
            raise ValueError(f"Centroid {centroid} not found")
        angle = pd.DataFrame(centroid_df[[f"{centroid}_theta"]], columns=[title])
        if inplace:
            self.points.add(title.replace("_x", "").replace("_y", "").replace("_r", "").replace("_theta", ""))
            self.data = pd.concat([self.data, angle], axis=1)
        else:
            return angle

    @property
    def extract(self) -> pd.DataFrame:
        return pd.concat([self.data, self.polar], axis=1)

class Cluster:
    def __init__(self, data: FeatureExtractor, prediction_column: str):
        from sklearn.model_selection import train_test_split

        self.data = data.extract
        self.cols = data.cols
        self.pred = prediction_column
        self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(
            self.data.loc[:, self.cols], self.data.loc[:, self.pred], test_size=0.3, random_state=12345
        )

    def kmeans(self, n_clusters: int, output_all: bool = False):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.data.loc[:, self.cols])
        labels = kmeans.predict(self.data.loc[:, self.cols])
        clusters = pd.DataFrame(labels, columns=["KMeans-Cluster"])
        self.data = pd.concat([self.data, clusters], axis=1)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.data.loc[:, self.cols], labels)}')

        if output_all:
            return kmeans

    def affinity_propagation(self, n_clusters: int, preference: np.ndarray = None, output_all: bool = False):
        from sklearn.cluster import AffinityPropagation
        from sklearn.metrics import silhouette_score

        if preference is not None:
            ap = AffinityPropagation(n_clusters=n_clusters, preference=preference)
        else:
            ap = AffinityPropagation(n_clusters=n_clusters)
        ap.fit(self.data.loc[:, self.cols])
        labels = ap.predict(self.data.loc[:, self.cols])
        clusters = pd.DataFrame(labels, columns=["AffinityPropagation-Cluster"])
        self.data = pd.concat([self.data, clusters], axis=1)
        print(f'Silhouette Score(n={n_clusters}): {silhouette_score(self.data.loc[:, self.cols], labels)}')

        if output_all:
            return ap

    def knn(self, n_neighbors: int | list[int], bagged: bool = False, output_all: bool = False):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error

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
            from sklearn.model_selection import GridSearchCV

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

        if bagged:
            from sklearn.ensemble import BaggingRegressor

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

        if output_all:
            return knn_model
