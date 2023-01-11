import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Protocol
from sleapyfaces.utils.graph import cartesian_to_polar, euclidean_distance

class dataobjectprotocol(Protocol):
    data: pd.DataFrame
    scores: pd.DataFrame
    quant_cols: list[str]
    qual_cols: list[str]
    cols: tuple[list[str], list[str]]

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
