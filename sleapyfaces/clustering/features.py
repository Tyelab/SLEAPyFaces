from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import logging

import numpy as np
import pandas as pd

from sleapyfaces.utils.graph import cartesian_to_polar, euclidean_distance


class dataobjectprotocol(Protocol):
    all_data: pd.DataFrame
    all_scores: pd.DataFrame
    quant_cols: list[str]
    qual_cols: list[str]
    cols: tuple[list[str], list[str]]


@dataclass(slots=True)
class FeatureExtractor:
    """Summary:
        A functional class for extracting features from the data output from one of the base types of data objects. (i.e. sleapyfaces.types... Projects, Project, or Experiment)

    Args:
        dataObject (sleapyfaces.types.Projects | sleapyfaces.types.Project | sleapyfaces.types.Experiment): The data object from which to extract features.
        base_features (list[str]): A list of base features to build centroids of (i.e. ['nose', 'eye', 'ear'])

    Attributes:
        data (pd.DataFrame): The combined cartesian and polar data from the data object.
        cols (list[str]): The list of columns in the data attribute (good for quickly building an iterator). (e.g. ['upper_nose_x', 'upper_nose_y', 'upper_nose_r', 'upper_nose_theta', ...])
        points (list[str]): The list of facial points from which the data is extracted (good for quickly building an iterator). (e.g. ['upper_nose', 'lower_nose', 'upper_ear', 'lower_ear', ...])
        classes (list[str]): A list of the identity classes in the data object. (e.g. ['Weeks', 'Mouse', 'Trial', 'Trial_index', ...])
        basefeats (list[str]): The list of base features to build centroids of (i.e. ['nose', 'eye', 'ear'])
        cartesian (pd.DataFrame): The cartesian data from the data object.
        cartesianCols (list[str]): The list of columns in the cartesian attribute (good for quickly building an iterator). (e.g. ['upper_nose_x', 'upper_nose_y', 'upper_ear_x', 'upper_ear_y', ...])
        polar (pd.DataFrame): The polar data from the data object.
        polarCols (list[str]): The list of columns in the polar attribute (good for quickly building an iterator). (e.g. ['upper_nose_r', 'upper_nose_theta', 'upper_ear_r', 'upper_ear_theta', ...])
        scores (pd.DataFrame): The SLEAP tracking scores from the data object. (good for weighting the data in classifiers)
        calcData (pd.DataFrame): A dataframe of the extracted features (thus far).
        calcsLog (list[str]): A log of the calculations performed on the data.
        all_data (pd.DataFrame): A dataframe of all of the extracted features (including the original data) with validation checks in place.
    """

    data: pd.DataFrame
    cartesian: pd.DataFrame
    polar: pd.DataFrame
    scores: pd.DataFrame
    calcData: pd.DataFrame
    points: set[str]
    columns: list[str]
    cartesianCols: list[str]
    polarCols: list[str]
    classes: list[str]
    basefeats: list[str]
    calcsLog: list[str]
    qual_cols: list[str]
    _all_calcs: list[str] = None
    _all_data: pd.DataFrame = None

    def __init__(self, dataObject: dataobjectprotocol, base_features: list[str]):

        logging.info("Extracting Features...")
        self.basefeats: list[str] = base_features

        self.cartesian: pd.DataFrame = dataObject.all_data
        self.cartesianCols, self.classes = dataObject.cols

        self.polar: pd.DataFrame = cartesian_to_polar(
            self.cartesian, self.cartesianCols)
        self.polarCols: list[str] = [col.replace("_x", "_r").replace(
            "_y", "_theta") for col in self.cartesianCols]

        self.columns: list[str] = self.cartesianCols + self.polarCols
        self.points: set[str] = set(
            [point.replace("_x", "").replace("_y", "") for point in self.cartesianCols])

        all_data: pd.DataFrame = pd.concat(
            [self.cartesian, self.polar], axis=1)
        all_data.reset_index(inplace=True)
        num_data: list[pd.DataFrame] = [pd.DataFrame({'x': np.squeeze(all_data.loc[:, self.columns].filter(like=point).filter(like="_x").values), 'y': np.squeeze(all_data.loc[:, self.columns].filter(like=point).filter(like="_y").values), 'r': np.squeeze(
            all_data.loc[:, self.columns].filter(like=point).filter(like="_r").values), 'theta': np.squeeze(all_data.loc[:, self.columns].filter(like=point).filter(like="_theta").values).T}, columns=["x", "y", "r", "theta"]) for point in self.points]
        num_data: pd.DataFrame = pd.concat(num_data, axis=1, keys=self.points)
        qual_data: pd.DataFrame = all_data.loc[:, self.classes]
        qual_data: pd.DataFrame = qual_data.loc[:,
                                                ~qual_data.columns.duplicated()].copy()
        qual_data.columns = pd.MultiIndex.from_product(
            [["Classes"], self.classes])
        self.qual_cols: list[str] = qual_data.columns.to_list()
        self.data = pd.concat([qual_data, num_data], axis=1)

        self.cartesian.reset_index(inplace=True)
        self.cartesian.drop(columns=self.classes, inplace=True)
        self.polar.reset_index(inplace=True)
        self.polar.drop(columns=self.classes, inplace=True)
        self.calcData = qual_data

        self.calcsLog = []
        self.scores = dataObject.all_scores

        self._all_data = None
        self._all_calcs = None

    def extractCentroids(self, inplace: bool = False, topolar: bool = False) -> pd.DataFrame | None:
        """Extracts all the centroids of the base features.

        Args:
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.
            topolar (bool, optional): whether to convert the output to polar coordinates. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        if self.basefeats is None:
            raise ValueError("No base features have been defined.")
        self.calcsLog.append(
            f"extractCentroids(inplace={inplace}, topolar={topolar})")
        df_coords = []
        for feature in self.basefeats:
            for coord in ["x", "y"]:
                df_coord: pd.Series = self.cartesian.loc[:, self.cartesianCols].filter(
                    like=feature).filter(like=coord).apply(np.mean, axis=1)
                df_coords.append(df_coord.rename((feature, coord)))
        centroids: pd.DataFrame = pd.concat(df_coords, axis=1)
        if topolar:
            centroids = cartesian_to_polar(
                centroids, centroids.columns.to_list())
        if inplace:
            self.calcData = self.calcData.join(centroids, how="outer")
        else:
            return centroids

    def extractCentroid(self, feature: str, inplace: bool = False, topolar: bool = False, multiindex: bool = True) -> pd.DataFrame | None:
        """Extracts a centroid from the base features.

        Args:
            feature (str): the feature to extract the centroid from. Must be a base feature (see basefeats attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.
            topolar (bool, optional): whether to convert the output to polar coordinates. Defaults to False.
            multiindex (bool, optional): whether to use a multiindex for the output. Defaults to True.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        if feature not in self.basefeats:
            raise ValueError(f"Centroid {feature} not found")
        self.calcsLog.append(
            f"extractCentroid(feature={feature}, inplace={inplace}, topolar={topolar})")
        df_coords = []
        for coord in ["x", "y"]:
            df_coord: pd.Series = self.cartesian.loc[:, self.cartesianCols].filter(
                like=feature).filter(like=coord).apply(np.mean, axis=1)
            if multiindex:
                df_coords.append(df_coord.rename((feature, coord)))
            else:
                df_coords.append(df_coord.rename(f"{feature}_{coord}"))
        centroid: pd.DataFrame = pd.concat(df_coords, axis=1)
        if topolar:
            centroid = cartesian_to_polar(centroid, centroid.columns.to_list())
        if inplace:
            self.calcData = self.calcData.join(centroid, how="outer")
        else:
            return centroid

    def twoPointsDist(self, pointa: str, pointb: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the euclidean distance between two points.

        Args:
            pointa (str): the start point. Must be an original point (see points attribute).
            pointb (str): the end point. Must be an original point (see points attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        logging.info(
            f"\tExtracting distance between points: {pointa} and {pointb}")
        if pointa not in self.points or pointb not in self.points:
            raise ValueError(f"Point {pointa} or {pointb} not found")
        self.calcsLog.append(
            f"twoPointsDist(pointa={pointa}, pointb={pointb}, inplace={inplace})")
        distance = euclidean_distance(self.cartesian.loc[:, [f"{pointa}_x", f"{pointa}_y"]], self.cartesian.loc[:, [
                                      f"{pointb}_x", f"{pointb}_y"]], multiindex=True)
        if inplace:
            self.calcData = self.calcData.join(distance, how="outer")
        else:
            return distance

    def distToCentroid(self, point: str, centroid: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the euclidean distance between a point and a centroid.

        Args:
            point (str): the point. Must be an original point (see points attribute).
            centroid (str): the centroid. Must be a base feature (see basefeats attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        logging.info(
            f"\tExtracting distance between point and centroid: {point} and {centroid}")
        if centroid not in self.basefeats:
            raise ValueError(f"Centroid {centroid} not found")
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(
            f"distToCentroid(point={point}, centroid={centroid}, inplace={inplace})")
        centroid = self.extractCentroid(centroid, multiindex=False)
        distance = euclidean_distance(
            self.cartesian.loc[:, [f"{point}_x", f"{point}_y"]], centroid, multiindex=True)
        if inplace:
            self.calcData = self.calcData.join(distance, how="outer")
        else:
            return distance

    def twoCentroidsDist(self, centroida: str, centroidb: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the euclidean distance between two centroids.

        Args:
            centroida (str): the first centroid. Must be a base feature (see basefeats attribute).
            centroidb (str): the second centroid. Must be a base feature (see basefeats attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        logging.info(
            f"\tExtracting distance between centroids: {centroida} and {centroidb}")
        if centroida not in self.basefeats:
            raise ValueError(f"Centroid 1 {centroida} not found")
        if centroidb not in self.basefeats:
            raise ValueError(f"Centroid 2 {centroidb} not found")
        self.calcsLog.append(
            f"twoCentroidsDist(centroida={centroida}, centroidb={centroidb}, inplace={inplace})")
        centroida = self.extractCentroid(centroida, multiindex=False)
        centroidb = self.extractCentroid(centroidb, multiindex=False)
        distance = euclidean_distance(centroida, centroidb, multiindex=True)
        if inplace:
            self.calcData = self.calcData.join(distance, how="outer")
        else:
            return distance

    def pointAngle(self, point: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the vector angle of a point.

        Args:
            point (str): the point. Must be an original point (see points attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(
            f"pointAngleDiff(point={point}, inplace={inplace})")
        angle = pd.DataFrame(
            self.polar[f"{point}_theta"], columns=pd.MultiIndex.from_product([[point], ["theta"]]))
        if inplace:
            self.calcData = self.calcData.join(angle, how="outer")
        else:
            return angle

    def centroidAngle(self, centroid: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the vector angle of a centroid.

        Args:
            centroid (str): the centroid. Must be a base feature (see basefeats attribute).
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        if centroid not in self.basefeats:
            raise ValueError(f"Centroid {centroid} not found")
        self.calcsLog.append(
            f"centroidAngleDiff(centroid={centroid}, inplace={inplace})")
        centroid_df = self.extractCentroid(
            centroid, topolar=True, multiindex=True, inplace=False)
        if centroid_df is None:
            raise ValueError(f"Centroid {centroid} not found")
        angle = centroid_df.loc[:, (centroid, "theta")]
        if inplace:
            self.calcData = self.calcData.join(angle, how="outer")
        else:
            return angle

    def velocities(self, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the velocities of all the points.

        Args:
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        self.calcsLog.append(f"velocities(inplace={inplace})")
        velocities = self.cartesian.loc[:, self.cartesianCols].diff()
        vel = [pd.Series] * len(self.points)
        for i, point in enumerate(self.points):
            vel[i] = pd.Series(np.sqrt(np.real(
                np.square(velocities[f"{point}_x"]) + np.square(velocities[f"{point}_y"])))).T
            vel[i].iloc[0] = vel[i].iloc[1]
        velocities: pd.DataFrame = pd.concat(vel, axis=1, keys=self.points)
        velocities.columns = pd.MultiIndex.from_product(
            [list(self.points), ["velocity"]])
        if inplace:
            self.calcData = self.calcData.join(velocities, how="outer")
        else:
            return velocities

    def velocity(self, point: str, inplace: bool = False) -> pd.DataFrame | None:
        """Extracts the velocity of a single point.

        Args:
            inplace (bool, optional): whether to perform this computation in-place. Defaults to False.

        Returns:
            pd.DataFrame | None: The centroids of the base features. If inplace is True, returns None.
        """
        if point not in self.points:
            raise ValueError(f"Point {point} not found")
        self.calcsLog.append(f"velocity(inplace={inplace})")
        velocity = self.cartesian.loc[:, [f"{point}_x", f"{point}_y"]].diff()
        velocity = pd.Series(np.sqrt(np.real(
            np.square(velocity[f"{point}_x"]) + np.square(velocity[f"{point}_y"])))).T
        velocity: pd.DataFrame = pd.DataFrame(
            velocity, columns=pd.MultiIndex.from_product([list(point), ["velocity"]]))
        if inplace:
            self.calcData = self.calcData.join(velocity, how="outer")
        else:
            return velocity

    def flattenScores(self) -> pd.Series:
        flatScores = self.scores.mean(axis=1)
        flatScores.name = "MeanScores"
        return flatScores

    @property
    def all_data(self) -> pd.DataFrame:
        if self._all_data is None or self._all_calcs != self.calcsLog:
            self._all_calcs = self.calcsLog
            self._all_data = self.extract
        return self._all_data

    # @property
    # def cartesian(self) -> pd.DataFrame:
    #     return self.calcData.loc[:, self.cartesianCols]

    @property
    def extractManifold(self) -> pd.DataFrame:
        return pd.concat([self.cartesian, self.polar], axis=1)

    @property
    def extract(self) -> pd.DataFrame:
        logging.info("Extracting calculations")
        df = self.calcData
        logging.info("Extracting centroids")
        centroids = self.extractCentroids()
        if centroids is None:
            raise ValueError("No centroids found")
        df = df.join(centroids)
        logging.info("Converting centroids to polar coordinate")
        centroids = cartesian_to_polar(centroids, centroids.columns.to_list())
        df = df.join(centroids)
        del centroids
        logging.info("Extracting velocities")
        df = df.join(self.velocities())
        logging.info("Adding point data")
        return self.data.merge(df)

    @property
    def quant_cols(self) -> list[str]:
        return [col for col in self._all_data.columns.to_list() if col not in self.qual_cols]

    @property
    def cols(self) -> tuple[list[str], list[str]]:
        return self.qual_cols, self.quant_cols
