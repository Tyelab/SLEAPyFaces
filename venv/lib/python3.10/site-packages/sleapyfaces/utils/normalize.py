import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def mean_center(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """Mean center the data.

    Args:
        data (pd.DataFrame): The data to be mean centered.
        track_names (list[str]): The names of the tracks to mean center.

    Returns:
        pd.DataFrame: The mean centered data."""
    num_data = data.loc[:, track_names]
    num_data = num_data - num_data.mean()
    data.loc[:, track_names] = num_data
    return data


def z_score(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """z-score the data.

    Args:
        data (pd.DataFrame): The data to be z-scored.
        track_names (list[str]): The names of the tracks to be z-scored.

    Returns:
        pd.DataFrame: The z-scored data."""
    data = mean_center(data, track_names)
    for track in track_names:
        data.loc[:, track] = data.loc[:, track] / data.loc[:, track].std()
    return data


def pca(data: pd.DataFrame, track_names: list[str], multiindex: bool = None, *args, **kwargs) -> dict[str, pd.DataFrame]:
    """Runs 2D and 3D PCA dimensionality reduction on the data.

    Args:
        data (pd.DataFrame): The data to be reduced.
        track_names (list[str]): The names of the tracks to be reduced.
        *args, **kwargs: Additional arguments to pass to pd.DataFrame.reset_index().

    Returns:
        dict[str, pd.DataFrame]: The reduced data with keys "pca2d" and "pca3d"."""

    if multiindex is None:
        if isinstance(data.columns, pd.MultiIndex):
            multiindex = True
        else:
            multiindex = False

    num_data = data.loc[:, track_names]
    qual_data = data.drop(columns=track_names)
    pcas = {}

    pca2d = PCA(n_components=2)
    pca3d = PCA(n_components=3)

    num_data_2d = pca2d.fit_transform(num_data)
    num_data_3d = pca3d.fit_transform(num_data)

    num_data_2d = pd.DataFrame(
        num_data_2d, columns=["principal component 1", "principal component 2"]
    )
    num_data_3d = pd.DataFrame(
        num_data_3d,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    if multiindex:
        num_data_2d.columns = pd.MultiIndex.from_product([["PCA-2D"], num_data_2d.columns])
        num_data_3d.columns = pd.MultiIndex.from_product([["PCA-3D"], num_data_3d.columns])

    pcas["pca2d"] = pd.concat([qual_data.reset_index( *args, **kwargs), num_data_2d], axis=1)
    pcas["pca3d"] = pd.concat([qual_data.reset_index( *args, **kwargs), num_data_3d], axis=1)

    return pcas

# create gaussian kernel for smoothing
def gaussian_kernel(window_size: int, sigma=1) -> np.ndarray:
    """
        Summary:
        this function creates a gaussian kernel for back smoothing

    Args:
        window_size (int): how many frames to smooth over
        sigma (int, optional): relative standard deviation. Defaults to 1.

    Returns:
        np.array: returns a kernel to smooth over with shape (window_size,)
    """
    x_vals = np.arange(window_size)
    to_ret = np.exp(-((x_vals - window_size // 2) * 2) / (2 * sigma * 2))
    to_ret[: window_size // 2] = 0
    return to_ret
