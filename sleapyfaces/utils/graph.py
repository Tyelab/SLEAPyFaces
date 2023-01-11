import pandas as pd
import numpy as np

def euclidean_distance(df1: pd.DataFrame, df2: pd.DataFrame, multiindex: bool = False) -> pd.DataFrame:
    """Calculates the euclidean distance between two dataframes.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        pd.DataFrame: The euclidean distance between the two dataframes.
    """

    def extractName(cols: list[any]) -> str:
        if type(cols) is not str:
            cols = extractName(cols[0])
        return cols

    point1 = extractName(df1.columns.to_list()).replace('_x', '').replace('_y', '').replace('_r', '').replace('_theta', '')
    point2 = extractName(df2.columns.to_list()).replace('_x', '').replace('_y', '').replace('_r', '').replace('_theta', '')
    x = np.subtract(df1.filter(regex="_x").to_numpy(), df2.filter(regex="_x").to_numpy())
    y = np.subtract(df1.filter(regex="_y").to_numpy(), df2.filter(regex="_y").to_numpy())
    x = np.real(np.square(x))
    y = np.real(np.square(y))
    df = pd.DataFrame(np.squeeze(np.sqrt(x + y)).T)
    df.columns = [f"distance({point1}->{point2})"] if not multiindex else pd.MultiIndex.from_tuples([(f"{point1}->{point2}", "euclidean_distance")])
    return df

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
