import pandas as pd
import numpy as np
import itertools
import plotly.graph_objects as go

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

def graph_pcas(data: pd.DataFrame, data_cols: list[str] | list[tuple] | str = None, iter_cols: list[str] | list[tuple] | tuple | str = "All", color_col: tuple | str = None, n_components: int = 3, save: bool = False, save_path: str = None, show: bool = True, title: str = None, inplace: bool = True, interactive: bool = True, **kwargs) -> None:
    """Plots the 3D principal components of a dataframe.

    Args:
        data (pd.DataFrame): The dataframe to plot.
        iter_cols (list[str] | str, optional): The columns to iterate over. Defaults to "All".
        n_components (int, optional): The number of components to plot. Defaults to 3.
        save (bool, optional): Whether to save the plot. Defaults to False.
        save_path (str, optional): The path to save the plot to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        title (str, optional): The title of the plot. Defaults to None.
        inplace (bool, optional): Whether to plot the data inplace. Defaults to True.
        interactive (bool, optional): Whether to plot the data interactively. Defaults to True.
    """
    if data_cols is None:
        raise TypeError("data_cols must be a list, tuple, or str representing column name(s) in the data.")

    if iter_cols != "All":
        if not (type(iter_cols) is list or type(iter_cols) is tuple or type(iter_cols) is str or iter_cols in data.columns):
            raise TypeError("iter_cols must be a list, tuple, or str representing column name(s) in the data.")

        print(iter_cols)
        if type(iter_cols) is list:
            iterator = itertools.product(*[data[c].unique().tolist() for c in iter_cols])
        else:
            iterator = data.loc[:, iter_cols].unique().tolist()
        for itr in iterator:
            if type(iter_cols) is list:
                df = data.loc[np.logical_and(*[(data[c] == i).to_numpy() for c, i in zip(iter_cols, itr)])]
                _title = ' & '.join([f"{c} = {i}" for c, i in zip(iter_cols, itr)])
            else:
                df = data.loc[data[iter_cols] == itr]
                _title = f"{iter_cols} = {itr}"

            if color_col is not None:
                colors = df[color_col].to_numpy()

            if interactive:
                if n_components == 2:
                    plotly_2d(df[data_cols].to_numpy(),
                        color=colors,
                        title=title+_title if title is not None else _title,
                        save=save,
                        save_path=save_path,
                        show=show,
                        **kwargs)
                elif n_components == 3:
                    plotly_3d(df[data_cols].to_numpy(),
                        color=colors,
                        title=title+_title if title is not None else _title,
                        save=save,
                        save_path=save_path,
                        show=show,
                        **kwargs)
            else:
                if n_components == 2:
                    matplotlib_2d(df[data_cols].to_numpy(),
                        color=colors,
                        title=title+_title if title is not None else _title,
                        save=save,
                        save_path=save_path,
                        show=show,
                        **kwargs)
                elif n_components == 3:
                    matplotlib_3d(df[data_cols].to_numpy(),
                        color=colors,
                        title=title+_title if title is not None else _title,
                        save=save,
                        save_path=save_path,
                        show=show,
                        **kwargs)

    else:
        df = data[data_cols]
        if color_col is not None:
            colors = data[color_col].to_numpy()
        if interactive:
            if n_components == 2:
                plotly_2d(df.to_numpy(),
                    color=colors,
                    title=title,
                    save=save,
                    save_path=save_path,
                    show=show,
                    **kwargs)
            elif n_components == 3:
                plotly_3d(df.to_numpy(),
                    color=colors,
                    title=title,
                    save=save,
                    save_path=save_path,
                    show=show,
                    **kwargs)
        else:
            if n_components == 2:
                matplotlib_2d(df.to_numpy(),
                    color=colors,
                    title=title,
                    save=save,
                    save_path=save_path,
                    show=show,
                    **kwargs)
            elif n_components == 3:
                matplotlib_3d(df.to_numpy(),
                    color=colors,
                    title=title,
                    save=save,
                    save_path=save_path,
                    show=show,
                    **kwargs)

def plotly_2d(data: np.ndarray, color: list | str | np.ndarray = None, xrange: list | tuple = None, yrange: list | tuple = None, title: str = None, save: bool = False, save_path: str = None, show: bool = True, **kwargs) -> None:
    """Plots a 2D array using plotly.

    Args:
        data (np.ndarray): The array to plot.
        color (list | str | np.ndarray, optional): The color of the data. Defaults to None.
        xrange (list | tuple, optional): The x range of the plot. Defaults to None.
        yrange (list | tuple, optional): The y range of the plot. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        save (bool, optional): Whether to save the plot. Defaults to False.
        save_path (str, optional): The path to save the plot to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    import plotly.graph_objects as go
    if 2 in data.shape:
        if len(data.shape) > 2:
            data = np.squeeze(data)
        if data.shape[0] != 2:
            data = data.T
        else:
            data = data
    if color is not None:
        if isinstance(color, np.ndarray):
            if len(color.shape) > 1:
                if 1 in color.shape:
                    color = np.squeeze(color)
                if len(color.shape) > 1:
                    raise ValueError("color must be a 1D array.")
            if np.issubdtype(color.dtype, np.number):
                color = color.astype(np.int)
            else:
                raise TypeError("if color is instance of np.ndarray it must be a numeric array.")
        fig = go.Figure(
            go.Scatter(
                x=data[0],
                y=data[1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    colorscale='Viridis',
                    opacity=0.8,
                )))
    else:
        fig = go.Figure(
            go.Scatter(
                x=data[0],
                y=data[1],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.8,
                )))
    if title is not None:
        fig.update_layout(title=title)
    if xrange is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=xrange,
                    autorange=False,
                    title_text="PC1")))
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title_text="PC1")))
    if yrange is not None:
        fig.update_layout(
            scene=dict(
                yaxis=dict(
                    range=yrange,
                    autorange=False,
                    title_text="PC2")))
    else:
        fig.update_layout(
            scene=dict(
                yaxis=dict(title_text="PC2")))
    fig.update_layout(
        width=1920,
        height=1080,
        paper_bgcolor='rgba(0,0,0,0)',
        **kwargs)
    if save:
        if save_path is None:
            save_path = "plot.html"
        fig.write_html(save_path+title.replace(" ", "_")+".html")
    if show:
        fig.show()

def plotly_3d(data: np.ndarray, color: list | str | np.ndarray = None, xrange: list | tuple = None, yrange: list | tuple = None, zrange: list | tuple = None, title: str = None, save: bool = False, save_path: str = None, show: bool = True, **kwargs) -> None:
    """Plots a 2D array using plotly.

    Args:
        data (np.ndarray): The array to plot.
        color (list | str | np.ndarray, optional): The color of the data. Defaults to None.
        xrange (list | tuple, optional): The x range of the plot. Defaults to None.
        yrange (list | tuple, optional): The y range of the plot. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        save (bool, optional): Whether to save the plot. Defaults to False.
        save_path (str, optional): The path to save the plot to. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    import plotly.graph_objects as go
    if 3 in data.shape:
        if len(data.shape) > 2:
            data = np.squeeze(data)
        if data.shape[0] != 3:
            data = data.T
        else:
            data = data
    if color is not None:
        if isinstance(color, np.ndarray):
            if len(color.shape) > 1:
                if 1 in color.shape:
                    color = np.squeeze(color)
                if len(color.shape) > 1:
                    raise ValueError("color must be a 1D array.")
            if np.issubdtype(color.dtype, np.number):
                color = color.astype(np.int)
            else:
                raise TypeError("if color is instance of np.ndarray it must be a numeric array.")
        fig = go.Figure(
            go.Scatter3d(
                x=data[0],
                y=data[1],
                z=data[2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    colorscale='Viridis',
                    opacity=0.8,
                )))
    else:
        fig = go.Figure(
            go.Scatter(
                x=data[0],
                y=data[1],
                z=data[2],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.8,
                )))
    if title is not None:
        fig.update_layout(title=title)
    if xrange is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=xrange,
                    autorange=False,
                    title_text="PC1")))
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title_text="PC1")))
    if yrange is not None:
        fig.update_layout(
            scene=dict(
                yaxis=dict(
                    range=yrange,
                    autorange=False,
                    title_text="PC2")))
    else:
        fig.update_layout(
            scene=dict(
                yaxis=dict(title_text="PC2")))
    if zrange is not None:
        fig.update_layout(
            scene=dict(
                zaxis=dict(
                    range=zrange,
                    autorange=False,
                    title_text="PC3")))
    else:
        fig.update_layout(
            scene=dict(
                zaxis=dict(title_text="PC3")))
    fig.update_layout(
        width=1920,
        height=1080,
        paper_bgcolor='rgba(0,0,0,0)',
        **kwargs)
    if save:
        if save_path is None:
            save_path = "plot.html"
        fig.write_html(save_path+title.replace(" ", "_")+".html")
    if show:
        fig.show()

def matplotlib_2d(data: np.ndarray, color: list | str | np.ndarray = None, xrange: list | tuple = None, yrange: list | tuple = None, zrange: list | tuple = None, title: str = None, save: bool = False, save_path: str = None, show: bool = True, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import scatter

    if 2 in data.shape:
        if len(data.shape) > 2:
            data = np.squeeze(data)
        if data.shape[0] != 2:
            data = data.T
        else:
            data = data

    fig = plt.figure()
    ax = fig.add_subplot()

    if color is not None:
        if isinstance(color, np.ndarray):
            if len(color.shape) > 1:
                if 1 in color.shape:
                    color = np.squeeze(color)
                if len(color.shape) > 1:
                    raise ValueError("color must be a 1D array.")
            if np.issubdtype(color.dtype, np.number):
                color = color.astype(np.int)
            else:
                raise TypeError("if color is instance of np.ndarray it must be a numeric array.")
        if type(color) is not str:
            ax.scatter(data[0], data[1], c=color, cmap="viridis")
            feats = np.unique(color).tolist()
            cmap = plt.cm.viridis
            colors = [scatter([], [], color=cmap(i/len(feats))) for i in range(len(feats))]
            fig.legend(colors, feats, loc='upper right')
        else:
            ax.scatter(data[0], data[1], c=color)
    else:
        ax.scatter(data[0], data[1])

    if xrange is not None:
        ax.set_xlim(xrange)

    if yrange is not None:
        ax.set_ylim(yrange)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if title is None:
        title = "2D PCA"

    ax.set_title(title)

    if save:
        if save_path is None:
            save_path = f"PCA.png"
        plt.savefig(save_path+title.replace(" ", "_")+".png", dpi=300)
    if show:
        plt.show()

def matplotlib_3d(data: np.ndarray | pd.DataFrame, color: list | str | np.ndarray = None, xrange: list | tuple = None, yrange: list | tuple = None, zrange: list | tuple = None, title: str = None, save: bool = False, save_path: str = None, show: bool = True, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.pyplot import scatter
    import matplotlib.pyplot as plt

    if 3 in data.shape:
        if len(data.shape) > 2:
            data = np.squeeze(data)
        if data.shape[0] != 3:
            data = data.T
        else:
            data = data

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if color is not None:
        if isinstance(color, np.ndarray):
            if len(color.shape) > 1:
                if 1 in color.shape:
                    color = np.squeeze(color)
                if len(color.shape) > 1:
                    raise ValueError("color must be a 1D array.")
            if np.issubdtype(color.dtype, np.number):
                color = color.astype(np.int)
            else:
                raise TypeError("if color is instance of np.ndarray it must be a numeric array.")
        if type(color) is not str:
            ax.scatter(data[0], data[1], data[2], s=4, c=color, cmap="viridis")
            feats = np.unique(color).tolist()
            cmap = plt.cm.viridis
            colors = [scatter(x=[], y=[], s=4, color=cmap(i/len(feats))) for i in range(len(feats))]
            fig.legend(colors, feats, loc='upper right')
        else:
            ax.scatter(data[0], data[1],  data[2], c=color)
    else:
        ax.scatter(data[0], data[1], data[2])

    if xrange is not None:
        ax.set_xlim(xrange)

    if yrange is not None:
        ax.set_ylim(yrange)

    if zrange is not None:
        ax.set_ylim(zrange)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    if title is None:
        title = "3D PCA"

    ax.set_title(title)

    if save:
        if save_path is None:
            save_path = "PCA.png"
        plt.savefig(save_path+title.replace(" ", "_")+".png", dpi=300)
    if show:
        plt.show()
