import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import os

def write_chart_to_file(week, features):

    def gen_all_data(n, week) -> list[pd.DataFrame]:
        x = 0
        data = []
        color = []
        while x < features[("Classes", "Trial_index")].max():
            data.append(pd.concat([features.loc[(features[("Classes", "Week")] == week)].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC1")], features.loc[(features[("Classes", "Week")] == week)].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC2")], features.loc[(features[("Classes", "Week")] == week)].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC3")]], axis=1).T.to_numpy())
            col = features.loc[(features[("Classes", "Week")] == week)].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("Classes", "Mouse_type")].to_numpy()
            col[col == "Control"] = 0
            col[col == "Susceptible"] = 1
            col[col == "Resilient"] = 2
            col = np.array(col, dtype=np.int)
            color.append(col)
            x += (features[("Classes", "Trial_index")].max()/n)
        return data, color

    N=100

    data, colors = gen_all_data(N, week)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    scatter = ax.scatter(xs=[], ys=[], zs=[], s=4, c=[])

    def init():
        print(week)
        ax.set(xlim3d=(-50, 190), xlabel='PC1')
        ax.set(ylim3d=(-40, 110), ylabel='PC2')
        ax.set(zlim3d=(-15, 235), zlabel='PC3')
        scatter.set_array(colors[0])
        scatter.set_offsets(np.array([data[0][0], data[0][1]]).T)
        scatter.set_3d_properties(data[0][2], 'z')
        return scatter,


    def update(frame):
        print(frame)
        ax.set(xlim3d=(-50, 190), xlabel='PC1')
        ax.set(ylim3d=(-40, 110), ylabel='PC2')
        ax.set(zlim3d=(-15, 235), zlabel='PC3')
        scatter.set_array(colors[frame])
        scatter.set_offsets(np.array([data[frame][0], data[frame][1]]).T)
        scatter.set_3d_properties(data[frame][2], 'z')
        return scatter,

    ani = FuncAnimation(
            fig=fig,
            func=update,
            init_func=init,
            interval=24,
            blit=True)

    ani.save(os.path.join(os.getcwd(), f".ignore/graphs/PCA3d-AllMice-Matplotlib-nolines-Color=Mouse_type-{week}.mp4"))

def write_chart_to_file_wrapper(args):
    return write_chart_to_file(*args)
