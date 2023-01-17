import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import os

def write_chart_to_file(week, features):

    def gen_all_data(n, week) -> list[pd.DataFrame]:
        x = 0
        ctrlData = []
        resData = []
        susData = []
        while x < features[("Classes", "Trial_index")].max():
            ctrlData.append(pd.concat([features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Control")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC1")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Control")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC2")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Control")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC3")]], axis=1).T.to_numpy())
            resData.append(pd.concat([features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Resilient")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC1")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Resilient")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC2")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Resilient")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC3")]], axis=1).T.to_numpy())
            susData.append(pd.concat([features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Susceptible")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC1")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Susceptible")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC2")], features.loc[(features[("Classes", "Week")] == week) & (features[("Classes", "Mouse_type")] == "Susceptible")].loc[features[("Classes", "Trial_index")] <= x].loc[:, ("PCA-3d", "PC3")]], axis=1).T.to_numpy())
            x += (features[("Classes", "Trial_index")].max()/n)
        return ctrlData, resData, susData

    N = 360

    ctrlData, resData, susData = gen_all_data(N, week)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ctrlScatter = ax.scatter(xs=[], ys=[], zs=[], s=4, c="purple")
    resScatter = ax.scatter(xs=[], ys=[], zs=[], s=4, c="green")
    susScatter = ax.scatter(xs=[], ys=[], zs=[], s=4, c="yellow")

    fig.legend((ctrlScatter, resScatter, susScatter), ('Control', 'Resilient', 'Susceptible'), loc='upper right')

    def init():
        print(week)
        ax.set(xlim3d=(-50, 190), xlabel='PC1')
        ax.set(ylim3d=(-40, 110), ylabel='PC2')
        ax.set(zlim3d=(-15, 235), zlabel='PC3')
        ctrlScatter.set_offsets(np.array([ctrlData[0][0], ctrlData[0][1]]).T)
        ctrlScatter.set_3d_properties(ctrlData[0][2], 'z')
        ctrlScatter.set(color="purple")
        resScatter.set_offsets(np.array([resData[0][0], resData[0][1]]).T)
        resScatter.set_3d_properties(resData[0][2], 'z')
        resScatter.set(color="green")
        susScatter.set_offsets(np.array([susData[0][0], susData[0][1]]).T)
        susScatter.set_3d_properties(susData[0][2], 'z')
        susScatter.set(color="yellow")
        fig.legend((ctrlScatter, resScatter, susScatter), ('Control', 'Resilient', 'Susceptible'), loc='upper right')
        return ctrlScatter, resScatter, susScatter

    def update(frame):
        angle = (frame-N) * 2
        print(frame)
        if frame < N:
            ax.set(xlim3d=(-50, 190), xlabel='PC1')
            ax.set(ylim3d=(-40, 110), ylabel='PC2')
            ax.set(zlim3d=(-15, 235), zlabel='PC3')
            ctrlScatter.set_offsets(np.array([ctrlData[frame][0], ctrlData[frame][1]]).T)
            ctrlScatter.set_3d_properties(ctrlData[frame][2], 'z')
            ctrlScatter.set(color="purple")
            resScatter.set_offsets(np.array([resData[frame][0], resData[frame][1]]).T)
            resScatter.set_3d_properties(resData[frame][2], 'z')
            resScatter.set(color="green")
            susScatter.set_offsets(np.array([susData[frame][0], susData[frame][1]]).T)
            susScatter.set_3d_properties(susData[frame][2], 'z')
            susScatter.set(color="yellow")
            fig.legend((ctrlScatter, resScatter, susScatter), ('Control', 'Resilient', 'Susceptible'), loc='upper right')
        else:
            ax.set(xlim3d=(-50, 190), xlabel='PC1')
            ax.set(ylim3d=(-40, 110), ylabel='PC2')
            ax.set(zlim3d=(-15, 235), zlabel='PC3')
            ctrlScatter.set_offsets(np.array([ctrlData[N-1][0], ctrlData[N-1][1]]).T)
            ctrlScatter.set_3d_properties(ctrlData[N-1][2], 'z')
            ctrlScatter.set(color="purple")
            resScatter.set_offsets(np.array([resData[N-1][0], resData[N-1][1]]).T)
            resScatter.set_3d_properties(resData[N-1][2], 'z')
            resScatter.set(color="green")
            susScatter.set_offsets(np.array([susData[N-1][0], susData[N-1][1]]).T)
            susScatter.set_3d_properties(susData[N-1][2], 'z')
            susScatter.set(color="yellow")
            fig.legend((ctrlScatter, resScatter, susScatter), ('Control', 'Resilient', 'Susceptible'), loc='upper right')
            # Normalize the angle to the range [-180, 180] for display
            angle_norm = (angle + 135) % 360 - 180

            # Update the axis view and title
            ax.view_init(azim=angle_norm)

        return ctrlScatter, resScatter, susScatter

    ani = FuncAnimation(
            fig=fig,
            func=update,
            init_func=init,
            interval=24,
            frames=540,
            blit=True)

    ani.save(os.path.join(os.getcwd(), f".ignore/graphs/PCA3d-AllMice-Matplotlib-nolines-Color=Mouse_type-{week}.mp4"))

def write_chart_to_file_wrapper(args):
    return write_chart_to_file(*args)
