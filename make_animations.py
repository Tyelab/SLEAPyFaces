import multiprocessing as mp
from multiprocessing import Pool
from make_video_graph_mp4 import write_chart_to_file_wrapper
import os
import pickle

total_parts = 6

if __name__ == '__main__':
    #spawn is critical to not share plt across threads.
    with open(os.path.join(os.getcwd(), ".ignore/all_data.pickle"), "rb") as f:
        features = pickle.load(f)

    weeks = features[("Classes", "Week")].unique().tolist()
    mp.set_start_method('spawn')
    with Pool() as p:
        weeks = features[("Classes", "Week")].unique().tolist()
        print(p.map(write_chart_to_file_wrapper, [[week, features] for week in weeks]))
